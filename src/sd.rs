//! SD 卡封装：在 `swm341_hal` 的 SDIO 驱动之上，提供 `BlockDevice` 接口给 FAT 层用。
//!
//! 注意：HAL 的 SDIO 驱动里 `Instance` trait 是私有的，外部 crate 不能写 `T: Instance`。
//! 而整颗 SWM341 只有一个 SDIO 外设，所以这里直接用具体设备类型 `Sdio<'d, SdioDevice>`，
//! 不再做泛型参数化——`Sdio` 自身的 `T: Instance` 约束由 HAL 内部的公开 impl 满足。
//!
//! 初始化时序严格照搬 `swm341-hal/examples/sdio_block.rs`（已在真机验证）：
//! CMD0 → CMD8 → CMD55+ACMD41(轮询) → CMD2 → CMD3 → CMD9 → 提速时钟 → CMD7 →
//! CMD55+ACMD6(4-bit) → CMD16 → CMD6(高速，可选)。

use cortex_m::asm::delay;

use crate::hal::sdio::{BusWidth, Config, ResponseType, Sdio};
use crate::hal::Peripheral;
use crate::hal::Sdio as SdioDevice;

/// SD 卡 / 块设备错误。
#[derive(Debug, Clone, Copy)]
pub enum SdCardError {
    /// 卡初始化 / 命令失败。
    CardError,
    /// 块读写失败。
    BlockDevice,
    /// 不支持的卡（非 SD v2、或初始化超时）。
    Unsupported,
}

/// 极简块设备接口（FAT 层只依赖这两个方法和容量）。
pub trait BlockDevice {
    /// 从 `block` 开始读 `count` 个 512 字节扇区到 `buf`。
    fn read_blocks(&mut self, buf: &mut [u8], block: u32, count: u32) -> Result<(), SdCardError>;
    /// 把 `buf` 里 `count` 个 512 字节扇区写到 `block` 开始的位置。
    fn write_blocks(&mut self, buf: &[u8], block: u32, count: u32) -> Result<(), SdCardError>;
    /// 卡的总扇区数（512 字节/扇区）。
    fn num_blocks(&self) -> u32;
}

/// 一张已枚举好的 SD 卡。
pub struct SdCard<'d> {
    sd: Sdio<'d, SdioDevice>,
    high_capacity: bool,
    num_blocks: u32,
}

impl<'d> SdCard<'d> {
    /// 上电、识别、进入 4-bit 传输模式。
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        peri: impl Peripheral<P = SdioDevice> + 'd,
        clk: impl Peripheral<P = crate::hal::gpio::PM2> + 'd,
        cmd: impl Peripheral<P = crate::hal::gpio::PM4> + 'd,
        d0: impl Peripheral<P = crate::hal::gpio::PM5> + 'd,
        d1: impl Peripheral<P = crate::hal::gpio::PM6> + 'd,
        d2: impl Peripheral<P = crate::hal::gpio::PN0> + 'd,
        d3: impl Peripheral<P = crate::hal::gpio::PN1> + 'd,
    ) -> Result<Self, SdCardError> {
        // 控制器 + pads 配置：1-bit 识别模式，低速时钟（ident clock，/256 ≈ 146 kHz）。
        // `new_4bit` 只做硬件/mux 配置，不碰卡片，所以它返回 `Self` 而非 `Result`。
        let sd = Sdio::new_4bit(
            peri,
            clk,
            cmd,
            d0,
            d1,
            d2,
            d3,
            Config::default().clock_divider(0xFF),
        );

        // 卡片上电时序：先让时钟跑一会再发第一条命令（~0.1s）。
        delay(15_000_000);

        // CMD0：GO_IDLE_STATE（无响应）。部分卡需要多敲几下。
        for _ in 0..2 {
            sd.send_command(0, 0, ResponseType::None, false, false)
                .map_err(|_| SdCardError::CardError)?;
        }

        // CMD8：检查电压窗口（0x1AA = 2.7-3.6V + 检查模式）。非 v2 卡不支持。
        let r = sd
            .send_command(8, 0x1AA, ResponseType::Short, false, false)
            .map_err(|_| SdCardError::CardError)?;
        if r[0] != 0x1AA {
            return Err(SdCardError::Unsupported); // 只支持 SD v2（SDHC/SDXC/SDSC v2）
        }

        // ACMD41 轮询直到卡 ready（OCR 的 bit31 = 上电完成；bit30 = HCS 表示 high-capacity）。
        let (_, high_capacity) = loop {
            // APP_CMD 先行，再发 ACMD41（与示例一致：HCS + 3.4-3.5V 窗口）。
            sd.send_command(55, 0, ResponseType::Short, false, false)
                .map_err(|_| SdCardError::CardError)?;
            let r = sd
                .send_command(41, 0x8010_0000 | 0x4000_0000, ResponseType::Short, false, false)
                .map_err(|_| SdCardError::CardError)?;
            let o = r[0];
            if o & (1 << 31) != 0 {
                break (o, (o & (1 << 30)) != 0);
            }
        };

        // ALL_SEND_CID（R2，取卡标识，忽略内容）。
        sd.send_command(2, 0, ResponseType::Long, false, false)
            .map_err(|_| SdCardError::CardError)?;
        // SEND_RCA（R6）：拿到卡的相对地址（在高 16 位）。
        let r = sd
            .send_command(3, 0, ResponseType::Short, false, false)
            .map_err(|_| SdCardError::CardError)?;
        let rca = r[0] >> 16;

        // 读 CSD（R2，取容量）。命令无数据相位，识别时钟即可。
        let csd = sd
            .send_command(9, (rca as u32) << 16, ResponseType::Long, false, false)
            .map_err(|_| SdCardError::CardError)?;
        let num_blocks = csd_capacity_blocks(&csd).unwrap_or(0);

        // 提速到 ~18 MHz（数据相位不能跑识别时钟，否则超时）。
        sd.configure_clock(2);

        // CMD7 选中卡（带 busy 响应）。
        sd.send_command(7, (rca as u32) << 16, ResponseType::Busy, false, false)
            .map_err(|_| SdCardError::CardError)?;

        // CMD55 + ACMD6：把卡切到 4-bit 总线，再把控制器总线也加宽。
        sd.send_command(55, (rca as u32) << 16, ResponseType::Short, false, false)
            .map_err(|_| SdCardError::CardError)?;
        sd.send_command(6, 0x02, ResponseType::Short, false, false)
            .map_err(|_| SdCardError::CardError)?;
        let _ = sd.set_bus_width(BusWidth::Bus4);

        // CMD16：固定 512 字节块长度（SDHC 也是 512，发一次保险）。
        sd.send_command(16, 512, ResponseType::Short, false, false)
            .map_err(|_| SdCardError::CardError)?;

        // CMD6 高速模式（仅 SDHC/SDXC，可选）：读 64 字节状态，若时序组报告 function 1，
        // 则提到 ~37.5 MHz。SDSC 不支持 switch，跳过（否则 addr() 会把切换参数当成块号乘 512）。
        if high_capacity {
            let mut hs = [0u32; 16];
            if sd
                .read_blocks(6, 0x80FFFF01, &mut hs, 64, 1)
                .is_ok()
                && (hs[4] & 0xF) == 1
            {
                sd.configure_clock(1);
            }
        }

        Ok(Self {
            sd,
            high_capacity,
            num_blocks,
        })
    }

    /// 总扇区数（512 字节/扇区）。
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// 把逻辑块号翻译成 CMD17/CMD24 的地址参数：SDHC/SDXC 直接用块号，SDSC 用字节偏移。
    fn addr(&self, block: u32) -> u32 {
        if self.high_capacity {
            block
        } else {
            block * 512
        }
    }
}

impl<'d> BlockDevice for SdCard<'d> {
    fn read_blocks(&mut self, buf: &mut [u8], block: u32, count: u32) -> Result<(), SdCardError> {
        let mut wbuf = [0u32; 128];
        for i in 0..count {
            let addr = self.addr(block + i);
            self.sd
                .read_blocks(17, addr, &mut wbuf, 512, 1)
                .map_err(|_| SdCardError::BlockDevice)?;
            for j in 0..128 {
                let off = j * 4;
                buf[off..off + 4].copy_from_slice(&wbuf[j].to_le_bytes());
            }
        }
        Ok(())
    }

    fn write_blocks(&mut self, buf: &[u8], block: u32, count: u32) -> Result<(), SdCardError> {
        let mut wbuf = [0u32; 128];
        for i in 0..count {
            let addr = self.addr(block + i);
            for j in 0..128 {
                let off = j * 4;
                wbuf[j] = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            }
            self.sd
                .write_blocks(24, addr, &mut wbuf, 512, 1)
                .map_err(|_| SdCardError::BlockDevice)?;
        }
        Ok(())
    }

    fn num_blocks(&self) -> u32 {
        self.num_blocks
    }
}

/// 从 CMD9 的 CSD 响应（R2，4×u32，每个字大端）解析总扇区数（512 字节/扇区）。
///
/// SDIO 控制器把 128 位 CSD 按 RESP[0]=CSD[127:96] … RESP[3]=CSD[31:0] 存成 4 个 32 位字，
/// 每个字大端。先还原成 16 字节大端位流，再按 SD 规范取位域，兼容 CSD v1（SDSC）和 v2（SDHC/SDXC）。
fn csd_capacity_blocks(csd: &[u32; 4]) -> Option<u32> {
    // 还原 16 字节大端字节流：CSD[127:120] 在 b[0] 的最高位。
    let mut b = [0u8; 16];
    for i in 0..4 {
        let w = csd[i];
        b[i * 4] = (w >> 24) as u8;
        b[i * 4 + 1] = (w >> 16) as u8;
        b[i * 4 + 2] = (w >> 8) as u8;
        b[i * 4 + 3] = w as u8;
    }
    // CSD 位位置 p（127=MSB）。p 落在字节 bi=(127-p)/8，字节内高位 bn=7-((127-p)%8)。
    let bit = |p: u32| -> u32 {
        let bi = ((127 - p) / 8) as usize;
        let bn = (7 - ((127 - p) % 8)) as u32;
        u32::from((b[bi] >> bn) & 1)
    };

    // CSD_STRUCTURE = bits[127:126]。
    let csd_structure = (bit(127) << 1) | bit(126);
    if csd_structure == 1 {
        // CSD v2：C_SIZE = bits[69:48]（22 位），容量 = (C_SIZE+1) * 512K 字节。
        let mut c_size = 0u32;
        for k in 0..22 {
            c_size = (c_size << 1) | bit(69 - k);
        }
        Some((c_size + 1) * 1024)
    } else if csd_structure == 0 {
        // CSD v1：C_SIZE = bits[73:62]（12 位），C_SIZE_MULT = bits[49:47]（3 位），READ_BL_LEN = bits[83:80]（4 位）。
        let mut c_size = 0u32;
        for k in 0..12 {
            c_size = (c_size << 1) | bit(73 - k);
        }
        let mut mult = 0u32;
        for k in 0..3 {
            mult = (mult << 1) | bit(49 - k);
        }
        let read_bl_len = (bit(83) << 3) | (bit(82) << 2) | (bit(81) << 1) | bit(80);
        let block_len = 1u32 << read_bl_len;
        let block_count = (c_size + 1) * (1u32 << (mult + 2));
        Some((block_count * block_len) / 512)
    } else {
        None
    }
}
