//! SD 卡封装：在 `swm341_hal` 的 SDIO 驱动之上，提供 `hadris-fat` 需要的
//! `embedded_io::Read` + `embedded_io::Seek` 块设备抽象。
//!
//! 注意：HAL 的 SDIO 驱动里 `Instance` trait 是私有的，外部 crate 不能写 `T: Instance`。
//! 而整颗 SWM341 只有一个 SDIO 外设，所以这里直接用具体设备类型 `Sdio<'d, SdioDevice>`，
//! 不再做泛型参数化——`Sdio` 自身的 `T: Instance` 约束由 HAL 内部的公开 impl 满足。
//!
//! 初始化时序严格照搬 `swm341-hal/examples/sdio_block.rs`（已在真机验证）：
//! CMD0 → CMD8 → CMD55+ACMD41(轮询) → CMD2 → CMD3 → CMD9 → 提速时钟 → CMD7 →
//! CMD55+ACMD6(4-bit) → CMD16 → CMD6(高速，可选)。

use cortex_m::asm::delay;

use embedded_io::{ErrorKind, ErrorType, Read, Seek, SeekFrom};
use rtt_target::rprintln;

use crate::hal::pac::SDIO;
use crate::hal::sdio::{BusWidth, Config, ResponseType, Sdio};
use crate::hal::Peripheral;
use crate::hal::Sdio as SdioDevice;

/// 打印 SDIO 控制器寄存器，用于上电诊断（照搬 `swm341-hal/examples/sdio_block.rs`）。
/// 重点看 `clkrdy`（时钟是否起稳）、`pwron`（上电）、`cardinst`/`hascardst`（卡是否在位）。
pub fn dump_sdio() {
    let regs = unsafe { &*SDIO::PTR };
    let cr1 = regs.cr1.read();
    let cr2 = regs.cr2.read();
    let stat = regs.stat.read().bits();
    let iflags = regs.if_.read().bits();
    rprintln!(
        "[SD] CR1: pwron={} 4bit={} 8bit={} cdsrc={} cdbit={}",
        cr1.pwron().bit(),
        cr1._4bit().bit(),
        cr1._8bit().bit(),
        cr1.cdsrc().bit(),
        cr1.cdbit().bit()
    );
    rprintln!(
        "[SD] CR2: clken={} sdclken={} sdclkdiv={} clkrdy={}",
        cr2.clken().bit(),
        cr2.sdclken().bit(),
        cr2.sdclkdiv().bits(),
        cr2.clkrdy().bit()
    );
    rprintln!(
        "[SD] STAT: cmdlinest={} cmdinhi={} datlinest={} cardinst={} hascardst={} (raw=0x{:08X})",
        stat & 1,
        (stat >> 1) & 1,
        (stat >> 2) & 1,
        (stat >> 16) & 1,
        (stat >> 18) & 1,
        stat
    );
    rprintln!("[SD] IF (raw): 0x{:08X}", iflags);
}

/// SD 卡 / 块设备错误。同时实现 `embedded_io::Error`，作为 `hadris-fat` 的 I/O 错误类型。
#[derive(Debug, Clone, Copy)]
pub enum SdCardError {
    /// 卡初始化 / 命令失败。
    CardError,
    /// 块读写失败。
    BlockDevice,
    /// 不支持的卡（非 SD v2、或初始化超时）。
    Unsupported,
}

impl embedded_io::Error for SdCardError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Other
    }
}

impl core::fmt::Display for SdCardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            SdCardError::CardError => "card init/command failed",
            SdCardError::BlockDevice => "block read/write failed",
            SdCardError::Unsupported => "unsupported card (need SD v2)",
        })
    }
}

// `embedded_io::Error` 在 0.7 要求 `core::error::Error`，补上即可（`no_std` 下也成立）。
impl core::error::Error for SdCardError {}

/// `embedded_io` 0.7 用 `ErrorType` 携带 `Error` 关联类型（`Read`/`Seek` 都依赖它）。
impl<'d> ErrorType for SdCard<'d> {
    type Error = SdCardError;
}

/// 一张已枚举好的 SD 卡。
///
/// 对外表现为一个可 `Seek` 的字节流：当前位置 `pos` 由 `Read`/`Seek` 维护，
/// `hadris-fat` 通过 `seek` + `read` 在磁盘上随机访问 BPB、FAT 表与数据簇。
pub struct SdCard<'d> {
    sd: Sdio<'d, SdioDevice>,
    high_capacity: bool,
    num_blocks: u32,
    /// 分区偏移：若卡带 MBR 分区表（如 macOS 格式化的 `diskNs1`），FAT 卷引导扇区在分区起始
    /// LBA，所有逻辑块号都需加上此偏移；无分区（superfloppy）时为 0。
    part_offset: u32,
    /// 当前字节位置（扇区大小恒为 512）。
    pos: u64,
}

/// 发命令并在失败时打印具体 CMD 编号与 `SdioError`（便于定位上电卡在哪一步）。
fn send(
    sd: &Sdio<'_, SdioDevice>,
    idx: u8,
    arg: u32,
    resp: ResponseType,
    data: bool,
    read: bool,
) -> Result<[u32; 4], SdCardError> {
    sd.send_command(idx, arg, resp, data, read).map_err(|e| {
        rprintln!("[SD] CMD{} 失败：{:?}", idx, e);
        SdCardError::CardError
    })
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
        rprintln!("[SD] SDIO host up (new_4bit)");
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
        // 上电诊断：确认 SDIO 时钟已起稳、卡在位。
        dump_sdio();

        // 卡片上电时序：先让时钟跑一会再发第一条命令（~0.1s）。
        delay(15_000_000);

        // CMD0：GO_IDLE_STATE（无响应）。部分卡需要多敲几下。
        for _ in 0..2 {
            if send(&sd, 0, 0, ResponseType::None, false, false).is_err() {
                rprintln!("[SD] CMD0 失败：卡可能未在位 / SDIO 时钟未起稳");
                return Err(SdCardError::CardError);
            }
        }
        rprintln!("[SD] CMD0 OK");

        // CMD8：检查电压窗口（0x1AA = 2.7-3.6V + 检查模式）。非 v2 卡不支持。
        let r = send(&sd, 8, 0x1AA, ResponseType::Short, false, false)?;
        if r[0] != 0x1AA {
            rprintln!("[SD] CMD8 返回 0x{:08X}，非 SD v2 卡", r[0]);
            return Err(SdCardError::Unsupported); // 只支持 SD v2（SDHC/SDXC/SDSC v2）
        }
        rprintln!("[SD] CMD8 OK: SD v2");

        // ACMD41 轮询直到卡 ready（OCR 的 bit31 = 上电完成；bit30 = HCS 表示 high-capacity）。
        // 必须加迭代上限：否则卡若不就绪会无线死循环（正是之前“卡住”的根因）。
        let mut high_capacity = false;
        let mut ocr = 0u32;
        for _ in 0..2000 {
            send(&sd, 55, 0, ResponseType::Short, false, false)?;
            let r = send(
                &sd,
                41,
                0x8010_0000 | 0x4000_0000,
                ResponseType::Short,
                false,
                false,
            )?;
            ocr = r[0];
            if ocr & (1 << 31) != 0 {
                high_capacity = (ocr & (1 << 30)) != 0;
                break;
            }
        }
        if ocr & (1 << 31) == 0 {
            rprintln!("[SD] ACMD41 超时：卡未就绪 (OCR=0x{:08X})", ocr);
            return Err(SdCardError::Unsupported);
        }
        rprintln!(
            "[SD] ACMD41 OK: {} ({})",
            if high_capacity { "SDHC/SDXC" } else { "SDSC" },
            ocr
        );

        // ALL_SEND_CID（R2，取卡标识，忽略内容）。
        send(&sd, 2, 0, ResponseType::Long, false, false)?;
        // SEND_RCA（R6）：拿到卡的相对地址（在高 16 位）。
        let r = send(&sd, 3, 0, ResponseType::Short, false, false)?;
        let rca = r[0] >> 16;
        rprintln!("[SD] CMD2/CMD3 OK: RCA=0x{:04X}", rca);

        // 读 CSD（R2，取容量）。命令无数据相位，识别时钟即可。
        let csd = send(&sd, 9, (rca as u32) << 16, ResponseType::Long, false, false)?;
        rprintln!(
            "[SD] CSD raw: {:08X} {:08X} {:08X} {:08X}",
            csd[0], csd[1], csd[2], csd[3]
        );
        let num_blocks = csd_capacity_blocks(&csd, high_capacity).unwrap_or(0);
        rprintln!("[SD] CMD9 OK: {} 扇区", num_blocks);

        // MBR 分区偏移（在 CMD7 选中卡、进入 transfer 态后再读扇区 0 判断，见下方）。
        let mut part_offset = 0u32;

        // 提速到 ~18 MHz（数据相位不能跑识别时钟，否则超时）。
        sd.configure_clock(2);
        rprintln!("[SD] 时钟提速到 ~18 MHz");

        // CMD7 选中卡（带 busy 响应）。
        send(&sd, 7, (rca as u32) << 16, ResponseType::Busy, false, false)?;
        rprintln!("[SD] CMD7 OK: 卡已选中");

        // 读扇区 0：判断是卷引导扇区还是 MBR 分区表。必须在 transfer 态（CMD7 之后）才能发
        // CMD17。带 MBR 分区（macOS 格式化的 `diskNs1`）时，FAT 卷引导扇区在分区起始 LBA，
        // hadris-fat 需要偏移才能找到 BPB；无分区（superfloppy）则偏移 0。
        {
            let mut mbr = [0u8; 512];
            let mut wbuf = [0u32; 128];
            if sd.read_blocks(17, 0, &mut wbuf, 512, 1).is_ok() {
                for j in 0..128 {
                    let off = j * 4;
                    mbr[off..off + 4].copy_from_slice(&wbuf[j].to_le_bytes());
                }
            }
            // 扇区 0 的 BPB 字节：bytes_per_sector 合法 ⇒ 它就是卷引导扇区（无分区）。
            let bps = u16::from_le_bytes([mbr[11], mbr[12]]);
            if bps == 512 || bps == 1024 || bps == 2048 || bps == 4096 {
                // superfloppy：偏移 0。
            } else if mbr[510] == 0x55 && mbr[511] == 0xAA {
                // 可能是 MBR：找第一个 FAT 类型分区，取其起始 LBA 作为偏移。
                for i in 0..4u32 {
                    let base = 446 + i as usize * 16;
                    let ptype = mbr[base + 4];
                    if matches!(ptype, 0x01 | 0x04 | 0x06 | 0x0B | 0x0C | 0x0E) {
                        let lba = u32::from_le_bytes([
                            mbr[base + 8],
                            mbr[base + 9],
                            mbr[base + 10],
                            mbr[base + 11],
                        ]);
                        if lba != 0 {
                            part_offset = lba;
                            break;
                        }
                    }
                }
            }
            rprintln!(
                "[SD] 分区偏移: {} 扇区（{}）",
                part_offset,
                if part_offset == 0 {
                    "无分区/卷引导扇区"
                } else {
                    "MBR 分区 1"
                }
            );
        }

        // CMD55 + ACMD6：把卡切到 4-bit 总线，再把控制器总线也加宽。
        send(&sd, 55, (rca as u32) << 16, ResponseType::Short, false, false)?;
        send(&sd, 6, 0x02, ResponseType::Short, false, false)?;
        let _ = sd.set_bus_width(BusWidth::Bus4);
        rprintln!("[SD] 4-bit 总线已切换");

        // CMD16：固定 512 字节块长度（SDHC 也是 512，发一次保险）。
        send(&sd, 16, 512, ResponseType::Short, false, false)?;
        rprintln!("[SD] CMD16 OK: 块长 512");

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
                rprintln!("[SD] 高速模式 ON: ~37.5 MHz");
            } else {
                rprintln!("[SD] 未支持高速模式，保持 ~18 MHz");
            }
        }

        rprintln!("[SD] 初始化完成");
        Ok(Self {
            sd,
            high_capacity,
            num_blocks,
            part_offset,
            pos: 0,
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

    /// 从 `block` 开始读 `count` 个 512 字节扇区到 `buf`（必须长度 ≥ count*512）。
    pub fn read_blocks(&mut self, buf: &mut [u8], block: u32, count: u32) -> Result<(), SdCardError> {
        for i in 0..count {
            let addr = self.addr(block + i);
            let base = (i as usize) * 512;
            let mut wbuf = [0u32; 128];
            self.sd
                .read_blocks(17, addr, &mut wbuf, 512, 1)
                .map_err(|_| SdCardError::BlockDevice)?;
            for j in 0..128 {
                let off = base + j * 4;
                buf[off..off + 4].copy_from_slice(&wbuf[j].to_le_bytes());
            }
        }
        Ok(())
    }

    /// 把 `buf`（长度 = count*512）里 `count` 个 512 字节扇区写到 `block` 开始的位置。
    pub fn write_blocks(&mut self, buf: &[u8], block: u32, count: u32) -> Result<(), SdCardError> {
        for i in 0..count {
            let addr = self.addr(block + i);
            let base = (i as usize) * 512;
            let mut wbuf = [0u32; 128];
            for j in 0..128 {
                let off = base + j * 4;
                wbuf[j] = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            }
            self.sd
                .write_blocks(24, addr, &mut wbuf, 512, 1)
                .map_err(|_| SdCardError::BlockDevice)?;
        }
        Ok(())
    }

    /// 从任意字节偏移 `pos` 读最多 `buf.len()` 字节（跨扇区），返回实际读取字节数。
    ///
    /// `hadris-fat` 总是先 `seek` 再 `read`，所以 `Read::read` 会带当前位置进来；这里单独
    /// 提供一个按绝对偏移读取的底层helper，便于对齐时一次搬整段连续扇区。
    fn read_at(&mut self, buf: &mut [u8], pos: u64) -> Result<usize, SdCardError> {
        let total = self.num_blocks as u64 * 512;
        let mut done = 0usize;
        while done < buf.len() {
            let cur = pos + done as u64;
            if cur >= total {
                break;
            }
            let off = (cur % 512) as usize;
            if off == 0 {
                // 扇区对齐：一次读整段连续扇区，避免逐扇区拷贝。
                // 用 `num_blocks`（u32）算剩余扇区，避免 `total`(u64) 在 32 位目标上转 usize 被截断。
                // 逻辑块号统一加上 MBR 分区偏移，使 hadris-fat 看到的是卷内地址。
                let block = (cur / 512) as u32 + self.part_offset;
                let max_sectors = ((self.num_blocks.saturating_sub(block)) as usize)
                    .min((buf.len() - done) / 512);
                if max_sectors > 0 {
                    let start = done;
                    let end = done + max_sectors * 512;
                    self.read_blocks(&mut buf[start..end], block, max_sectors as u32)?;
                    done += max_sectors * 512;
                    continue;
                }
            }
            // 尾部未对齐：读单扇区后只拷贝所需部分。
            let mut sector = [0u8; 512];
            self.read_blocks(&mut sector, (cur / 512) as u32 + self.part_offset, 1)?;
            let take = (512 - off).min(buf.len() - done);
            buf[done..done + take].copy_from_slice(&sector[off..off + take]);
            done += take;
        }
        Ok(done)
    }
}

/// `hadris-fat` 的块设备抽象：把 `SdCard` 当成一个可随机访问的字节流。
impl<'d> Read for SdCard<'d> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let n = self.read_at(buf, self.pos)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl<'d> Seek for SdCard<'d> {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        let total = self.num_blocks as u64 * 512;
        let new = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::Current(o) => self.pos as i64 + o,
            SeekFrom::End(o) => total as i64 + o,
        };
        let new = if new < 0 { 0 } else { new as u64 };
        self.pos = new;
        Ok(new)
    }
}

/// 从 CMD9 的 CSD 响应（R2，4×u32）解析总扇区数（512 字节/扇区）。
///
/// 不同芯片把 128 位 CSD 放进 4 个 32 位响应寄存器时，不仅**字序**可能相反（RESP[0] 是
/// MSB 还是 LSB），**每个字内的字节序**也可能相反（ARM 小端机器上寄存器常被按小端拆字节）。
/// 我们**已经**从 ACMD41 的 OCR 知道卡是 SDHC/SDXC（`high_capacity=true`）还是 SDSC，所以把
/// 这四种布局组合都试一遍，挑出 `CSD_STRUCTURE`（bits[127:126]，CSD v2 必为 1）与已知卡类型
/// 吻合的那种，从根本上避免猜字序。
fn csd_capacity_blocks(csd: &[u32; 4], high_capacity: bool) -> Option<u32> {
    let expected_struct = if high_capacity { 1u32 } else { 0u32 };
    for rev in [false, true] {
        for swap in [false, true] {
            // `rev`：把字序整体反转（RESP[0] 当成 LSB 字）；`swap`：每个 32 位字按小端拆字节。
            let raw: [u32; 4] = if rev {
                [csd[3], csd[2], csd[1], csd[0]]
            } else {
                [csd[0], csd[1], csd[2], csd[3]]
            };
            let words = if swap {
                [
                    raw[0].swap_bytes(),
                    raw[1].swap_bytes(),
                    raw[2].swap_bytes(),
                    raw[3].swap_bytes(),
                ]
            } else {
                raw
            };
            // 16 字节大端 CSD：words[0] 的字节在 b[0..3]（最高位），… words[3] 在 b[12..15]。
            let mut b = [0u8; 16];
            for i in 0..4 {
                let w = words[i];
                b[i * 4] = (w >> 24) as u8;
                b[i * 4 + 1] = (w >> 16) as u8;
                b[i * 4 + 2] = (w >> 8) as u8;
                b[i * 4 + 3] = w as u8;
            }
            // CSD 位位置 p（127=MSB）：落在字节 bi=(127-p)/8，字节内高位 bn=7-((127-p)%8)。
            let bit = |p: u32| -> u32 {
                let bi = ((127 - p) / 8) as usize;
                let bn = (7 - ((127 - p) % 8)) as u32;
                u32::from((b[bi] >> bn) & 1)
            };

            // CSD_STRUCTURE = bits[127:126]，CSD v2 必为 1。与已知卡类型一致才采用此布局。
            let csd_structure = (bit(127) << 1) | bit(126);
            rprintln!(
                "[SD]   CSD 候选 rev={} swap={} struct={}",
                rev,
                swap,
                csd_structure
            );
            if csd_structure != expected_struct {
                continue;
            }

            let blocks = if high_capacity {
                // CSD v2：C_SIZE = bits[69:48]（22 位），容量 = (C_SIZE+1) * 1024 个 512 字节块。
                let mut c_size = 0u32;
                for k in 0..22 {
                    c_size = (c_size << 1) | bit(69 - k);
                }
                (c_size + 1) * 1024
            } else {
                // CSD v1：C_SIZE=bits[73:62]（12 位），C_SIZE_MULT=bits[49:47]，READ_BL_LEN=bits[83:80]。
                let mut c_size = 0u32;
                for k in 0..12 {
                    c_size = (c_size << 1) | bit(73 - k);
                }
                let mut mult = 0u32;
                for k in 0..3 {
                    mult = (mult << 1) | bit(49 - k);
                }
                let read_bl_len =
                    (bit(83) << 3) | (bit(82) << 2) | (bit(81) << 1) | bit(80);
                let block_len = 1u32 << read_bl_len;
                let block_count = (c_size + 1) * (1u32 << (mult + 2));
                (block_count * block_len) / 512
            };
            rprintln!(
                "[SD] CSD 解析: 字序rev={} 字节swap={} 扇区={}",
                rev,
                swap,
                blocks
            );
            return Some(blocks);
        }
    }
    // 4 种布局都没匹配（极少见，可能是控制器把 136 位响应的 CRC 也塞进来导致整段 8 位偏移）。
    // 打印原始 CSD 寄存器值，并回退一个较大的默认容量（> 卡真实容量，只读挂载不会越界），
    // 让整条推理链路先能跑起来，再用原始值精确修正。
    rprintln!(
        "[SD] CSD 解析失败（4 种布局均非 v{}），回退默认 ~16GB 以便挂载",
        if high_capacity { 2 } else { 1 }
    );
    Some(0x0200_0000)
}
