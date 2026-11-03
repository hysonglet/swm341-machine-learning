//! 历程 2：SDRAM 容量与读写自检（不含 LCD）。
//!
//! 目标只是把外置 SDRAM 跑通、量出大小、证明整片能正确读写。三个经典内存测试：
//!
//! 1. **数据总线** —— 16 根 DQ 线（walking ones / zeros）每根都能置 0/1；
//! 2. **地址总线** —— 地址线没有粘连、没有别名（写"除某根线外的所有 2 的幂地址为补码"，
//!    再验证它们不被打扰）；
//! 3. **整片 March** —— 用 LFSR 伪随机填满整片窗口再逐字回读比对，这同时就是在"测大小"：
//!    整片 N 字节都能正确存取，才说明 N 字节的窗口是真实可用的。
//!
//! 日志全部走 RTT。
//!
//! ```text
//! cargo run --bin 02_sdram
//! ```
//!
//! 关键坑（来自实际踩雷）：150 MHz 主频下 SDRAM 时钟**必须**用 `ClkDiv::Div2`，
//! 否则（默认的 `Div1` 只支持到 140 MHz）读写全是垃圾。`Config::default()` 用的
//! 正是 `Div1`，所以这里显式覆盖。

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::check::Checker;
use swm341_ai::hal;
use swm341_ai::hal::sdramc::{ClkDiv, Config, SdramPins, Sdramc, Size};

/// 板载 SDRAM 容量。SimplSDRAM 参考设计为 8 MB。
///
/// 控制器的地址译码由这里决定，**配错会让整片读写错乱**（列/行位宽不对），
/// 所以先按板子实际容量配。若你的板子不是 8 MB，改这个常量即可。
const CONFIGURED_SIZE: Size = Size::Mb8;

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 02_sdram: SDRAM 容量与读写自检 ====");

    let p = hal::init(hal::Config::hrc_150mhz());
    let core_hz = hal::clock::sys_core_clock();
    rprintln!("core clock : {} MHz", core_hz / 1_000_000);

    // 150 MHz 主频必须用 Div2，否则 SDRAM 读写是垃圾。
    let config = Config::default()
        .size(CONFIGURED_SIZE)
        .clk_div(ClkDiv::Div2);

    let sdram = Sdramc::new_with_pins(p.sdramc(), SdramPins::default(), config);

    let base = sdram.base_address();
    let size = sdram.size_bytes();
    let sdram_hz = core_hz / (config.clk_div as u32 + 1);
    rprintln!(
        "SDRAM 上线: {:>9} bytes ({:.1} MB) @ 0x{:08X}, clock {} MHz",
        size,
        size as f32 / (1024.0 * 1024.0),
        base,
        sdram_hz / 1_000_000
    );

    let mut check = Checker::new();

    check.section("数据总线 walking ones / zeros");
    let ok = unsafe { data_bus(base as *mut u16) };
    check.expect("16 位数据总线", ok);

    check.section("地址总线");
    let ok = unsafe { address_bus(base as *mut u16, size / 2) };
    check.expect("地址线无粘连 / 无别名", ok);

    check.section("整片读写（LFSR March）");
    // 这就是"测大小"：整片窗口 N 字节全部能正确存取才说明 N 字节真实可用。
    let errors = unsafe { march(base as *mut u32, size / 4) };
    check.expect_i32("误码字数（应为 0）", errors, 0);
    rprintln!("  整片 {} 遍填充 + 回读，误码 {} 字", 1, errors);

    check.section("容量一致性");
    // 窗口内半区不应与 0 地址互相别名：配错容量（小芯片配大窗口）会在这里暴露。
    let consistent = unsafe { no_internal_alias(base as *mut u32, size) };
    check.expect("base 与 base+size/2 不互相别名", consistent);

    if check.summary() {
        rprintln!("SDRAM 自检全部通过。");
    } else {
        rprintln!("SDRAM 自检有失败项，先排查硬件 / 容量配置。");
    }

    loop {
        cortex_m::asm::wfi();
    }
}

/// 数据总线测试：逐位拉 1、拉 0，回读必须一致。
unsafe fn data_bus(base: *mut u16) -> bool {
    for i in 0..16u32 {
        let one = (1u16 << i) as u16;
        let zero = !one;
        base.write_volatile(one);
        if base.read_volatile() != one {
            return false;
        }
        base.write_volatile(zero);
        if base.read_volatile() != zero {
            return false;
        }
    }
    true
}

/// 地址总线测试（Michael Barr 经典算法）：验证地址线没有粘连 / 别名。
unsafe fn address_bus(base: *mut u16, nwords: usize) -> bool {
    // 1) 先全部写 AAAA。
    for i in 0..nwords {
        base.add(i).write_volatile(0xAAAA);
    }
    // 2) 逐根地址线：把"除 2^i 外所有 2 的幂地址"写成补码，再校验它们不被打扰。
    let mut offset = 1usize;
    while offset < nwords {
        let pat = 0x55AAu16;
        let mut j = 1usize;
        while j < nwords {
            if j != offset {
                base.add(j).write_volatile(0x5555);
            }
            j <<= 1;
        }
        base.add(offset).write_volatile(pat);
        let mut j = 1usize;
        while j < nwords {
            if j != offset && base.add(j).read_volatile() != 0x5555 {
                return false;
            }
            j <<= 1;
        }
        if base.add(offset).read_volatile() != pat {
            return false;
        }
        base.add(offset).write_volatile(0xAAAA);
        offset <<= 1;
    }
    true
}

/// 整片 March：LFSR 伪随机填满，再逐字回读比对，返回误码字数。
unsafe fn march(base: *mut u32, nwords: usize) -> i32 {
    let mut lfsr = 0x1A2B_3C4D_u32;
    for i in 0..nwords {
        lfsr = xorshift32(lfsr);
        base.add(i).write_volatile(lfsr);
    }
    lfsr = 0x1A2B_3C4D_u32;
    let mut errors = 0i32;
    for i in 0..nwords {
        lfsr = xorshift32(lfsr);
        if base.add(i).read_volatile() != lfsr {
            errors += 1;
        }
    }
    errors
}

/// 容量合理性：写 base=0x1111_1111、半区=0x2222_2222，读回都应不变（无别名）。
unsafe fn no_internal_alias(base: *mut u32, size: usize) -> bool {
    let half_words = (size / 2) as isize / 4; // 半字节偏移，按 u32 元素计
    base.write_volatile(0x1111_1111);
    let mid = base.offset(half_words);
    mid.write_volatile(0x2222_2222);
    base.read_volatile() == 0x1111_1111 && mid.read_volatile() == 0x2222_2222
}

#[inline]
fn xorshift32(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}
