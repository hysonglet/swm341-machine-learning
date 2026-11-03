//! 历程 0：板子、工具链、RTT 是否都通。
//!
//! 跑通这一个，后面的历程才有意义。它打印：
//! * 核心时钟（`hal::init` 把 HRC 拉到 150 MHz）；
//! * 一次两输入两输出的定点矩阵乘，确认 `embedded-nn` 真的被链进来了；
//! * 每秒一次心跳，确认 RTT 通道稳定。
//!
//! ```text
//! cargo run --bin 00_hello
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::hal;
use swm341_ai::nn::{self, *};

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 00_hello: SWM341 bring-up ====");

    let _p = hal::init(hal::Config::hrc_150mhz());
    rprintln!("core clock : {} Hz", hal::clock::sys_core_clock());

    // 链接检查：这几个符号来自纯 Rust 的 embedded-nn。
    rprintln!("nn runtime : embedded-nn 0.2.3 (pure Rust, #![no_std])");

    // 跑一个最小的全连接：y = W·x + b，量化参数取恒等，期望值一眼可算。
    let input: [i8; 2] = [3, 4];
    let filter: [i8; 2] = [10, 100]; // 单输出通道
    let bias: [i32; 1] = [-7];
    let mut out = [0i8; 1];

    let fc = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: ACT_S8,
    };
    let result = fully_connected_s8(
        &fc,
        &nn::identity_quant(),
        &Dims::new(1, 1, 1, 2),
        &input,
        &Dims::new(2, 1, 1, 1),
        &filter,
        Some(&bias),
        &Dims::new(1, 1, 1, 1),
        &mut out,
    );
    // 3*10 + 4*100 - 7 = 423 → int8 溢出被裁到 127。
    rprintln!("smoke fc   : {:?} -> {:?} (期望 [127])", result, out);

    rprintln!("RTT 通了，开始心跳：");

    let mut beat: u32 = 0;
    loop {
        beat += 1;
        rprintln!("  heartbeat {}", beat);
        // ~1 s @ 150 MHz。
        cortex_m::asm::delay(150_000_000);
    }
}
