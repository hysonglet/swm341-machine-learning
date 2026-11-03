//! SWM341 端侧机器学习实验场。
//!
//! 这是「用 Rust 在 SWM341 上做手写数字识别」的共享库，`src/bin/` 下的每个
//! 可执行文件是一个独立的小历程：
//!
//! * `00_hello`    —— RTT 打印 / 时钟 / 依赖是否通的冒烟测试；
//! * `01_nn_smoke` —— 定点算子库自检；
//! * `02_sdram`    —— SDRAM 容量与读写自检；
//! * `02_infer`    —— 加载 gen/model.bin，预处理 → int8 推理；
//! * `03_sd_image` —— SD 卡读 PNG → 解码 → 预处理 → int8 推理。
//!
//! 把公共代码放在 `lib` 里，历程之间就不用复制粘贴；每个 bin 仍然自带
//! `#[entry]` 与 panic handler，互不干扰。

#![no_std]

// PNG 解码、FAT 目录遍历都需要动态分配，而 64K SRAM 装不下，堆落在 SDRAM 上
// （见 `heap` 模块，必须在 SDRAM 上电之后 `init_heap`）。
extern crate alloc;

pub mod bench;
pub mod check;
pub mod heap;
pub mod image;
pub mod nn;
pub mod sd;
pub mod fat;

pub use swm341_hal as hal;

/// 与硬件无关的纯 Rust 推理核心：预处理 + int8 模型。
///
/// 板端和宿主机 `host/` 共用同一份代码、同一个 `embedded-nn` 算子，
/// 所以宿主机的量化验证通过，板端结果就必然一致。
pub use core_ml;
