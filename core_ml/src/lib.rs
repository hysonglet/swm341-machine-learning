//! 与硬件无关的纯 Rust 推理核心。
//!
//! 这里放的是"既要在 Mac 上验证、又要在 SWM341 上跑"的那部分代码：
//!
//! * [`preprocess`] —— 把一张任意尺寸的灰度图变成 28×28（背景 0、笔画 255），
//!   与 MNIST 一致。这是第 3 步（预处理）的板端实现，用纯整数运算写，无堆分配。
//! * [`Model`] —— 一个 int8 全连接（Softmax 回归）分类器，推理直接调用
//!   `embedded-nn` 的 `fully_connected_per_channel_s8`。
//!
//! 关键设计：**主机端和板端调用的是同一个 `fully_connected_per_channel_s8`**。
//! 只要主机端用 `Model::from_bytes` 加载权重、跑出正确率，板端结果必然一致
//! （前提是喂进来的 28×28 像素字节相同）。所以我们能在没有硬件的情况下，
//! 把整条链路量化误差验证干净。

#![no_std]

/// 28×28 摊平后的输入维度。
pub const N_PIXELS: usize = 28 * 28;
/// 类别数（0-9 十个数字）。
pub const N_CLASSES: usize = 10;

pub mod model;
pub mod preprocess;

pub use model::Model;
