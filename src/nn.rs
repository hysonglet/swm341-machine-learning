//! `embedded-nn` 的再导出层。
//!
//! 算子库本身是纯 Rust 的 [`embedded_nn`]（`#![no_std]`、零堆分配），API 与
//! Arm CMSIS-NN 一一对应：
//!
//! | CMSIS-NN (C)                    | embedded-nn (Rust)                  |
//! | ------------------------------- | ----------------------------------- |
//! | `arm_convolve_s8`               | [`convolve_per_channel_s8`]         |
//! | `arm_depthwise_conv_s8`         | [`depthwise_conv_per_channel_s8`]   |
//! | `arm_fully_connected_s8`        | [`fully_connected_s8`]              |
//! | `arm_max_pool_s8` / `arm_avgpool_s8` | [`max_pool_s8`] / [`avg_pool_s8`] |
//! | `arm_softmax_s8`                | [`softmax_s8`]                      |
//! | `arm_elementwise_add_s8`        | [`elementwise_add_s8`]              |
//! | `arm_relu_q7` / `arm_relu6_s8`  | [`relu_s8`] / [`relu6_s8`]          |
//!
//! 这里只做两件事：把代码里常用的类型/函数集中导出（方便 `use swm341_ai::nn::*;`），
//! 以及提供几个"量化参数取恒等"的常量，让自检用例的期望值可以手算。

pub use embedded_nn::activations::{relu6_s8, relu_s8, activation_s8};
pub use embedded_nn::support::{doubling_high_mult_no_sat, requantize};
pub use embedded_nn::{
    avg_pool_s8, convolve_per_channel_s8, depthwise_conv_per_channel_s8, elementwise_add_s8,
    elementwise_mul_s8, fully_connected_s8, max_pool_s8, softmax_s8, Activation, ConvParams,
    Dims, DwConvParams, ElementwiseAddParams, ElementwiseMulParams, Error, FcParams, Padding2D,
    PerChannelQuantParams, PerTensorQuantParams, PoolParams, Result, Tile,
};

/// Q31 里的 1.0。
///
/// `requantize(val, UNITY_MULTIPLIER, 0)` 对 `|val| < 2^30` 是恒等映射：
/// `(val * (2^31-1) + 2^30) >> 31 == val`。自检用例靠它把量化参数"关掉"，
/// 于是期望值就是朴素的整数结果，算错了能一眼看出来。
pub const UNITY_MULTIPLIER: i32 = 0x7FFF_FFFF;

/// int8 全范围激活（不裁剪）。
pub const ACT_S8: Activation = Activation::new(i8::MIN as i32, i8::MAX as i32);

/// 恒等的每张量量化参数（乘子 1.0、shift 0）。
pub const fn identity_quant() -> PerTensorQuantParams {
    PerTensorQuantParams::new(UNITY_MULTIPLIER, 0)
}

/// 从"输入 H/W、卷积核、stride、padding"直接算出输出 H/W（VALID/SAME 通用）。
pub const fn out_hw(in_h: i32, in_w: i32, k_h: i32, k_w: i32, stride: i32, padding: i32) -> (i32, i32) {
    (
        (in_h + 2 * padding - k_h) / stride + 1,
        (in_w + 2 * padding - k_w) / stride + 1,
    )
}
