//! int8 Softmax 回归分类器（单层全连接：784 → 10）。
//!
//! 权重由主机端 `host/` 工具训练并量化后，序列化成一段字节（`CML1` 格式），
//! 板端用 [`Model::from_bytes`] 原样加载，再用 `embedded-nn` 的
//! `fully_connected_per_channel_s8` 推理。因为主机端和板端调用的是**同一个算子**，
//! 量化误差可以完全在主机端验证。
//!
//! ## 量化约定（为什么这么定）
//!
//! 浮点模型：`z_k = Σ_j w[k][j]·x_j + b_k`，其中 `x_j = pixel/255 ∈ [0,1]`。
//! 部署时：
//! * 输入以 **int8 对称** 表示：`q_in[i] = pixel − 128`，配合 `input_offset = +128`，
//!   算子内部算的其实是 `Σ (pixel) · q_w[k][j]`；
//! * 权重 **per-channel 对称**：`q_w[k][j] = round(w[k][j] / S_w[k])`，
//!   `S_w[k] = max_j|w[k][j]| / 127`；
//! * 偏置并进 int32：`bias32[k] = round(b_k · 255 / S_w[k])`；
//! * 算子输出再做一次 per-channel 重量化，有效乘子 `S'_k = S_w[k] / (255 · S_out)`，
//!   其中 `S_out = max|z| / 127`（用训练集校准得到的全局最大值）。
//!
//! 这样 `fully_connected_per_channel_s8` 的输出 `q_logit[k] ≈ z_k / S_out`，
//! 于是 `argmax(q_logit) == argmax(z) == argmax(softmax(z))` ——
//! 也就是**量化后的预测与浮点预测完全一致**，softmax 只用于展示，不影晌判类。

use embedded_nn::{
    fully_connected_per_channel_s8, Activation, Dims, FcParams, PerChannelQuantParams,
};

use crate::{N_CLASSES, N_PIXELS};

/// 输入偏移：存 `pixel−128`，算子加回 `+128` 得到真实像素值。
const INPUT_OFFSET: i32 = 128;

/// 一段可直接 `include_bytes!` 进固件的量化模型权重。
pub struct Model {
    /// 行主序：`q_w[out_c * 784 + j]`，int8。
    pub q_w: [i8; N_CLASSES * N_PIXELS],
    /// 每类 int32 偏置。
    pub bias: [i32; N_CLASSES],
    /// per-channel 重量化乘子（Q31 定点）。
    pub mult: [i32; N_CLASSES],
    /// per-channel 重量化移位。
    pub shift: [i32; N_CLASSES],
}

impl Model {
    pub fn new(
        q_w: [i8; N_CLASSES * N_PIXELS],
        bias: [i32; N_CLASSES],
        mult: [i32; N_CLASSES],
        shift: [i32; N_CLASSES],
    ) -> Self {
        Self {
            q_w,
            bias,
            mult,
            shift,
        }
    }

    /// 从 `CML1` 字节流加载。格式见 [`Model::to_writer`] 的注释。
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut c = 0usize;
        let magic = b.get(c..c + 4)?;
        if magic != b"CML1" {
            return None;
        }
        c += 4;
        let n_classes = u32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?) as usize;
        c += 4;
        let input_dim = u32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?) as usize;
        c += 4;
        let _input_offset = i32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?);
        c += 4;
        let _s_out = f32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?);
        c += 4;
        if n_classes != N_CLASSES || input_dim != N_PIXELS {
            return None;
        }

        let mut q_w = [0i8; N_CLASSES * N_PIXELS];
        for x in q_w.iter_mut() {
            *x = *b.get(c)? as i8;
            c += 1;
        }
        let mut bias = [0i32; N_CLASSES];
        for x in bias.iter_mut() {
            *x = i32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?);
            c += 4;
        }
        let mut mult = [0i32; N_CLASSES];
        for x in mult.iter_mut() {
            *x = i32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?);
            c += 4;
        }
        let mut shift = [0i32; N_CLASSES];
        for x in shift.iter_mut() {
            *x = i32::from_le_bytes(b.get(c..c + 4)?.try_into().ok()?);
            c += 4;
        }
        Some(Self::new(q_w, bias, mult, shift))
    }

    /// 推理：返回 10 个 int8 logits（`q_logit[k] ≈ z_k / S_out`）。
    ///
    /// `img28` 是已经预处理好的 28×28 灰度图（背景 0、笔画 255）。
    pub fn infer(&self, img28: &[u8; N_PIXELS]) -> [i8; N_CLASSES] {
        // 输入转 int8：pixel − 128，配合算子的 input_offset=+128 还原成像素值。
        let mut input = [0i8; N_PIXELS];
        for i in 0..N_PIXELS {
            input[i] = (img28[i] as i32 - INPUT_OFFSET) as i8;
        }

        let mut out = [0i8; N_CLASSES];
        let fc = FcParams {
            input_offset: INPUT_OFFSET,
            filter_offset: 0,
            output_offset: 0,
            activation: Activation::int8_unconstrained(),
        };
        let qp = PerChannelQuantParams::new(&self.mult, &self.shift);
        let _ = fully_connected_per_channel_s8(
            &fc,
            &qp,
            &Dims::new(1, 1, 1, N_PIXELS as i32),
            &input,
            &Dims::new(N_PIXELS as i32, 1, 1, N_CLASSES as i32),
            &self.q_w,
            Some(&self.bias),
            &Dims::new(1, 1, 1, N_CLASSES as i32),
            &mut out,
        );
        out
    }

    /// 推理并直接返回预测类别（logits 的 argmax）。
    pub fn predict(&self, img28: &[u8; N_PIXELS]) -> usize {
        let logits = self.infer(img28);
        let mut best = 0usize;
        let mut best_v = logits[0] as i32;
        for k in 1..N_CLASSES {
            let v = logits[k] as i32;
            if v > best_v {
                best_v = v;
                best = k;
            }
        }
        best
    }
}

/// `CML1` 格式布局（全部小端）：
///
/// ```text
/// [4B "CML1"][4B n_classes][4B input_dim][4B input_offset(i32)]
/// [4B s_out(f32)][q_w: n_classes*input_dim 字节][bias: n_classes*4 字节]
/// [mult: n_classes*4 字节][shift: n_classes*4 字节]
/// ```
///
/// `emit` 每收到一段字节就写出去；无 `alloc` 也能用（直接推给 RTT/缓冲）。
pub fn write_model(
    mut emit: impl FnMut(&[u8]) -> Result<(), ()>,
    model: &Model,
    s_out: f32,
) -> Result<(), ()> {
    emit(b"CML1")?;
    emit(&(N_CLASSES as u32).to_le_bytes())?;
    emit(&(N_PIXELS as u32).to_le_bytes())?;
    emit(&INPUT_OFFSET.to_le_bytes())?;
    emit(&s_out.to_le_bytes())?;
    for &x in model.q_w.iter() {
        emit(&[x as u8])?;
    }
    for &x in model.bias.iter() {
        emit(&x.to_le_bytes())?;
    }
    for &x in model.mult.iter() {
        emit(&x.to_le_bytes())?;
    }
    for &x in model.shift.iter() {
        emit(&x.to_le_bytes())?;
    }
    Ok(())
}
