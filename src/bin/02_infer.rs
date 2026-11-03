//! 历程 2：端侧手写数字识别（int8 推理）。
//!
//! 这是「训练 → 量化 → 部署」链路在 SWM341 上的终点：
//!
//! * 权重由 `host/` 工具训练并量化后，导成 `gen/model.bin`，这里 `include_bytes!` 进来；
//! * 几个测试样本（同由 `host/` 导出为 `gen/samples.bin`）模拟「摄像头采到的 28×28 图」；
//! * 用 `core_ml::Model`（内部调用 embedded-nn 的 `fully_connected_per_channel_s8`）
//!   跑 int8 推理 + `softmax_s8`，打印预测 / 真值 / 置信度，并用 DWT 统计每个数字的
//!   推理周期数与 cycles/MAC。
//!
//! 这一份 bin 和 host 端 `eval_int8` 跑的是**同一个算子**，所以只要 host 验证过精度，
//! 板上结果必然一致（前提是喂进来的 28×28 像素字节相同）。
//!
//! ```text
//! cargo run --release --bin 02_infer
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::bench::CycleMeter;
use swm341_ai::core_ml::{self, Model, N_CLASSES, N_PIXELS};

// 由 host/ 工具生成（先跑 `cd host && cargo run --release`）。
const MODEL_BIN: &[u8] = include_bytes!("../../gen/model.bin");
// 格式：每样本 = 1 字节 label + 784 字节图像。
const SAMPLES_BIN: &[u8] = include_bytes!("../../gen/samples.bin");

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 02_infer: 端侧手写数字识别（int8）====");

    let p = swm341_ai::hal::init(swm341_ai::hal::Config::hrc_150mhz());
    let core_hz = swm341_ai::hal::clock::sys_core_clock();
    let mut meter = CycleMeter::new(p.dwt());

    let model = match Model::from_bytes(MODEL_BIN) {
        Some(m) => m,
        None => {
            rprintln!("✗ 解析 model.bin 失败！先跑 `cd host && cargo run --release`");
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    rprintln!("模型加载成功：784 → 10 全连接（int8, per-channel）");

    // s_out 存于 model.bin 偏移 16 处：logits ≈ z / S_out，所以 z = logits * S_out，
    // 用它做真正的 softmax 才能得到正确概率（embedded-nn 的 softmax_s8 输出标度不标准）。
    let s_out = f32::from_le_bytes(MODEL_BIN[16..20].try_into().unwrap());

    // 解析样例。
    let (n_samples, samples) = parse_samples(SAMPLES_BIN);
    rprintln!("样例数：{} 张", n_samples);

    // 先看几个样本长什么样 + 预测结果。
    rprintln!("\n--- 前 4 个样本（图 + 预测/真值）---");
    for i in 0..4.min(n_samples) {
        let (label, img) = samples[i];
        core_ml::preprocess::ascii_art(img, |line| rprintln!("{}", line));
        let pred = model.predict(img);
        let conf = softmax_confidence(&model, img, s_out);
        rprintln!("  -> 预测 {}  真值 {}  置信度 {:.2}%\n", pred, label, conf * 100.0);
    }

    // 全量统计：准确率 + 推理延迟。
    let mut correct = 0usize;
    let mut total_cycles = 0u64;
    let mut cycles_n = 0u32;
    for i in 0..n_samples {
        let (label, img) = samples[i];
        if model.predict(img) == label as usize {
            correct += 1;
        }
        // 计一次 pure FC 的耗时。
        let (_, cyc) = meter.time(|| {
            let _ = model.infer(img);
        });
        total_cycles += cyc as u64;
        cycles_n += 1;
    }

    let acc = if n_samples > 0 {
        100.0 * correct as f32 / n_samples as f32
    } else {
        0.0
    };
    let avg_cyc = if cycles_n > 0 {
        total_cycles / cycles_n as u64
    } else {
        0
    };
    let macs = (N_PIXELS * N_CLASSES) as u64;
    rprintln!("========================================");
    rprintln!("端侧准确率：{:.2}%  ({}/{} 张)", acc, correct, n_samples);
    rprintln!(
        "平均推理：{} cycles ≈ {:.2} cycles/MAC, {:.1} MMAC/s",
        avg_cyc,
        avg_cyc as f32 / macs as f32,
        macs as f32 / (avg_cyc as f32 / core_hz as f32) / 1e6
    );
    rprintln!("（FC 只有 7840 MAC，纯算力不是瓶颈；瓶颈在后面的摄像头/屏）");

    loop {
        cortex_m::asm::wfi();
    }
}

/// 把 `samples.bin` 解析成 `(label, &[u8; 784])` 切片数组（零拷贝，全部指向静态字节）。
fn parse_samples<'a>(buf: &'a [u8]) -> (usize, alloc_free::Samples<'a>) {
    let stride = 1 + N_PIXELS;
    let n = buf.len() / stride;
    let mut out = alloc_free::Samples::new();
    for i in 0..n {
        let base = i * stride;
        let label = buf[base];
        // 指向 buf 里这段 784 字节；用 `try_into` 固定长度。
        let img: &'a [u8; N_PIXELS] = buf[base + 1..base + 1 + N_PIXELS]
            .try_into()
            .expect("样本长度不对");
        out.push((label, img));
    }
    (n, out)
}

/// 用真正的 f32 softmax 算置信度：int8 的 `logits` 已经等于 `z / S_out`，
/// 所以 `z = logits * S_out`，再 `softmax(z)` 才得到正确概率（softmax 只用于展示，
/// 判类靠的是 `argmax(logits)`，与温度缩放无关）。
fn softmax_confidence(model: &Model, img: &[u8; N_PIXELS], s_out: f32) -> f32 {
    let logits = model.infer(img);
    // z = logits * S_out，同时记录最大值用于数值稳定（减去最大值再 exp）。
    let mut z = [0.0f32; N_CLASSES];
    let mut zmax = f32::NEG_INFINITY;
    for k in 0..N_CLASSES {
        z[k] = logits[k] as f32 * s_out;
        if z[k] > zmax {
            zmax = z[k];
        }
    }
    // sum exp(z - zmax)，并把 z[k] 就地改成 exp 值以便取最大项。
    let mut sum = 0.0f32;
    for k in 0..N_CLASSES {
        z[k] = libm::expf(z[k] - zmax);
        sum += z[k];
    }
    if sum <= 0.0 {
        return 0.0;
    }
    // 置信度 = 最大那一项 / 总和。
    let mut best_v = z[0];
    for k in 1..N_CLASSES {
        if z[k] > best_v {
            best_v = z[k];
        }
    }
    best_v / sum
}

// 一个极小的、编译期定长的「样例列表」，避免引入 alloc。
mod alloc_free {
    use super::N_PIXELS;

    pub struct Samples<'a> {
        items: [(u8, &'a [u8; N_PIXELS]); 64],
        len: usize,
    }
    impl<'a> Samples<'a> {
        pub fn new() -> Self {
            // 全 0 初始化：label=0，img 指向一个静态全 0 缓冲（'static 可协变到任意 'a）。
            const ZERO: &[u8; N_PIXELS] = &[0u8; N_PIXELS];
            Self {
                items: [(0u8, ZERO); 64],
                len: 0,
            }
        }
        pub fn push(&mut self, item: (u8, &'a [u8; N_PIXELS])) {
            if self.len < 64 {
                self.items[self.len] = item;
                self.len += 1;
            }
        }
    }
    impl<'a> core::ops::Index<usize> for Samples<'a> {
        type Output = (u8, &'a [u8; N_PIXELS]);
        fn index(&self, i: usize) -> &Self::Output {
            &self.items[i]
        }
    }
}
