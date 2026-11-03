//! 历程 1：定点算子库自检。
//!
//! 目的：在还没有摄像头、没有屏幕、没有模型的时候，先确认"算子算得对"。
//!
//! 用例都是能一眼手算的小张量，量化参数刻意取恒等（乘子 = Q31 的 1.0、shift = 0），
//! 所以期望值就是朴素的整数结果；任何一项不对，说明是维度/零点的约定弄错了，
//! 而不是"我数学算错了"。softmax 则用 CMSIS-NN 官方单测里那条 TFLite 参考向量，
//! 属于"外来标准答案"。
//!
//! 最后用 DWT 周期计数器量 cycles/MAC，给后面的模型选型一个手感。
//!
//! ```text
//! cargo run --bin 01_nn_smoke
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::bench::CycleMeter;
use swm341_ai::check::Checker;
use swm341_ai::hal;
use swm341_ai::nn::{self, *};

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 01_nn_smoke: 定点算子自检 ====");

    let p = hal::init(hal::Config::hrc_150mhz());
    let core_hz = hal::clock::sys_core_clock();

    let mut meter = CycleMeter::new(p.dwt());
    let mut check = Checker::new();

    check.section("运行环境");
    rprintln!("  算子库  : embedded-nn 0.2.3（纯 Rust，#![no_std]，零堆分配）");
    rprintln!("  主频    : {} MHz", core_hz / 1_000_000);
    rprintln!(
        "  dsp cfg : {}（stable rustc 不把 arm 的 dsp 暴露到 cfg，详见 README）",
        cfg!(target_feature = "dsp")
    );
    check.expect("DWT 周期计数器可用", meter.has_cycle_counter());

    check.section("定点数学 support::requantize");
    // 乘子 1.0 + shift 0 时必须原样返回，这是后面所有恒等量化用例的前提。
    let probes: [i32; 9] = [0, 1, -1, 127, -128, 1000, -1000, 1 << 20, -(1 << 20)];
    let mut requantized = [0i32; 9];
    for (i, &v) in probes.iter().enumerate() {
        requantized[i] = requantize(v, UNITY_MULTIPLIER, 0);
    }
    check.expect_i32_slice("requantize(x, 1.0, 0) == x", &requantized, &probes);

    test_fully_connected(&mut check);
    test_conv(&mut check);
    test_depthwise_conv(&mut check);
    test_pooling(&mut check);
    test_elementwise(&mut check);
    test_activation(&mut check);
    test_softmax(&mut check);

    if meter.has_cycle_counter() {
        bench_fully_connected(&mut check, &mut meter, core_hz);
        bench_conv(&mut check, &mut meter, core_hz);
    }

    if check.summary() {
        rprintln!("结论：算子库工作正常，可以进入下一步（图像采集）。");
    } else {
        rprintln!("结论：有算子不符合预期，先修到全绿再往前走。");
    }

    loop {
        cortex_m::asm::wfi();
    }
}

// ------------------------------------------------------------------ 全连接 --

fn test_fully_connected(check: &mut Checker) {
    check.section("全连接 fully_connected_s8");

    // 输入 x = [1..8]，四个输出通道分别取：x0、x1、sum(x)、-x0。
    let input: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    let mut filter = [0i8; 4 * 8];
    filter[0] = 1;
    filter[8 + 1] = 1;
    for k in 0..8 {
        filter[2 * 8 + k] = 1;
    }
    filter[3 * 8] = -1;

    let bias: [i32; 4] = [10, -20, 0, 0];
    let fc = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: ACT_S8,
    };

    let mut out = [0i8; 4];
    let result = fully_connected_s8(
        &fc,
        &nn::identity_quant(),
        // 输入维度 [N, H, W, C_IN]；权重维度里的 n 表示"输入深度"
        &Dims::new(1, 1, 1, 8),
        &input,
        &Dims::new(8, 1, 1, 1),
        &filter,
        Some(&bias),
        &Dims::new(1, 1, 1, 4),
        &mut out,
    );

    check.expect_ok("status", result);
    check.expect_i8("y = W·x + b", &out, &[11, -18, 36, -1]);
}

// ------------------------------------------------------------------- 卷积 --

fn test_conv(check: &mut Checker) {
    check.section("卷积 convolve_per_channel_s8");

    // 4x4 单通道，2x2 全 1 卷积核（VALID，stride 1）→ 3x3，每个点是 2x2 窗口之和。
    let input: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let filter = [1i8; 4];
    let mult = [UNITY_MULTIPLIER; 1];
    let shift = [0i32; 1];

    let conv = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Padding2D::default(),
        dilation: Tile::new(1, 1),
        activation: ACT_S8,
    };

    let mut out = [0i8; 9];
    let result = convolve_per_channel_s8(
        &conv,
        &PerChannelQuantParams::new(&mult, &shift),
        &Dims::new(1, 4, 4, 1),
        &input,
        &Dims::new(1, 2, 2, 1),
        &filter,
        None,
        &Dims::new(1, 3, 3, 1),
        &mut out,
    );

    check.expect_ok("status", result);
    check.expect_i8("3x3 窗口和", &out, &[14, 18, 22, 30, 34, 38, 46, 50, 54]);
}

fn test_depthwise_conv(check: &mut Checker) {
    check.section("逐通道卷积 depthwise_conv_per_channel_s8");

    // 3x3x2 输入（HWC 交织），每通道一个 2x2 全 1 卷积核 → 2x2x2。
    // 通道 0 的窗口和：12 / 16 / 24 / 28；通道 1：48 / 52 / 60 / 64。
    let input: [i8; 18] = [
        1, 10, 2, 11, 3, 12, //
        4, 13, 5, 14, 6, 15, //
        7, 16, 8, 17, 9, 18,
    ];
    let filter = [1i8; 8];
    let mult = [UNITY_MULTIPLIER; 2];
    let shift = [0i32; 2];

    let dw = DwConvParams {
        input_offset: 0,
        output_offset: 0,
        ch_mult: 1,
        stride: Tile::new(1, 1),
        padding: Padding2D::default(),
        dilation: Tile::new(1, 1),
        activation: ACT_S8,
    };

    let mut out = [0i8; 8];
    let result = depthwise_conv_per_channel_s8(
        &dw,
        &PerChannelQuantParams::new(&mult, &shift),
        &Dims::new(1, 3, 3, 2),
        &input,
        &Dims::new(1, 2, 2, 2),
        &filter,
        None,
        &Dims::new(1, 2, 2, 2),
        &mut out,
    );

    check.expect_ok("status", result);
    check.expect_i8("每通道 2x2 窗口和", &out, &[12, 48, 16, 52, 24, 60, 28, 64]);
}

// ------------------------------------------------------------------- 池化 --

fn test_pooling(check: &mut Checker) {
    check.section("池化 max_pool_s8 / avg_pool_s8");

    let pool = PoolParams {
        stride: Tile::new(2, 2),
        padding: Padding2D::default(),
        activation: ACT_S8,
    };
    let kernel = Tile::new(2, 2);

    let max_input: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mut max_out = [0i8; 4];
    let result = max_pool_s8(
        &pool,
        &kernel,
        &Dims::new(1, 4, 4, 1),
        &max_input,
        &Dims::new(1, 2, 2, 1),
        &mut max_out,
    );
    check.expect_ok("maxpool status", result);
    check.expect_i8("maxpool 2x2/2", &max_out, &[6, 8, 14, 16]);

    // 每块和都能被 4 整除，避免纠结平均池化的取整方向。
    let avg_input: [i8; 16] = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
    let mut avg_out = [0i8; 4];
    let result = avg_pool_s8(
        &pool,
        &kernel,
        &Dims::new(1, 4, 4, 1),
        &avg_input,
        &Dims::new(1, 2, 2, 1),
        &mut avg_out,
    );
    check.expect_ok("avgpool status", result);
    check.expect_i8("avgpool 2x2/2", &avg_out, &[5, 9, 13, 17]);
}

// -------------------------------------------------------------- 逐元素运算 --

fn test_elementwise(check: &mut Checker) {
    check.section("逐元素 elementwise_add_s8 / _mul_s8");

    let a: [i8; 4] = [1, 2, 3, -4];
    let b: [i8; 4] = [10, 20, -30, 4];
    let add = ElementwiseAddParams {
        input1_offset: 0,
        input1_mult: UNITY_MULTIPLIER,
        input1_shift: 0,
        input2_offset: 0,
        input2_mult: UNITY_MULTIPLIER,
        input2_shift: 0,
        left_shift: 0,
        output_offset: 0,
        output_mult: UNITY_MULTIPLIER,
        output_shift: 0,
        activation: ACT_S8,
    };
    let mut sum = [0i8; 4];
    let result = elementwise_add_s8(&a, &b, &mut sum, &add);
    check.expect_ok("add status", result);
    check.expect_i8("a + b", &sum, &[11, 22, -27, 0]);

    let m1: [i8; 3] = [2, 3, 4];
    let m2: [i8; 3] = [5, 6, 7];
    let mul = ElementwiseMulParams {
        input1_offset: 0,
        input2_offset: 0,
        output_offset: 0,
        output_mult: UNITY_MULTIPLIER,
        output_shift: 0,
        activation: ACT_S8,
    };
    let mut product = [0i8; 3];
    let result = elementwise_mul_s8(&m1, &m2, &mut product, &mul);
    check.expect_ok("mul status", result);
    check.expect_i8("a * b", &product, &[10, 18, 28]);
}

fn test_activation(check: &mut Checker) {
    check.section("激活 relu_s8 / relu6_s8");

    let mut relu = [-5i8, 0, 5, 127, -128, 64];
    relu_s8(&mut relu);
    check.expect_i8("relu", &relu, &[0, 0, 5, 127, 0, 64]);

    let mut relu6 = [-5i8, 0, 5, 127, -128, 64];
    relu6_s8(&mut relu6);
    check.expect_i8("relu6", &relu6, &[0, 0, 5, 6, 0, 6]);
}

// ----------------------------------------------------------------- softmax --

fn test_softmax(check: &mut Checker) {
    check.section("softmax softmax_s8");

    // 取自 CMSIS-NN 官方单测（TestData/softmax），期望值是 TFLite 参考实现生成的。
    // 注意 int8 softmax 的"概率和"约定：(out + 128) 之和 ≈ 256。
    let input: [i8; 10] = [101, 49, 6, -34, -75, -79, -38, 120, -55, 115];
    let want: [i8; 10] = [-57, -70, -79, -86, -92, -94, -88, -54, -91, -56];
    let mut out = [0i8; 10];

    let result = softmax_s8(&input, 2, 5, 1_077_952_640, 19, -3968, &mut out);
    check.expect_ok("status", result);
    check.expect_i8("两行 5 类的 int8 概率", &out, &want);

    // 再验一次归一化：单调递增的输入，概率也必须单调递增且总和 ≈ 256。
    let probe: [i8; 5] = [10, 20, 30, 40, 50];
    let mut probe_out = [0i8; 5];
    let result = softmax_s8(&probe, 1, 5, 1_077_952_640, 19, -3968, &mut probe_out);
    check.expect_ok("status(probe)", result);

    let mut monotonic = true;
    for i in 1..probe_out.len() {
        if probe_out[i] < probe_out[i - 1] {
            monotonic = false;
        }
    }
    let mass: i32 = probe_out.iter().map(|&v| v as i32 + 128).sum();
    check.expect("概率随输入单调不减", monotonic);
    check.expect("(out + 128) 之和落在 250..=260", (250..=260).contains(&mass));
    rprintln!("  探针输出 = {:?}，质量和 = {}", probe_out, mass);
}

// -------------------------------------------------------------------- 基准 --

fn bench_fully_connected(check: &mut Checker, meter: &mut CycleMeter<'_>, core_hz: u32) {
    check.section("基准：全连接 128 -> 32");

    const IN: usize = 128;
    const OUT: usize = 32;
    const REPS: u32 = 500;

    let input = [1i8; IN];
    let filter = [1i8; IN * OUT];
    let bias = [0i32; OUT];
    let fc = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: ACT_S8,
    };
    let mut out = [0i8; OUT];

    let result = fully_connected_s8(
        &fc,
        &nn::identity_quant(),
        &Dims::new(1, 1, 1, IN as i32),
        &input,
        &Dims::new(IN as i32, 1, 1, 1),
        &filter,
        Some(&bias),
        &Dims::new(1, 1, 1, OUT as i32),
        &mut out,
    );
    check.expect_ok("首次调用", result);

    let (_, cycles) = meter.time(|| {
        for _ in 0..REPS {
            let _ = fully_connected_s8(
                &fc,
                &nn::identity_quant(),
                &Dims::new(1, 1, 1, IN as i32),
                &input,
                &Dims::new(IN as i32, 1, 1, 1),
                &filter,
                Some(&bias),
                &Dims::new(1, 1, 1, OUT as i32),
                &mut out,
            );
        }
    });

    report_throughput("全连接", cycles, (IN * OUT) as u64 * REPS as u64, core_hz);
}

fn bench_conv(check: &mut Checker, meter: &mut CycleMeter<'_>, core_hz: u32) {
    check.section("基准：卷积 16x16x1 -> 4 个 3x3 核");

    const REPS: u32 = 200;
    const OUT_HW: usize = 14; // (16 - 3)/1 + 1

    let input = [1i8; 16 * 16];
    let filter = [1i8; 4 * 3 * 3];
    let bias = [0i32; 4];
    let mult = [UNITY_MULTIPLIER; 4];
    let shift = [0i32; 4];

    let conv = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Padding2D::default(),
        dilation: Tile::new(1, 1),
        activation: ACT_S8,
    };
    let mut out = [0i8; OUT_HW * OUT_HW * 4];

    let run = |out: &mut [i8]| {
        convolve_per_channel_s8(
            &conv,
            &PerChannelQuantParams::new(&mult, &shift),
            &Dims::new(1, 16, 16, 1),
            &input,
            &Dims::new(1, 3, 3, 1),
            &filter,
            Some(&bias),
            &Dims::new(1, OUT_HW as i32, OUT_HW as i32, 4),
            out,
        )
    };

    let result = run(&mut out);
    check.expect_ok("首次调用", result);
    check.expect_i32("全 1 输入/卷积核 → 每个输出 = 9", out[0] as i32, 9);

    let (_, cycles) = meter.time(|| {
        for _ in 0..REPS {
            let _ = run(&mut out);
        }
    });

    let macs = (OUT_HW * OUT_HW * 4 * 3 * 3) as u64 * REPS as u64;
    report_throughput("卷积", cycles, macs, core_hz);
}

/// 打印总周期、cycles/MAC 与等效吞吐。
fn report_throughput(label: &str, cycles: u32, macs: u64, core_hz: u32) {
    let per_mac = cycles as f32 / macs as f32;
    let seconds = cycles as f32 / core_hz as f32;
    let mmac_s = macs as f32 / seconds / 1_000_000.0;

    rprintln!(
        "  {}: {} cycles, {:.2} cycles/MAC, {:.1} MMAC/s",
        label, cycles, per_mac, mmac_s
    );
}
