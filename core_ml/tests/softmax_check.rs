//! 复现固件 02_infer 的 `infer + softmax_s8` 路径，把 logits / prob 真实打出来，
//! 定位「置信度恒为 10%」的 bug。运行：`cargo test -p core_ml --test softmax_check -- --nocapture`

use core_ml::{Model, N_CLASSES, N_PIXELS};

#[test]
fn softmax_check() {
    let model_bin = std::fs::read("../gen/model.bin").unwrap();
    let samples_bin = std::fs::read("../gen/samples.bin").unwrap();
    let model = Model::from_bytes(&model_bin).unwrap();

    let s_out = f32::from_le_bytes(model_bin[16..20].try_into().unwrap());
    // embedded-nn 的 softmax_s8 内部用 `exp_on_negative_values`，期望
    // exp_input = diff * (mult / 2^(31-shift)) 落在「百万量级」（按 ln2 拆分），
    // 与 s_out 无关。用库自带测试里的标度：mult = 1<<30, shift = 20。
    let mult = 1i32 << 30;
    let shift = 20i32;
    let diff_min = -256i32;
    println!("s_out={} mult={} shift={} diff_min={}", s_out, mult, shift, diff_min);

    let stride = 1 + N_PIXELS;
    for s in 0..4.min(samples_bin.len() / stride) {
        let base = s * stride;
        let label = samples_bin[base];
        let img: &[u8; N_PIXELS] = samples_bin[base + 1..base + 1 + N_PIXELS]
            .try_into()
            .unwrap();
        let logits = model.infer(img);
        println!("样本{} 真值{} logits={:?}", s, label, &logits[..]);
        let mut prob = [0i8; N_CLASSES];
        let r = embedded_nn::softmax_s8(&logits, 1, N_CLASSES, mult, shift, diff_min, &mut prob);
        println!("  softmax_s8={:?} prob={:?}", r, &prob[..]);
        let best = *prob.iter().max().unwrap();
        let mass: i32 = prob.iter().map(|&v| v as i32 + 128).sum();
        println!(
            "  embedded-nn softmax: best_v+128={} mass={} conf={:.3}",
            best as i32 + 128,
            mass,
            (best as i32 + 128) as f32 / mass as f32
        );

        // 正确的 f32 softmax：z = logits * S_out
        let z: [f32; N_CLASSES] = core::array::from_fn(|k| logits[k] as f32 * s_out);
        let zmax = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut zi = [0.0f32; N_CLASSES];
        for k in 0..N_CLASSES {
            zi[k] = (z[k] - zmax).exp();
            sum += zi[k];
        }
        let pred_idx = z
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let conf_true = zi[pred_idx] / sum;
        println!(
            "  proper f32 softmax: argmax={} conf={:.3}",
            pred_idx, conf_true
        );
    }
}
