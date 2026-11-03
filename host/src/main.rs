//! 在 Mac 上完成「训练 → 量化 → 验证」整条链路，并把权重导出给 SWM341 固件。
//!
//! 跑法：
//! ```text
//! cd host
//! cargo run --release            # 默认 每类 1500 训练 / 1000 测试
//! cargo run --release -- --train 2000 --test 1000
//! ```
//!
//! 它做四件事：
//! 1. 从 `../data/by_field.zip` 抽样本，用 **core_ml::preprocess**（与板端同一份代码）预处理；
//! 2. 训练 Softmax 回归（单层全连接 784→10），照搬 hardwrite 第 6 章；
//! 3. 把权重量化成 int8（per-channel），写成 `../gen/model.bin`；
//! 4. 用 **core_ml::Model**（内部调用真正的 embedded-nn 算子）跑 int8 推理，
//!    对比浮点精度，证明量化误差可控——这一步通过，板端结果就必然一致。

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};
use core_ml::{Model, N_CLASSES, N_PIXELS};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// 输入维度（784 像素 + 1 偏置位）。
const D: usize = N_PIXELS + 1;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut train_per_class = 1500usize;
    let mut test_per_class = 1000usize;
    let mut seed = 42u64;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--train" => train_per_class = it.next().context("--train 缺参数")?.parse()?,
            "--test" => test_per_class = it.next().context("--test 缺参数")?.parse()?,
            "--seed" => seed = it.next().context("--seed 缺参数")?.parse()?,
            other => bail!("未知参数 {other}"),
        }
    }

    let zip_path = "../data/by_field.zip";
    let (train_imgs, train_lbls, test_imgs, test_lbls) =
        load_dataset(zip_path, train_per_class, test_per_class, seed)?;
    println!(
        "数据集：训练 {} 张，测试 {} 张，每类已按书写者划分（hsf_7 作测试）",
        train_imgs.len(),
        test_imgs.len()
    );

    // ---- 2. 训练 Softmax 回归 ----
    let (txs, tys) = build(&train_imgs, &train_lbls, train_imgs.len());
    println!(
        "模型：784 → 10 全连接 + Softmax，共 {} 参数（浮点）",
        D * N_CLASSES
    );
    let w = train_softmax(&txs, &tys, 64, 0.05, 8, seed);
    let float_acc = accuracy(&w, &txs, &tys);
    println!("浮点训练准确率 {float_acc:.2}%");

    // ---- 3. 量化成 int8 ----
    let (model, s_out) = quantize(&w, &txs);
    println!(
        "量化：q_w {} 字节 + bias/mult/shift {} 字节，S_out={:.5}",
        N_CLASSES * N_PIXELS,
        N_CLASSES * 12,
        s_out
    );

    // ---- 4. 验证 int8 推理 ----
    let int8_acc = eval_int8(&model, &test_imgs, &test_lbls);
    let (ex_imgs, ex_lbls) = build(&test_imgs, &test_lbls, test_imgs.len());
    let float_test = accuracy(&w, &ex_imgs, &ex_lbls);
    println!("浮点测试准确率 {float_test:.2}%");
    println!(
        "int8 测试准确率 {:.2}%  （与浮点差距 {:.2} 个百分点）",
        int8_acc,
        float_test - int8_acc
    );

    // ---- 导出权重 + 几个样例给固件 ----
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let gen = root.join("gen");
    fs::create_dir_all(&gen).context("创建 gen/ 失败")?;

    let mut buf = Vec::new();
    core_ml::model::write_model(
        |b| {
            buf.extend_from_slice(b);
            Ok(())
        },
        &model,
        s_out,
    )
    .unwrap();
    fs::write(gen.join("model.bin"), &buf).context("写 model.bin 失败")?;

    // 导出最多 64 个测试样本（label + 784 像素），供固件上板自测。
    let n_sample = test_imgs.len().min(64);
    let mut sbuf = Vec::with_capacity(n_sample * (1 + N_PIXELS));
    for i in 0..n_sample {
        sbuf.push(test_lbls[i]);
        sbuf.extend_from_slice(&test_imgs[i]);
    }
    fs::write(gen.join("samples.bin"), &sbuf).context("写 samples.bin 失败")?;
    println!("已导出 {} 字节权重 -> gen/model.bin", buf.len());
    println!("已导出 {} 个样例 -> gen/samples.bin", n_sample);

    // ---- 肉眼检查：打印前 3 个样本的图 + 预测 ----
    println!("\n前 3 个测试样本（左：图，右：预测/真值）：");
    for i in 0..3.min(test_imgs.len()) {
        core_ml::preprocess::ascii_art(&test_imgs[i], |line| print!("{line}"));
        let pred = model.predict(&test_imgs[i]);
        println!("  -> 预测 {}  真值 {}\n", pred, test_lbls[i]);
    }

    // ---- 端到端回读：走「固件那条路径」再验一次 ----
    // 从导出的字节用 Model::from_bytes 加载（02_infer 板端就走这条），跑同一个算子。
    let rb = fs::read(gen.join("model.bin")).context("回读 model.bin 失败")?;
    let loaded = Model::from_bytes(&rb).expect("from_bytes 失败（板端也会走这条路径）");
    let sb = fs::read(gen.join("samples.bin")).context("回读 samples.bin 失败")?;
    let n2 = sb.len() / (1 + N_PIXELS);
    let mut ok2 = 0usize;
    for i in 0..n2 {
        let base = i * (1 + N_PIXELS);
        let label = sb[base];
        let img: &[u8; N_PIXELS] = sb[base + 1..base + 1 + N_PIXELS].try_into().unwrap();
        if loaded.predict(img) == label as usize {
            ok2 += 1;
        }
    }
    println!(
        "端到端回读（字节→from_bytes→predict，即板端路径）：{:.2}% ({}/{})",
        100.0 * ok2 as f32 / n2 as f32,
        ok2,
        n2
    );

    Ok(())
}

// ============================================================ 数据加载

type Dataset = (Vec<[u8; N_PIXELS]>, Vec<u8>);

fn load_dataset(
    zip_path: &str,
    train_per_class: usize,
    test_per_class: usize,
    seed: u64,
) -> Result<(Vec<[u8; N_PIXELS]>, Vec<u8>, Vec<[u8; N_PIXELS]>, Vec<u8>)> {
    let file = fs::File::open(zip_path).with_context(|| format!("打不开 {zip_path}，先放好数据集"))?;
    let mut archive = zip::ZipArchive::new(file)?;
    println!("压缩包内共 {} 个条目，扫描数字类别中……", archive.len());

    // 阶段 1：只登记条目位置（key = (是否测试集, 标签)）。
    let mut groups: HashMap<(bool, u8), Vec<usize>> = HashMap::new();
    for i in 0..archive.len() {
        let name = match archive.by_index(i) {
            Ok(e) => e.name().to_string(),
            Err(_) => continue,
        };
        let parts: Vec<&str> = name.split('/').collect();
        // by_field/hsf_X/digit/XX/XX_00001.png
        if parts.len() < 5 || parts[2] != "digit" {
            continue;
        }
        let code = match u8::from_str_radix(parts[3], 16) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !(0x30..0x3a).contains(&code) {
            continue;
        }
        let label = code - 0x30;
        let is_test = parts[1] == "hsf_7";
        groups.entry((is_test, label)).or_default().push(i);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut train_imgs = Vec::new();
    let mut train_lbls = Vec::new();
    let mut test_imgs = Vec::new();
    let mut test_lbls = Vec::new();

    for is_test in [false, true] {
        let per_class = if is_test { test_per_class } else { train_per_class };
        for label in 0..10u8 {
            let idxs = match groups.get(&(is_test, label)) {
                Some(v) => v,
                None => continue,
            };
            let mut picked: Vec<usize> = idxs.clone();
            picked.shuffle(&mut rng);
            let take = per_class.min(picked.len());
            let mut buf = Vec::new();
            for &i in &picked[..take] {
                buf.clear();
                let mut e = archive.by_index(i)?;
                e.read_to_end(&mut buf)?;
                if let Some(px) = decode(&buf) {
                    if is_test {
                        test_imgs.push(px);
                        test_lbls.push(label);
                    } else {
                        train_imgs.push(px);
                        train_lbls.push(label);
                    }
                }
            }
            println!(
                "  {:<5} 数字 {}: {}/{} 张",
                if is_test { "测试" } else { "训练" },
                label,
                take,
                idxs.len()
            );
        }
    }

    // 关键：图片和标签要用「同一套随机排列」一起打乱，否则会错位。
    // （hardwrite 的 Dataset::shuffle 就是一次置换同时重排两者；这里不能分别 shuffle。）
    let mut ti_idx: Vec<usize> = (0..train_imgs.len()).collect();
    ti_idx.shuffle(&mut rng);
    train_imgs = ti_idx.iter().map(|&i| train_imgs[i]).collect();
    train_lbls = ti_idx.iter().map(|&i| train_lbls[i]).collect();

    let mut te_idx: Vec<usize> = (0..test_imgs.len()).collect();
    te_idx.shuffle(&mut rng);
    test_imgs = te_idx.iter().map(|&i| test_imgs[i]).collect();
    test_lbls = te_idx.iter().map(|&i| test_lbls[i]).collect();

    Ok((train_imgs, train_lbls, test_imgs, test_lbls))
}

/// 解码一张 PNG 成 28×28 灰度（背景 0、笔画 255）。NIST 是白底黑字，需反色。
fn decode(png: &[u8]) -> Option<[u8; N_PIXELS]> {
    let img = image::load_from_memory(png).ok()?.to_luma8();
    let (w, h) = (img.width(), img.height());
    core_ml::preprocess::preprocess(img.as_raw(), w, h, true)
}

// ============================================================ 训练（Softmax 回归）

fn build(imgs: &[[u8; N_PIXELS]], labels: &[u8], n: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = Vec::with_capacity(D);
        for &p in imgs[i].iter() {
            x.push(p as f32 / 255.0);
        }
        x.push(1.0); // 偏置位
        xs.push(x);
        ys.push(labels[i] as usize);
    }
    (xs, ys)
}

fn train_softmax(
    xs: &[Vec<f32>],
    ys: &[usize],
    batch: usize,
    lr: f32,
    epochs: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = xs.len();
    let mut w = vec![vec![0.0f32; D]; N_CLASSES];
    let mut rng = StdRng::seed_from_u64(seed);

    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let steps = n / batch;
        for _ in 0..steps {
            let mut grad = vec![vec![0.0f32; D]; N_CLASSES];
            for _ in 0..batch {
                let i = rng.gen_range(0..n);
                let mut logits = forward(&w, &xs[i]);
                softmax(&mut logits);
                total_loss += -(logits[ys[i]] + 1e-12).ln();
                for k in 0..N_CLASSES {
                    let dz = if k == ys[i] { logits[k] - 1.0 } else { logits[k] };
                    for (g, &xv) in grad[k].iter_mut().zip(xs[i].iter()) {
                        *g += dz * xv;
                    }
                }
            }
            for k in 0..N_CLASSES {
                for j in 0..D {
                    w[k][j] -= lr * grad[k][j] / batch as f32;
                }
            }
        }
        let acc = accuracy(&w, xs, ys);
        println!(
            "  epoch {epoch}: 平均损失 {:.4}, 训练准确率 {:.2}%",
            total_loss / (steps * batch) as f32,
            acc
        );
    }
    w
}

fn forward(w: &[Vec<f32>], x: &[f32]) -> [f32; N_CLASSES] {
    let mut out = [0.0f32; N_CLASSES];
    for k in 0..N_CLASSES {
        let mut s = 0.0f32;
        for (wj, xj) in w[k].iter().zip(x.iter()) {
            s += wj * xj;
        }
        out[k] = s;
    }
    out
}

fn softmax(logits: &mut [f32]) {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - m).exp();
        sum += *l;
    }
    for l in logits.iter_mut() {
        *l /= sum;
    }
}

fn predict(w: &[Vec<f32>], x: &[f32]) -> usize {
    let mut logits = forward(w, x);
    softmax(&mut logits);
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn accuracy(w: &[Vec<f32>], xs: &[Vec<f32>], ys: &[usize]) -> f32 {
    let ok = xs
        .iter()
        .zip(ys.iter())
        .filter(|(x, &y)| predict(w, x) == y)
        .count();
    100.0 * ok as f32 / xs.len() as f32
}

// ============================================================ 量化

/// 把浮点权重量化成 int8 模型（per-channel）。
///
/// 约定见 `core_ml::model` 模块文档。返回 `(Model, S_out)`。
fn quantize(w: &[Vec<f32>], calib_xs: &[Vec<f32>]) -> (Model, f32) {
    // 校准：在训练集上跑一遍浮点前向，取 max|z| 定 S_out。
    let mut max_abs = 0.0f32;
    for x in calib_xs {
        let z = forward(w, x);
        for &v in z.iter() {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    let s_out = if max_abs > 0.0 {
        max_abs / 127.0
    } else {
        1.0
    };

    let mut q_w = [0i8; N_CLASSES * N_PIXELS];
    let mut bias32 = [0i32; N_CLASSES];
    let mut mult = [0i32; N_CLASSES];
    let mut shift = [0i32; N_CLASSES];

    for k in 0..N_CLASSES {
        // 权重只量化像素那 784 个，偏置位(第 785 个)单独进 int32。
        let mut maxw = 0.0f32;
        for j in 0..N_PIXELS {
            let a = w[k][j].abs();
            if a > maxw {
                maxw = a;
            }
        }
        let s_w = if maxw > 0.0 { maxw / 127.0 } else { 1.0 };
        for j in 0..N_PIXELS {
            let q = (w[k][j] / s_w).round().clamp(-127.0, 127.0);
            q_w[k * N_PIXELS + j] = q as i8;
        }
        let b = w[k][N_PIXELS]; // 偏置位
        bias32[k] = (b * 255.0 / s_w).round() as i32;

        // 有效乘子 S'_k = S_w / (255 · S_out)，再转成 (mult, shift)。
        let s_prime = s_w / (255.0 * s_out);
        let (m, sh) = quantize_multiplier(s_prime as f64);
        mult[k] = m;
        shift[k] = sh;
    }

    (Model::new(q_w, bias32, mult, shift), s_out)
}

/// 把实数乘子编码成 CMSIS 风格的 `(mult, shift)`：`mult ∈ [2^30, 2^31)`，shift 可正可负。
fn quantize_multiplier(s: f64) -> (i32, i32) {
    if s <= 0.0 {
        return (0, 0);
    }
    let mut shift = 0i32;
    let mut sig = s;
    while sig >= 1.0 {
        sig /= 2.0;
        shift += 1;
    }
    while sig < 0.5 {
        sig *= 2.0;
        shift -= 1;
    }
    let mut mult = (sig * (1u64 << 31) as f64).round() as i64;
    if mult >= (1i64 << 31) {
        mult = (1i64 << 31) - 1;
        shift -= 1;
    }
    (mult as i32, shift)
}

fn eval_int8(model: &Model, imgs: &[[u8; N_PIXELS]], labels: &[u8]) -> f32 {
    let mut ok = 0;
    for (i, &lab) in labels.iter().enumerate() {
        if model.predict(&imgs[i]) == lab as usize {
            ok += 1;
        }
    }
    100.0 * ok as f32 / labels.len() as f32
}
