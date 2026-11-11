//! 第 3 步：把任意尺寸灰度图预处理成 28×28（背景 0、笔画 255）。
//!
//! 流水线（参考 hardwrite 的 `prep`，但用纯整数/定点实现，可在 `no_std` 下运行）：
//!
//! 1. **反色**：白底黑字（背景 255）→ 黑底白字（背景 0），与 MNIST 一致；
//! 2. **裁剪包围盒**：原始字符只占图中央一小块，先把有墨水的区域框出来；
//! 3. **等比缩放**：让最长边等于 `FIT`（留 4px 边距，即 28−2×4=20），保持长宽比；
//! 4. **按质心居中**：把墨水重心放到 28×28 正中间，减少位置抖动带来的干扰。
//!
//! 输入是行主序的灰度切片 `&[u8]`（长度 = w×h，取值 0..=255）。输出固定 `[u8; 784]`。

use crate::N_PIXELS;

/// 判断「这一格算不算笔画」的阈值（0-255）。太小会把抗锯齿噪点也算进去。
const INK_THRESHOLD: u8 = 30;
/// 缩放后字符最长边的目标尺寸（留出 4 像素边距）。
const FIT: f32 = 20.0;
/// 支持的最大输入边长（NIST SD19 是 128×128）。
/// 只做上限校验，不再按它开栈上定长缓冲（16 KiB × 2 会在 SWM341 上爆栈）。
const MAX_DIM: usize = 128;

/// `no_std` 下没有 `f32::floor`，自己实现一个（向负无穷取整）。
fn floor_f(x: f32) -> f32 {
    let t = x as i32;
    if (t as f32) > x {
        t as f32 - 1.0
    } else {
        t as f32
    }
}

/// `no_std` 下没有 `f32::round`，自己实现「四舍五入、half 远离零」。
fn round_f(x: f32) -> f32 {
    let r = floor_f(x.abs() + 0.5);
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// 把 `src`（w×h 行主序灰度）预处理成 28×28。
///
/// * `inverse` = true 表示原始是白底黑字，需要先反色。
/// 返回 `None` 的情况：尺寸非法、空图、或整张图没有墨水（包围盒为空）。
pub fn preprocess(src: &[u8], w: u32, h: u32, inverse: bool) -> Option<[u8; N_PIXELS]> {
    let w = w as usize;
    let h = h as usize;
    if w == 0 || h == 0 || w > MAX_DIM || h > MAX_DIM || src.len() < w * h {
        return None;
    }

    // 1) 反色 + 包围盒：反色在读像素时算，不落地整张 128×128 的栈缓冲。
    //    早期版本用 `inv` + `crop` 两个 16 KiB 定长数组，加上 main 的栈帧会
    //    把 SWM341 的 64 KiB SRAM 撑爆，踩坏 LVGL 对象/TLSF 堆导致 HardFault。
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;
    for y in 0..h {
        for x in 0..w {
            let p = src[y * w + x];
            let v = if inverse { 255 - p } else { p };
            if v > INK_THRESHOLD {
                found = true;
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if !found {
        return None;
    }

    // 2) 包围盒尺寸。
    let bw = max_x - min_x + 1;
    let bh = max_y - min_y + 1;

    // 3) 等比缩放，让最长边 = FIT。直接从原图按包围盒坐标做双线性采样，
    //    不再拷贝一份 `crop`，栈上只剩 28×28 的 `small`/`out`。
    let s = FIT / (bw.max(bh) as f32);
    let nw = (round_f(bw as f32 * s) as i32).max(1) as usize;
    let nh = (round_f(bh as f32 * s) as i32).max(1) as usize;
    let mut small = [0u8; 28 * 28]; // 最多 20×20，28×28 足够容下
    resize_bilinear_cropped(src, w, min_x, min_y, bw, bh, &mut small, nw, nh, inverse);

    // 4) 按质心居中到 28×28。
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sw = 0.0f32;
    for y in 0..nh {
        for x in 0..nw {
            let v = small[y * nw + x] as f32 / 255.0;
            sx += v * (x as f32 + 0.5);
            sy += v * (y as f32 + 0.5);
            sw += v;
        }
    }
    if sw == 0.0 {
        return None;
    }
    let cx = sx / sw;
    let cy = sy / sw;
    let off_x = 14.0 - cx; // 28/2 = 14
    let off_y = 14.0 - cy;

    let mut out = [0u8; N_PIXELS];
    for y in 0..28usize {
        for x in 0..28usize {
            let ix = floor_f(x as f32 - off_x) as i32;
            let iy = floor_f(y as f32 - off_y) as i32;
            if (0..nw as i32).contains(&ix) && (0..nh as i32).contains(&iy) {
                out[y * 28 + x] = small[iy as usize * nw + ix as usize];
            }
        }
    }
    Some(out)
}

/// 双线性缩放：把 `src` 里以 `(min_x, min_y)` 为原点、`sw×sh` 的包围盒
/// 缩放到 `dst`(dw×dh)。反色在采样时完成，等价于先反色 → 裁剪 → 缩放，
/// 但不需要任何大块栈缓冲。纯 `no_std`，不依赖浮点库也能编。
fn resize_bilinear_cropped(
    src: &[u8],
    src_w: usize,
    min_x: usize,
    min_y: usize,
    sw: usize,
    sh: usize,
    dst: &mut [u8],
    dw: usize,
    dh: usize,
    inverse: bool,
) {
    let pixel = |x: usize, y: usize| -> f32 {
        let p = src[(min_y + y) * src_w + (min_x + x)];
        (if inverse { 255 - p } else { p }) as f32
    };
    for dy in 0..dh {
        for dx in 0..dw {
            let fx = (dx as f32 + 0.5) * sw as f32 / dw as f32 - 0.5;
            let fy = (dy as f32 + 0.5) * sh as f32 / dh as f32 - 0.5;
            let x0 = floor_f(fx).max(0.0) as usize;
            let y0 = floor_f(fy).max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let wx = fx - x0 as f32;
            let wy = fy - y0 as f32;

            let v00 = pixel(x0, y0);
            let v01 = pixel(x1, y0);
            let v10 = pixel(x0, y1);
            let v11 = pixel(x1, y1);
            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            let val = top * (1.0 - wy) + bot * wy;
            dst[dy * dw + dx] = round_f(val).clamp(0.0, 255.0) as u8;
        }
    }
}

/// 把一张 28×28 的图逐行推给 `emit`（板端用 `rprintln!`、主机用 `print!` 都能接）。
///
/// 不依赖 `alloc`，便于在 `no_std` 环境里肉眼检查数据是否正确。
pub fn ascii_art(img: &[u8], mut emit: impl FnMut(&str)) {
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let mut line = [0u8; 29]; // 28 字符 + '\n'
    for r in 0..28 {
        for c in 0..28 {
            let v = img[r * 28 + c] as usize;
            line[c] = ramp[(v * ramp.len()) / 256] as u8;
        }
        line[28] = b'\n';
        // ramp 与换行都是 ASCII，from_utf8 必然成功。
        emit(core::str::from_utf8(&line).unwrap());
    }
}
