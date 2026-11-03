//! 极简 PNG 解码器（只支持 8-bit、非交错、颜色类型 0/2/6 → 灰度）。
//!
//! 够把手写数字图片（白底黑字或黑底白字）解成灰度 `Vec<u8>` 喂给 `core_ml::preprocess`。
//! 不带 CRC 校验、不带 LFN、不支持隔行（Adam7）和 16-bit——这些都超出本项目的需要。

use alloc::vec;
use alloc::vec::Vec;

use miniz_oxide::inflate::decompress_to_vec_zlib;

/// 解码错误。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngError {
    /// 文件头签名不对。
    Signature,
    /// IHDR 长度/字段不对。
    Header,
    /// 不支持的格式（非 8-bit、隔行、或非 0/2/6 颜色类型）。
    Unsupported,
    /// zlib 解压失败。
    Inflate,
    /// 解滤镜失败（数据被截断或未知滤镜）。
    Defilter,
}

const SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// 把整张 PNG 文件解码成 `(width, height, 灰度字节)`。
pub fn decode_png(file: &[u8]) -> Result<(u32, u32, Vec<u8>), PngError> {
    if file.len() < 8 || file[..8] != SIGNATURE {
        return Err(PngError::Signature);
    }

    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut channels = 0u32;
    let mut idat = Vec::new();

    // 逐 chunk 解析：len(4 BE) + type(4) + data(len) + crc(4)。
    while pos + 8 <= file.len() {
        let len = u32::from_be_bytes([file[pos], file[pos + 1], file[pos + 2], file[pos + 3]])
            as usize;
        let kind = &file[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start + len;
        if data_end + 4 > file.len() {
            return Err(PngError::Defilter); // 截断
        }
        if kind == b"IHDR" {
            if len < 13 {
                return Err(PngError::Header);
            }
            width = u32::from_be_bytes(file[data_start..data_start + 4].try_into().unwrap());
            height = u32::from_be_bytes(file[data_start + 4..data_start + 8].try_into().unwrap());
            let bit_depth = file[data_start + 8];
            let color_type = file[data_start + 9];
            let interlace = file[data_start + 12];
            if bit_depth != 8 {
                return Err(PngError::Unsupported);
            }
            channels = match color_type {
                0 => 1,
                2 => 3,
                6 => 4,
                _ => return Err(PngError::Unsupported),
            };
            if interlace != 0 {
                return Err(PngError::Unsupported);
            }
        } else if kind == b"IDAT" {
            idat.extend_from_slice(&file[data_start..data_end]);
        } else if kind == b"IEND" {
            break;
        }
        pos = data_end + 4; // 跳过 CRC
    }

    if width == 0 || height == 0 || channels == 0 {
        return Err(PngError::Header);
    }

    // zlib 解压（含 2 字节头和 adler32 校验尾）。
    let raw = decompress_to_vec_zlib(&idat).map_err(|_| PngError::Inflate)?;

    // 反滤镜 + 转灰度。
    let bpp = channels as usize;
    let stride = width as usize * bpp;
    let expected = height as usize * (1 + stride);
    if raw.len() < expected {
        return Err(PngError::Defilter);
    }

    let mut gray: Vec<u8> = Vec::with_capacity(width as usize * height as usize);
    let mut prev = vec![0u8; stride];
    let mut off = 0usize;
    for _ in 0..height as usize {
        let filter = raw[off];
        off += 1;
        let row = &raw[off..off + stride];
        off += stride;
        let mut cur = vec![0u8; stride];
        for x in 0..stride {
            let rv = row[x];
            let a = if x >= bpp { cur[x - bpp] } else { 0 };
            let b = prev[x];
            let c = if x >= bpp { prev[x - bpp] } else { 0 };
            let recon = match filter {
                0 => rv,
                1 => rv.wrapping_add(a),
                2 => rv.wrapping_add(b),
                3 => rv.wrapping_add(((a as u16 + b as u16) >> 1) as u8),
                4 => rv.wrapping_add(paeth(a as i32, b as i32, c as i32) as u8),
                _ => return Err(PngError::Defilter),
            };
            cur[x] = recon;
        }
        // 转灰度。
        for i in 0..width as usize {
            let v = match channels {
                1 => cur[i] as u32,                   // 灰度
                3 => {                                // RGB
                    let r = cur[i * 3] as u32;
                    let g = cur[i * 3 + 1] as u32;
                    let bb = cur[i * 3 + 2] as u32;
                    (r * 77 + g * 150 + bb * 29) >> 8 // 近似 0.299/0.587/0.114
                }
                4 => {                                // RGBA
                    let r = cur[i * 4] as u32;
                    let g = cur[i * 4 + 1] as u32;
                    let bb = cur[i * 4 + 2] as u32;
                    (r * 77 + g * 150 + bb * 29) >> 8
                }
                _ => unreachable!(),
            };
            gray.push(v as u8);
        }
        prev = cur;
    }

    Ok((width, height, gray))
}

/// Paeth 预测器（滤镜 4）。
fn paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// 最近邻缩小灰度图，使最长边不超过 `max`（preprocess 的栈缓冲上限是 128）。
///
/// 比 `max` 小的图原样返回。
pub fn downscale_to_max(
    gray: &[u8],
    w: u32,
    h: u32,
    max: u32,
) -> (u32, u32, Vec<u8>) {
    let (w, h) = (w as usize, h as usize);
    let longest = w.max(h) as f32;
    if longest <= max as f32 {
        return (w as u32, h as u32, gray.to_vec());
    }
    let s = longest / max as f32;
    // no_std 下没有 f32::round，用 libm::roundf（四舍五入到整数）。
    let nw = (libm::roundf(w as f32 / s) as usize).max(1);
    let nh = (libm::roundf(h as f32 / s) as usize).max(1);
    let mut out = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        let sy = ((y as f32 + 0.5) * h as f32 / nh as f32) as usize;
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) * w as f32 / nw as f32) as usize;
            out.push(gray[sy * w + sx]);
        }
    }
    (nw as u32, nh as u32, out)
}
