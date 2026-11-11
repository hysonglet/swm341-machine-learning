//! 构建脚本：准备 `memory.x`、补 LVGL 链接存根、生成页面演示用的精简样本。
//!
//! LVGL 的 C 源码由 `oxivgl-sys` 自己的 build.rs（cc + bindgen）编译，这里不重复；
//! 算子库仍是纯 Rust 的 `embedded-nn`。

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::copy("memory.x", out.join("memory.x")).expect("copy memory.x");
    println!("cargo:rustc-link-search={}", out.display());

    // oxivgl-sys 的 build.rs 对**所有**非 ESP 目标无条件发 `-lSDL2`（它只在
    // __xtensa__ 下跳过）。即使我们在 lv_conf.h 里把 LV_USE_SDL 钉成 0、一个 SDL
    // 源文件都不编，这条 link 指令照样发出，rust-lld 会报 "unable to find
    // library -lSDL2"。
    //
    // 在 OUT_DIR 放一个合法的空 ar 存档作为存根：没有任何 SDL 符号被引用，
    // 空库既满足 -lSDL2，又不往镜像里加一个字节。OUT_DIR 本来就已在链接
    // 搜索路径里（见上面的 rustc-link-search）。
    fs::write(out.join("libSDL2.a"), b"!<arch>\n").expect("write empty libSDL2.a stub");

    // 06_digit_ui 的内嵌样本：`gen/samples.bin` 有 64 个样本 / 50 KB，整块塞进
    // .rodata 会把 512 KiB flash 撑爆（页面只需要轮播展示几张图）。这里截前 12 个
    // 生成精简版（每样本 1 字节 label + 784 字节图像），约 9 KB。02_infer 仍用
    // 完整的 gen/samples.bin 做精度统计，互不影响。
    const SAMPLE_STRIDE: usize = 1 + 28 * 28;
    let samples = fs::read("gen/samples.bin").expect("read gen/samples.bin");
    let n = (samples.len() / SAMPLE_STRIDE).min(12);
    fs::write(out.join("samples_ui.bin"), &samples[..n * SAMPLE_STRIDE])
        .expect("write samples_ui.bin");

    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=gen/samples.bin");
}
