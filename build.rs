//! 构建脚本只为 `cortex-m-rt` 准备存储布局。
//!
//! 算子库是纯 Rust 的 `embedded-nn`，不需要交叉编译 C，所以这里没有 cc / bindgen，
//! 全部构建都只经过 rustc。

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

    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
}
