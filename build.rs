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
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
}
