//! `oxivgl-build 0.1.2` 的本地精简替代（通过根 `Cargo.toml` 的
//! `[patch.crates-io]` 生效）。
//!
//! 上游 `oxivgl 0.9.2` 的 `build.rs` 会**无条件**调用三次 `image_asset()`，
//! 把 crate 自带的三个示例 PNG 用 `LVGLImage.py` 转成 C 数组并编译进静态库：
//!
//! 1. 该脚本在 import 阶段就强制要求 `pypng` + `lz4`，裸机固件构建因此要装
//!    Python 包（macOS 的 Homebrew Python 还有 PEP 668 限制）；
//! 2. 生成的 `img_cogwheel_argb` / `img_skew_strip` / `img_star` 只供 oxivgl
//!    自己的 examples 使用，我们的固件一个都不引用，却要占 512 KiB flash；
//! 3. 上游没有提供任何跳过开关（只有 `DOCS_RS`，但那会让 oxivgl-sys 改用
//!    x86_64 预生成 bindings，在 32 位目标上是错的）。
//!
//! 所以这里保留 API 形状、把 `image_asset` 变成 no-op。`emit_identity_mark`
//! 只在 xtensa 目标被调用，thumbv8m 用不到，但符号必须存在以便编译期解析。
//!
//! 若以后要用这些示例图片，删掉根 `Cargo.toml` 里的 `[patch.crates-io]` 段，
//! 并按 oxivgl 官方文档安装 `pypng` + `lz4` 即可。

/// 上游 `ImageConfig` 的占位类型。
pub struct ImageConfig;

impl ImageConfig {
    /// 上游会在这里读取 `DEP_LV_CONFIG_PATH`、定位 LVGL 源码树；no-op 版本不需要。
    pub fn from_env() -> Self {
        Self
    }

    /// 不做任何转换、不生成 C 文件、不链接额外目标文件。
    pub fn image_asset(&self, _name: &str, _png_path: &str) {}
}

/// 上游给 ESP32 示例写 build mark 用；本目标不链接这些信息。
pub fn emit_identity_mark(_prefix: &str, _hash_len: usize) {}
