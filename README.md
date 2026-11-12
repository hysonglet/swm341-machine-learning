# swm341-ai —— 在 SWM341 上手写数字识别

用 Rust 在 Synwit SWM341（STAR-MC1，Armv8-M Mainline + DSP + 单精度 FPU，150 MHz，
512K flash / 64K SRAM + 8M SDRAM）上做**手写数字识别**。一步一步来，每个 `src/bin/` 都是一个
能独立烧录、能独立验证的小历程。

算子库用纯 Rust 的 [`embedded-nn`](https://crates.io/crates/embedded-nn) 0.2.3
（`#![no_std]`、零堆分配），交叉编译只走 `rustc`，不需要 `arm-none-eabi-gcc` /
CMake / bindgen。GUI 用 [`oxivgl`](https://crates.io/crates/oxivgl) 0.9.2（LVGL 9.5 的安全
Rust 绑定，底层 `oxivgl-sys` 用 cc + bindgen 从源码编译 LVGL C 库）。

## 路线图

| 步骤 | bin                    | 内容                                                                                                                                              | 状态 |
| :--- | :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :--- |
| 0    | `00_hello`             | RTT / 时钟 / 依赖链路的冒烟测试                                                                                                                   | ✅   |
| 1    | `01_nn_smoke`          | 定点算子自检 + cycles/MAC 基准                                                                                                                    | ✅   |
| 2a   | `02_sdram`             | SDRAM 容量与读写自检（数据/地址总线 + 整片 March）                                                                                                | ✅   |
| 2b   | `03_sd_image`          | 图像采集：SD 卡读 PNG（SDIO → hadris-fat 解析 FAT12/16/32 → PNG 解码 → 灰度），落到 SDRAM 堆上                                                    | ✅   |
| 3    | —                      | 预处理（no_std）：反色 → 包围盒裁剪 → 等比缩放 20×20 → 质心居中 28×28（`core_ml::preprocess`）                                                    | ✅   |
| 4    | `02_infer`             | int8 推理：784→10 Softmax 回归，`fully_connected_per_channel_s8`，权重 `include_bytes!` 进 flash；置信度用**真 f32 softmax**（不是 `softmax_s8`） | ✅   |
| 4b   | `host/`                | 训练+量化工具：从 by_field.zip 训练、per-channel 量化、导出权重；**int8 91.48% vs 浮点 91.49%**                                                   | ✅   |
| 5    | `04_lcd` / `05_oxivgl` | TFT-LCD 点亮（裸帧缓冲）+ 接入 oxivgl/LVGL 9.5，移植 `flush_cb`                                                                                   | ✅   |
| 6    | `06_digit_ui`          | 识别页面：内置样本轮播 + 概率条 + 真 f32 softmax（当前 `default-run`）                                                                            | ✅   |
| 7    | `07_sd_ui`             | 从 SD 卡**流式**读手写数字 PNG → 边显示边识别（每帧只驻留当前一张）                                                                               | ✅   |
| 8    | `08_draw_ui`           | 手绘数字识别：GT911 触摸实时手写 → 预处理 → 推理 → 概率条                                                                                         | ✅   |

## 目录结构

```
.
├── Cargo.toml            # 多 bin：每个历程一个可执行文件；依赖 core_ml + embedded-nn + oxivgl
├── build.rs              # memory.x 准备；libSDL2.a 链接存根；06_digit_ui 的精简样本 samples_ui.bin
├── memory.x              # 512K flash @0x0 / 64K SRAM @0x2000_0000 / 8M SDRAM @0x8000_0000
├── .cargo/config.toml    # 默认 target、runner（probe-rs + RTT）、LVGL 构建环境变量
├── conf/                 # lv_conf.h（LVGL 9.5 配置，DEP_LV_CONFIG_PATH 指向这里）
├── core_ml/              # 与硬件无关的纯 Rust 推理核心（no_std，被 host/ 与固件共用）
│   └── src
│       ├── lib.rs       # N_PIXELS / N_CLASSES / 模块导出
│       ├── preprocess.rs # 第 3 步：灰度 → 28×28（反色/裁剪/缩放/质心居中）
│       └── model.rs     # int8 Softmax 回归：fully_connected_per_channel_s8 + CML1 权重格式
├── host/                # 宿主机工具：训练 + 量化 + 验证（跑在 Mac 上，不在 SWM341）
│   └── src/main.rs      # 读 by_field.zip → 训练 → per-channel 量化 → 导出 gen/
│   └── .cargo/config.toml # 覆盖根目录的 thumb target，改回本机三元组
├── gen/                 # host 生成的产物（gitignore 友好）：model.bin / samples.bin
├── vendor/
│   └── oxivgl-build/    # 用本地 no-op 版本顶掉 oxivgl-build 默认生成的示例图片资源
├── docs/                # 设计文档与框图（drawio / png / jpg / md）
└── src
    ├── lib.rs            # 公共库：bench / check / nn / heap / sd / image / core_ml 再导出
    ├── nn.rs             # embedded-nn 的再导出 + 恒等量化常量
    ├── check.rs          # 板端 PASS/FAIL 断言（no_std 没有 test harness）
    ├── bench.rs          # DWT 周期计数封装
    ├── heap.rs           # SDRAM 上的全局分配器（embedded-alloc），必须 SDRAM 上电后 init
    ├── sd.rs             # SD 卡封装：在 swm341-hal 的 SDIO 之上实现 embedded_io 的 Read+Seek（hadris-fat 块设备抽象）
    ├── image.rs          # PNG 解码（miniz_oxide）+ 灰度 + 画板灰度缓冲
    └── bin
        ├── 00_hello.rs
        ├── 01_nn_smoke.rs
        ├── 02_sdram.rs
        ├── 02_infer.rs   # 端侧手写数字识别：加载权重 → int8 推理 + 真 f32 softmax + 周期统计
        ├── 03_sd_image.rs # 2b：SD 卡读 PNG → 解码 → 预处理 → 推理，RTT 打印
        ├── 04_lcd.rs     # LCD 点亮（裸帧缓冲）
        ├── 05_oxivgl.rs  # 接入 oxivgl/LVGL 9.5 + flush_cb 移植
        ├── 06_digit_ui.rs # 识别页面（内置样本轮播），default-run
        ├── 07_sd_ui.rs   # SD 卡流式读图 + 识别
        └── 08_draw_ui.rs # 触摸手绘数字实时识别（GT911）
```

## 跑起来

```bash
# 目标三元组（一次性）
rustup target add thumbv8m.main-none-eabihf

# 冒烟
cargo run --bin 00_hello

# 算子自检（RTT 打印，probe-rs 直接挂在终端里）
cargo run --bin 01_nn_smoke
# 想要真实速度数字就用 release：
cargo run --release --bin 01_nn_smoke

# SDRAM 容量 + 读写自检
cargo run --bin 02_sdram

# 端侧推理（先跑下面的 host 工具生成 gen/model.bin 与 gen/samples.bin）
cargo run --release --bin 02_infer

# 2b：SD 卡读 PNG → 解码 → 预处理 → 推理（详见下方「SD 卡 + PNG 怎么跑」）
cargo run --bin 03_sd_image

# 5：LCD 点亮
cargo run --bin 04_lcd
# 5+：接入 oxivgl / LVGL 9.5
cargo run --bin 05_oxivgl

# 6：手写数字识别页面（默认烧录项，轮播内置样本）
cargo run --bin 06_digit_ui

# 7：从 SD 卡流式读 PNG 并识别
cargo run --bin 07_sd_ui

# 8：触摸手绘数字实时识别（GT911 + 画板）
cargo run --bin 08_draw_ui

# 只编译不烧录
cargo build --release --bins
```

### 训练 + 量化：先在 Mac 上跑 `host/`

```bash
cd host
cargo run --release                 # 默认每类 1500 训练 / 1000 测试
cargo run --release -- --train 2000 --test 1000   # 想冲更高精度就加量
```

它会从 `../data/by_field.zip` 抽样本，用 **与板端同一份** `core_ml::preprocess` 预处理，
训练 Softmax 回归（照搬 hardwrite 第 6 章），per-channel 量化成 int8，并验证：
**int8 测试准确率 ≈ 浮点测试准确率（实测 91.48% vs 91.49%）**。然后把权重与样例导出到
`../gen/model.bin`、`../gen/samples.bin`，供 `02_infer` 用 `include_bytes!` 烧进 flash。

> 关键：host 和固件调用的是**同一个 `embedded-nn` 算子**（`fully_connected_per_channel_s8`）。
> 所以主机端验证过的量化精度，板上必然一致——只要喂进去的 28×28 像素字节相同。

`swm341-hal` 目前用 `path = "../swm341-hal"` 引用，按需改成 git 依赖。

> LVGL 构建依赖（`.cargo/config.toml`）：`DEP_LV_CONFIG_PATH` 指向 `conf/` 提供 `lv_conf.h`；
> `LVGL_SRC_DIR` 指向本地已打补丁的 LVGL 源码树（默认 `target/lvgl-src/lvgl-9.5.0`），避免
> 重复下载 97 MB tarball；`BINDGEN_EXTRA_CLANG_ARGS` 补 macOS SDK 头文件路径。
> `vendor/oxivgl-build` 用本地 no-op 版本顶掉 `oxivgl-build` 默认生成的示例图片资源
> （占 flash、需 pypng/lz4）。首次构建 LVGL 较慢，属正常现象。

## LCD / LVGL / 触摸 怎么接的

- **帧缓冲**：SDRAM 分区为 `0x8000_0000..0x8040_0000` 给 Rust 堆（embedded-alloc），
  `0x8040_0000` 起放 800×480×2 = 768 KiB 的 LCDC 帧缓冲（`04_lcd`/`05_oxivgl` 里 `FB_BASE =
HEAP_BASE + HEAP_SIZE`），其上是 LVGL 的 stripe draw buffer（`05_oxivgl` 起）。
- **`flush_cb` 移植**：LVGL 用 PARTIAL 模式，只负责画脏矩形；`flush_cb` 把每 40 行 stripe
  逐行 copy 到 `FB_BASE`，`flush_ready` 立刻返回，不切双缓冲、不靠中断。代价是一次
  SDRAM→SDRAM 行拷贝，对 800×480@30fps 完全够用。
- **字体**：LVGL 内置 Montserrat 不含中文，页面文案一律英文/数字；字号种类刻意收敛
  （`06` 只引 14/20/36 三套），未引用字体靠 `--gc-sections` 去掉以省 flash。
- **触摸**：`08_draw_ui` 用 GT911（I2C1，参考 `swm341-hal/examples/i2c_block.rs`）轮询触点，
  坐标线性映射到 LCD 像素；若镜像/偏移，改文件顶部 `TP_SWAP` / `TP_FLIP_X` / `TP_FLIP_Y` 或
  按 RTT 首帧 raw 坐标校准 `TP_X_MAX/TP_Y_MAX`。

## 算子自检怎么设计的

**1. 把量化参数"关掉"，让期望值可手算。**
`embedded-nn::support::requantize(val, mult, shift)` 在 `mult = 0x7FFF_FFFF`、`shift = 0`
时是恒等映射：

```
(val * (2^31 - 1) + 2^30) >> 31 == val     (|val| < 2^30 时成立)
```

所以自检里所有卷积/全连接都用这个"1.0 乘子"，期望值就是朴素的整数和，例如
4×4 输入 + 2×2 全 1 核 → 3×3 窗口和 `[14, 18, 22, 30, 34, 38, 46, 50, 54]`。
算错了一眼能看出来是维度/零点约定错了，而不是"数学没算明白"。

**2. softmax 用外来标准答案。**
直接搬 CMSIS-NN 官方单测 `TestData/softmax` 的输入和 TFLite 参考输出
（`mult = 1077952640, shift = 19, diff_min = -3968`），逐字节比对；另外再验一次
"概率随输入单调不减"和"`(out + 128)` 之和 ≈ 256"。

**3. 顺带把 `requantize` 本身也验一遍**，它是所有定点算子的公共底座。

> 注意：上面这套 softmax 自检是**算子库层面**的验证（用 `softmax_s8` 做参照）。
> 固件端 `02_infer` 及其后的识别页面**不**用 `softmax_s8` 出概率——它的标度不标准
> （0.882 下限，给不出真概率），而是用 `softmax_confidence` 做**真 f32 softmax**：
> `z = logits * S_out`（`S_out` 从 `model.bin` 偏移 16 读取），再用 `libm::expf` 算，判类
> 仍靠 `argmax(logits)`，与量化无关。

## 关于 DSP 指令

SWM341 的 STAR-MC1 有 DSP 扩展，`embedded-nn` 里也有一段 `smlad` 的内联汇编，
但它被 `#[cfg(target_feature = "dsp")]` 包着。实测（rustc 1.93.1）：

```
$ rustc --print cfg --target thumbv8m.main-none-eabihf -C target-feature=+dsp
# 没有任何 target_feature 行
```

ARM 的 `dsp` 在 stable rustc 里属于 unstable target feature，**不会**出现在 `cfg` 里，
因此那段汇编在 stable 工具链上是死代码；即使加 `-C target-feature=+dsp` 重新编译，
固件里 `smlad` 的数量仍然是 0（LLVM 也没有把这个 4 路展开的点积 pattern 合成 SMLAD）。
当前实际跑的是**标量 4 路展开**内核，板端真实 cycles/MAC 由 `01_nn_smoke` 打印出来。

后续要真正吃到 SIMD，有三条路：nightly + `arm_target_feature`、自己用
`core::arch::asm!` 写 SMLAD 内核、或者回到 C 版 CMSIS-NN。三条都不影响上层模型代码，
先按标量跑通整条链路。

### 实测性能（dev 构建，150 MHz，标量内核）

`01_nn_smoke` 用 DWT 量出来的真实数字：

| 算子   | 形状              | cycles/MAC |        吞吐 |
| :----- | :---------------- | ---------: | ----------: |
| 全连接 | 128 → 32          |      11.26 | 13.3 MMAC/s |
| 卷积   | 16×16×1 → 4×(3×3) |      83.52 |  1.8 MMAC/s |

这是 **dev 配置**（`opt-level="s"`、未开 LTO）的数字；release（LTO）大约还能再快一截。
卷积的 83 cycles/MAC 偏高，是标量逐输出通道循环 + 每点 requantize 的代价。给模型选型
一个手感：比如第一层 28×28×1 → 8×(3×3)，约 4.9 万 MAC，按 83 算 ≈ 4 M 周期 ≈ 27 ms；
整网（几层卷积 + 一个 128 宽全连接）大概几百毫秒级，做"手写完→识别"的演示够用，
但别指望高帧率视频流。要榨性能就回到上面三条 SIMD 路径之一。

`02_infer` 的 FC 只有 7840 MAC，纯算力不是瓶颈，瓶颈在摄像头/屏；板上端侧准确率实测
≈ 91.5%（与 host 验证一致）。

## SD 卡 + PNG 怎么跑（`03_sd_image` / `07_sd_ui`）

完整链路：**SDIO 枚举 SD 卡 → hadris-fat 挂卷（自动识别 FAT12/16/32，支持长文件名）→
遍历目录挑 `.PNG` → `miniz_oxide` 解压 + 手写反滤镜 → 灰度 → `core_ml::preprocess` 成 28×28
→ `Model::predict`**（日志走 RTT，`07_sd_ui` 直接上屏）。

### 准备 SD 卡

```bash
# 1) 卡格式化成 FAT16 或 FAT32（exFAT 不支持）。
# 2) 根目录放 PNG（任意文件名，hadris-fat 支持长文件名）：N0.PNG … N9.PNG，或任意 XXXX.PNG。
#    PNG 要求：8-bit、非交错（no Adam7）、颜色类型 0/2/6（灰度/RGB/RGBA），
#    解码后转灰度再喂给预处理。手写数字建议白底黑字（见下）。
# 3) 07_sd_ui 约定：test/digit/30/*.png = 数字 0，31/*.png = 数字 1，……，39/*.png = 数字 9
#    （文件夹名 − 30 = 真实标签）；程序流式轮播全部目录。
```

> 文件系统用 [`hadris-fat`](https://crates.io/crates/hadris-fat) 2.5（纯 Rust、`no_std` 兼容，
> 支持 FAT12/16/32 读写 + 长文件名）。`SdCard` 在 `swm341-hal` 的 SDIO 之上实现
> `embedded_io` 0.7 的 `Read`+`Seek`，被 hadris-fat 当作随机访问字节流；堆落在 SDRAM 上。

### 运行输出

固件会先打印 SDRAM/堆就绪、SD 卡扇区数、FAT 挂载成功，然后逐张 PNG 打印尺寸、ASCII
预览（28×28 字符画）、预测数字与推理周期。找图方向：大多数手写数字 PNG 是**白底黑字**，
`03_sd_image.rs` 顶部 `INVERT = true` 会反色成「黑底白字」再喂预处理；若你的图已经是
黑底白字，把它改成 `false`。

### 踩过的坑

- **SDIO 初始化时序照搬 `swm341-hal/examples/sdio_block.rs`**：短响应在 `resp[0]`（不是
  `resp[3]`）、RCA 在 CMD3 响应的**高 16 位**、SDSC 卡要把块号×512 当 CMD17/CMD24 地址、
  数据相位时钟必须在发 CMD7 之前提到 ~18 MHz（否则数据引擎在识别时钟下超时）。这些在
  `src/sd.rs` 里都按示例对齐了。
- **堆必须在 SDRAM 上电后 `init`**（`src/heap.rs`）：顺序错会在第一次分配时 HardFault。
- **macOS 上 `probe-rs` 中途被杀会卡住 STLink 的 USB 接口**：现象是 `probe-rs run` 报
  `cannot perform this operation while interfaces are claimed`。拔插一下 STLink（或重启
  调试会话）即可，跟代码无关。
- **`embedded-io` 版本冲突**：hadris-io 用 0.7.1，而 `swm341-hal` 经 embassy 拉进 0.6.1，
  依赖图里会同时存在两版；根 `Cargo.toml` 直接依赖 `embedded-io = "0.7"` 才能对上，且
  `SdCardError` 需实现 `core::error::Error + Display` 并 `impl ErrorType for SdCard`。

## SDRAM 注意事项（踩过的坑）

- **150 MHz 主频必须 `ClkDiv::Div2`**：`embedded-nn` 这里无关，但 SDRAM 控制器在
  `Div1` 下只支持到 140 MHz，150 MHz 跑 `Div1` 会读出/写入垃圾。`Config::default()`
  用的是 `Div1`，`02_sdram` 里显式 `.clk_div(ClkDiv::Div2)` 覆盖了它。
- **容量要按板子实际配**：控制器的地址译码（`HADDR` → 行/列/库 的拆分）由
  `Config::size` 决定，配错会让整片读写错乱。`02_sdram` 顶部 `CONFIGURED_SIZE`
  默认 `Mb8`（SimplSDRAM 参考设计），板子不同就改这个常量。
- **Heap 必须在 SDRAM 初始化之后初始化**：`HEAP.init()` 会立刻往那块内存写空闲链表头，
  若 SDRAM 还没上电，写的是垃圾，第一次分配就 HardFault。顺序是 `Sdramc::new_with_pins`
  先、再 `HEAP.init` 后。

## 内存预算

release 实测（`arm-none-eabi-size`，`opt-level="s"` + LTO）：

| bin           |           text | 说明                                            |
| :------------ | -------------: | :---------------------------------------------- |
| `00_hello`    |         9.2 KB | 冒烟                                            |
| `01_nn_smoke` |        31.2 KB | 算子自检                                        |
| `02_sdram`    |         ~25 KB | SDRAM 自检                                      |
| `02_infer`    |         ~30 KB | 端侧推理                                        |
| `03_sd_image` | ~93 KB (debug) | SD + PNG 解码                                   |
| `05_oxivgl`   |        300+ KB | LVGL 9.5 接进来后明显增大（字体/控件）          |
| `06_digit_ui` |        300+ KB | 识别页面（build.rs 截 12 样本 ≈ 9 KB 进 flash） |

bss 里主要是 RTT 缓冲与 SDRAM 基地址常量（堆/帧缓冲在 SDRAM，不占 SRAM）。
28×28 输入 + 小型模型权重（几万字节）放进 flash 毫无压力，真正吃紧的是帧缓冲——
800×480×2 ≈ 768 KB，必须用板载 SDRAM，不能进 `.bss`。`03_sd_image` 把 PNG 解码缓冲
和 FAT 目录都放在 SDRAM 堆上（`src/heap.rs` 在 `0x8000_0000` 起 4 MiB）；`05` 之后的页面
把帧缓冲也放在堆之后。dev 配置下 `07_sd_ui`（LVGL + FAT + PNG）不开 LTO 会超 512K flash
约 6 KB，故 dev profile 也开了 `lto = true` 留出约 17K 余量。
