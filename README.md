# swm341-ai —— 在 SWM341 上手写数字识别

用 Rust 在 Synwit SWM341（STAR-MC1，Armv8-M Mainline + DSP + 单精度 FPU，150 MHz，
512K flash / 64K SRAM）上做**手写数字识别**。一步一步来，每个 `src/bin/` 都是一个
能独立烧录、能独立验证的小历程。

算子库用纯 Rust 的 [`embedded-nn`](https://crates.io/crates/embedded-nn) 0.2.3
（`#![no_std]`、零堆分配），交叉编译只走 `rustc`，不需要 `arm-none-eabi-gcc` /
CMake / bindgen。

## 路线图

| 步骤 | 内容 | 状态 |
| :--- | :--- | :--- |
| 0 | `00_hello`：RTT / 时钟 / 依赖链路的冒烟测试 | ✅ |
| 1 | `01_nn_smoke`：定点算子自检 + cycles/MAC 基准 | ✅ |
| 2a | `02_sdram`：SDRAM 容量与读写自检（数据/地址总线 + 整片 March） | ✅ |
| 2b | 图像采集：**SD 卡读 PNG**（SDIO → hadris-fat 解析 FAT12/16/32 → PNG 解码 → 灰度），落到 SDRAM 堆上 | ✅ |
| 3 | 预处理（no_std）：反色 → 包围盒裁剪 → 等比缩放 20×20 → 质心居中 28×28（`core_ml::preprocess`） | ✅ |
| 4 | int8 推理：784→10 Softmax 回归，`fully_connected_per_channel_s8` + `softmax_s8`，权重 `include_bytes!` 进 flash（`02_infer`） | ✅ |
| 4b | 训练+量化工具 `host/`：从 by_field.zip 训练、per-channel 量化、导出权重；**int8 91.48% vs 浮点 91.49%** | ✅ |
| 5 | 完整应用：LCD 上边写边识别、结果回显、帧率/延迟统计 | ⬜ |

## 目录结构

```
.
├── Cargo.toml            # 多 bin：每个经历一个可执行文件；依赖 core_ml + embedded-nn
├── build.rs              # 只负责给 cortex-m-rt 准备 memory.x
├── memory.x              # 512K flash @0x0 / 64K SRAM @0x2000_0000
├── .cargo/config.toml    # 默认 target、runner（probe-rs + RTT）
├── core_ml/              # 与硬件无关的纯 Rust 推理核心（no_std，被 host/ 与固件共用）
│   └── src
│       ├── preprocess.rs # 第 3 步：灰度 → 28×28（反色/裁剪/缩放/质心居中）
│       └── model.rs     # int8 Softmax 回归：fully_connected_per_channel_s8 + CML1 权重格式
├── host/                # 宿主机工具：训练 + 量化 + 验证（跑在 Mac 上，不在 SWM341）
│   └── src/main.rs      # 读 by_field.zip → 训练 → per-channel 量化 → 导出 gen/
│   └── .cargo/config.toml # 覆盖根目录的 thumb target，改回本机三元组
├── gen/                 # host 生成的产物（gitignore 友好）：model.bin / samples.bin
└── src
    ├── lib.rs            # 公共库：bench / check / nn / heap / sd / image / core_ml 再导出
    ├── nn.rs             # embedded-nn 的再导出 + 恒等量化常量
    ├── check.rs          # 板端 PASS/FAIL 断言（no_std 没有 test harness）
    ├── bench.rs          # DWT 周期计数封装
    ├── heap.rs           # SDRAM 上的全局分配器（embedded-alloc），必须 SDRAM 上电后 init
    ├── sd.rs             # SD 卡封装：在 swm341-hal 的 SDIO 之上实现 embedded_io 的 Read+Seek（hadris-fat 块设备抽象）
    └── bin
        ├── 00_hello.rs
        ├── 01_nn_smoke.rs
        ├── 02_sdram.rs
        ├── 02_infer.rs   # 端侧手写数字识别：加载权重 → int8 推理 + softmax + 周期统计
        └── 03_sd_image.rs # 2b 收尾：SD 卡读 PNG → 解码 → 预处理 → 推理，RTT 打印
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

# 2b 收尾：SD 卡读 PNG → 解码 → 预处理 → 推理（详见下方「SD 卡 + PNG 怎么跑」）
cargo run --bin 03_sd_image

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

## 关于 DSP 指令（重要，别被 README 忽悠）

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

| 算子 | 形状 | cycles/MAC | 吞吐 |
| :--- | :--- | ---: | ---: |
| 全连接 | 128 → 32 | 11.26 | 13.3 MMAC/s |
| 卷积 | 16×16×1 → 4×(3×3) | 83.52 | 1.8 MMAC/s |

这是 **dev 配置**（`opt-level="s"`、未开 LTO）的数字；release（LTO）大约还能再快一截。
卷积的 83 cycles/MAC 偏高，是标量逐输出通道循环 + 每点 requantize 的代价。给模型选型
一个手感：比如第一层 28×28×1 → 8×(3×3)，约 4.9 万 MAC，按 83 算 ≈ 4 M 周期 ≈ 27 ms；
整网（几层卷积 + 一个 128 宽全连接）大概几百毫秒级，做"手写完→识别"的演示够用，
但别指望高帧率视频流。要榨性能就回到上面三条 SIMD 路径之一。

## SD 卡 + PNG 怎么跑（`03_sd_image`）

完整链路：**SDIO 枚举 SD 卡 → hadris-fat 挂卷（自动识别 FAT12/16/32，支持长文件名）→
遍历根目录挑 `.PNG` → `miniz_oxide` 解压 + 手写反滤镜 → 灰度 → `core_ml::preprocess` 成 28×28
→ `Model::predict`**。日志全部走 RTT。

### 准备 SD 卡

```bash
# 1) 卡格式化成 FAT16 或 FAT32（exFAT 不支持）。
# 2) 根目录放 PNG（任意文件名，hadris-fat 支持长文件名）：N0.PNG … N9.PNG，或任意 XXXX.PNG。
#    PNG 要求：8-bit、非交错（no Adam7）、颜色类型 0/2/6（灰度/RGB/RGBA），
#   解码后转灰度再喂给预处理。手写数字建议白底黑字（见下）。
```

> 文件系统用 [`hadris-fat`](https://crates.io/crates/hadris-fat)（纯 Rust、`no_std` 兼容，
> 支持 FAT12/16/32 读写 + 长文件名）。`SdCard` 在 `swm341-hal` 的 SDIO 之上实现
> `embedded_io` 的 `Read`+`Seek`，被 hadris-fat 当作随机访问字节流；堆落在 SDRAM 上。

### 运行输出

固件会先打印 SDRAM/堆就绪、SD 卡扇区数、FAT 挂载成功，然后逐张 PNG 打印尺寸、ASCII
预览（28×28 字符画）、预测数字与推理周期。找图方向：大多数手写数字 PNG 是**白底黑字**，
`03_sd_image.rs` 顶部 `INVERT = true` 会反色成「黑底白字」再喂预处理；若你的图已经是
黑底白字，把它改成 `false`。

### 踩过的坑

* **SDIO 初始化时序照搬 `swm341-hal/examples/sdio_block.rs`**：短响应在 `resp[0]`（不是
  `resp[3]`）、RCA 在 CMD3 响应的**高 16 位**、SDSC 卡要把块号×512 当 CMD17/CMD24 地址、
  数据相位时钟必须在发 CMD7 之前提到 ~18 MHz（否则数据引擎在识别时钟下超时）。这些在
  `src/sd.rs` 里都按示例对齐了。
* **堆必须在 SDRAM 上电后 `init`**（`src/heap.rs`）：顺序错会在第一次分配时 HardFault。
* **macOS 上 `probe-rs` 中途被杀会卡住 STLink 的 USB 接口**：现象是 `probe-rs run` 报
  `cannot perform this operation while interfaces are claimed`。拔插一下 STLink（或重启
  调试会话）即可，跟代码无关。

## SDRAM 注意事项（踩过的坑）

* **150 MHz 主频必须 `ClkDiv::Div2`**：`embedded-nn` 这里无关，但 SDRAM 控制器在
  `Div1` 下只支持到 140 MHz，150 MHz 跑 `Div1` 会读出/写入垃圾。`Config::default()`
  用的是 `Div1`，`02_sdram` 里显式 `.clk_div(ClkDiv::Div2)` 覆盖了它。
* **容量要按板子实际配**：控制器的地址译码（`HADDR` → 行/列/库 的拆分）由
  `Config::size` 决定，配错会让整片读写错乱。`02_sdram` 顶部 `CONFIGURED_SIZE`
  默认 `Mb8`（SimplSDRAM 参考设计），板子不同就改这个常量。
* **Heap 必须在 SDRAM 初始化之后初始化**：这是第 5 步接 `embedded-alloc` 时的坑——
  `HEAP.init()` 会立刻往那块内存写空闲链表头，若 SDRAM 还没上电，写的是垃圾，
  第一次分配就 HardFault。顺序是 `Sdramc::new_with_pins` 先、再 `HEAP.init` 后。

## 内存预算

release 实测（`arm-none-eabi-size`）：

| bin | text | bss |
| :--- | ---: | ---: |
| `00_hello` | 9.2 KB | 2.3 KB |
| `01_nn_smoke` | 31.2 KB | 2.3 KB |
| `02_sdram` | ~25 KB | 2.3 KB |
| `02_infer` | ~30 KB | 2.3 KB |
| `03_sd_image` | ~93 KB (debug) | 2.3 KB |

bss 里主要是 RTT 缓冲与 4 MB SDRAM 堆的基地址常量（堆本身在 SDRAM，不占 SRAM）。
28×28 输入 + 小型 CNN 的权重（几万字节）放进 flash 毫无压力，真正吃紧的是第 2 步的帧缓冲
——640×480×2 ≈ 600 KB，必须用板载 SDRAM，不能进 `.bss`。`03_sd_image` 把 PNG 解码缓冲
和 FAT 目录都放在 SDRAM 堆上（`src/heap.rs` 在 `0x8000_0000` 起 4 MB）。
