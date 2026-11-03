//! 历程 3：从 SD 卡读 PNG → 解码 → 预处理 → int8 推理。
//!
//! 这一步把第 2b（图像采集）和第 4 步（推理）打通：SD 卡上放一些手写数字 PNG
//! （白底黑字，任意文件名），固件挂 FAT（hadris-fat，纯 Rust / `no_std`）、逐张解码、
//! 跑 `core_ml::preprocess` 成 28×28，再 `Model::predict`。日志全部走 RTT。
//!
//! ```text
//! cargo run --bin 03_sd_image
//! ```
//!
//! 依赖链：SDIO（`swm341_hal`）→ `SdCard` 实现 `embedded_io` 的 `Read`+`Seek`
//! （hadris-fat 的块设备抽象）→ hadris-fat 解析 FAT12/16/32 → PNG 解码（`image`），
//! 三者都用落在 SDRAM 上的堆（`alloc`）。

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use hadris_fat::sync::{DirectoryEntry, FatDir, FatVolume, FatVolumeReadExt};

use swm341_ai::heap::init_heap;
use swm341_ai::bench::CycleMeter;
use swm341_ai::core_ml::preprocess::preprocess;
use swm341_ai::core_ml::Model;
use swm341_ai::hal;
use swm341_ai::hal::sdramc::{ClkDiv, Config, Size, SdramPins, Sdramc};
use swm341_ai::image::decode_png;
use swm341_ai::hal::gpio::{Input, Pull};
use swm341_ai::sd::SdCard;

// 由 host/ 工具生成（先跑 `cd host && cargo run --release`）。
const MODEL_BIN: &[u8] = include_bytes!("../../gen/model.bin");

/// 手写数字图片大多是白底黑字，需要反色成「黑底白字」再喂预处理；若你的图已经是
/// 黑底白字，把它改成 `false`。
const INVERT: bool = true;
/// preprocess 的栈缓冲上限（28×28 路径里写死 128），超过就先最近邻缩小。
const MAX_DIM: u32 = 128;

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 03_sd_image: SD 卡读 PNG → 识别（hadris-fat）====");

    let p = hal::init(hal::Config::hrc_150mhz());

    // 1) SDRAM 上电（150 MHz 必须 Div2）。
    let _sdram = Sdramc::new_with_pins(
        p.sdramc(),
        SdramPins::default(),
        Config::default().size(Size::Mb8).clk_div(ClkDiv::Div2),
    );
    rprintln!("SDRAM 上线（堆将在其上分配）");

    // 2) 关键顺序：SDRAM 上电后才能 init_heap，否则第一次分配就 HardFault。
    unsafe { init_heap() };
    rprintln!("堆已就绪（{} KB @ SDRAM）", swm341_ai::heap::HEAP_SIZE / 1024);

    // 3) SD 卡上电 + 枚举。注意：先取走 dwt 句柄（p.dwt() 走 &mut self，不造成部分 move），
    // 因为下面 `p.PM2…` 是直接字段 move，之后 `p` 就不能再整体借用（含 p.dwt()）。
    let mut meter = CycleMeter::new(p.dwt());

    // 先取走 SDIO 控制器句柄（方法调用），再做各引脚的直接 move，避免部分 move 后再借用 p。
    let sdio = p.sdio();
    let pm2 = p.PM2;
    let pm4 = p.PM4;
    let pm5 = p.PM5;
    let pm6 = p.PM6;
    let pn0 = p.PN0;
    let pn1 = p.PN1;
    let pm3 = p.PM3;

    // 3b) 卡检测（PM3 = SD 座 CD 引脚，低电平表示插入）。先确认卡在位，再初始化。
    let cd = Input::new(pm3, Pull::Up);
    rprintln!(
        "CD(PM3)={} -> 卡{}",
        if cd.is_high() { "HIGH" } else { "LOW " },
        if cd.is_low() { "已插入" } else { "未插入" }
    );

    let sd = match SdCard::init(sdio, pm2, pm4, pm5, pm6, pn0, pn1) {
        Ok(s) => s,
        Err(e) => {
            rprintln!("✗ SD 卡初始化失败：{:?}", e);
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    rprintln!("SD 卡就绪：{} 扇区", sd.num_blocks());

    // 4) 挂 FAT（hadris-fat，自动识别 FAT12/16/32，支持长文件名）。
    rprintln!("→ 挂载 FAT（hadris-fat）…");
    let fs = match FatVolume::open(sd) {
        Ok(f) => f,
        Err(e) => {
            rprintln!("✗ FAT 挂载失败：{:?}（卡是否 FAT 格式化？）", e);
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    rprintln!("FAT 挂载成功：{:?}\n", fs.fat_type());

    // 5) 模型加载。
    let model = match Model::from_bytes(MODEL_BIN) {
        Some(m) => m,
        None => {
            rprintln!("✗ 解析 model.bin 失败！先跑 `cd host && cargo run --release`");
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    rprintln!("模型加载成功：784 → 10 全连接（int8）\n");

    // 6) 评测：test/digit/30、31、32 … 各子目录对应数字 0、1、2 …（label = 文件夹名 − 30）。
    //    hadris-fat 同步 API 无「按路径打开目录」，只能遍历父目录、按名匹配目录项、再 open_entry。
    let root = fs.root_dir();

    // 在根目录找并打开 test/。
    let test_dir = {
        let mut it = root.entries();
        let mut found: Option<FatDir<'_, SdCard<'_>>> = None;
        while let Some(res) = it.next_entry() {
            if let Ok(DirectoryEntry::Entry(f)) = res {
                if f.is_directory() && f.name().eq_ignore_ascii_case("TEST") {
                    found = root.open_entry(&f).ok();
                    break;
                }
            }
        }
        found
    };
    let test_dir = match test_dir {
        Some(d) => d,
        None => {
            rprintln!("✗ 找不到 test 目录（SD 卡根目录需有 test/）");
            loop {
                cortex_m::asm::wfi();
            }
        }
    };

    // 在 test/ 下找并打开 digit/。
    let digit_dir = {
        let mut it = test_dir.entries();
        let mut found: Option<FatDir<'_, SdCard<'_>>> = None;
        while let Some(res) = it.next_entry() {
            if let Ok(DirectoryEntry::Entry(f)) = res {
                if f.is_directory() && f.name().eq_ignore_ascii_case("DIGIT") {
                    found = test_dir.open_entry(&f).ok();
                    break;
                }
            }
        }
        found
    };
    let digit_dir = match digit_dir {
        Some(d) => d,
        None => {
            rprintln!("✗ 找不到 digit 目录（需有 test/digit/）");
            loop {
                cortex_m::asm::wfi();
            }
        }
    };

    // 遍历 test/digit/ 下的数字子目录（30, 31, 32 …），逐张预测并与真值比对，统计正确率。
    let mut per_correct = [0u32; 10];
    let mut per_total = [0u32; 10];
    let mut total = 0u32;
    let mut correct = 0u32;
    let mut cycle_sum = 0u64;

    let mut diter = digit_dir.entries();
    while let Some(dres) = diter.next_entry() {
        let dir_entry = match dres {
            Ok(e) => e,
            Err(e) => {
                rprintln!("目录项解析错误：{:?}", e);
                continue;
            }
        };
        // `DirectoryEntry` 目前只有 `Entry(FileEntry)` 一个变体。
        let DirectoryEntry::Entry(sub) = dir_entry;
        if !sub.is_directory() {
            continue;
        }
        // 文件夹名即数字（如 "30"），真值标签 = 名 − 30（30→0, 31→1, …）。
        let folder: u32 = match sub.name().trim().parse() {
            Ok(n) => n,
            Err(_) => {
                rprintln!("   跳过非数字目录：{}", sub.name());
                continue;
            }
        };
        let label = folder as i32 - 30;
        if !(0..=9).contains(&label) {
            rprintln!("   跳过越界标签目录：{} (label={})", sub.name(), label);
            continue;
        }
        let label = label as usize;

        let sub_dir = match digit_dir.open_entry(&sub) {
            Ok(d) => d,
            Err(e) => {
                rprintln!("   打开子目录 {} 失败：{:?}", sub.name(), e);
                continue;
            }
        };

        let mut siter = sub_dir.entries();
        while let Some(sres) = siter.next_entry() {
            let fentry = match sres {
                Ok(e) => e,
                Err(e) => {
                    rprintln!("   目录项解析错误：{:?}", e);
                    continue;
                }
            };
            let DirectoryEntry::Entry(file) = fentry;
            if file.is_directory() {
                continue;
            }
            let name = file.name(); // Cow<str>，长文件名优先、否则 8.3
            // 跳过 macOS 的 AppleDouble 资源文件（._xxx）等以 '.' 开头的隐藏文件。
            if name.starts_with('.') {
                continue;
            }
            if !name.to_ascii_uppercase().ends_with(".PNG") {
                continue;
            }

            let mut reader = match fs.read_file(&file) {
                Ok(r) => r,
                Err(e) => {
                    rprintln!("   {} 打开失败：{:?}", name, e);
                    continue;
                }
            };
            let bytes = match reader.read_to_vec() {
                Ok(b) => b,
                Err(e) => {
                    rprintln!("   {} 读取失败：{:?}", name, e);
                    continue;
                }
            };
            let (w, h, gray) = match decode_png(&bytes) {
                Ok(d) => d,
                Err(e) => {
                    rprintln!("   {} 解码失败：{:?}", name, e);
                    continue;
                }
            };
            // 超出 preprocess 栈上限就先缩小到 ≤128。
            let (w, h, gray) = {
                let (nw, nh, g) = swm341_ai::image::downscale_to_max(&gray, w, h, MAX_DIM);
                (nw, nh, g)
            };
            let img28 = match preprocess(&gray, w, h, INVERT) {
                Some(i) => i,
                None => {
                    rprintln!("   {} 预处理失败", name);
                    continue;
                }
            };

            let (_, cyc) = meter.time(|| {
                let _ = model.predict(&img28);
            });
            let pred = model.predict(&img28);
            cycle_sum += cyc as u64;
            total += 1;
            if pred == label {
                correct += 1;
                per_correct[label as usize] += 1;
            }
            per_total[label as usize] += 1;
            let mark = if pred == label { "OK" } else { "X" };
            rprintln!(
                "   {} 预测 {} 真值 {} {}  ({} cyc)",
                name, pred, label, mark, cyc
            );
        }
    }

    rprintln!(
        "\n评测完成：{}/{} 正确 = {:.1}%  平均推理 {} cyc",
        correct,
        total,
        if total > 0 {
            100.0 * correct as f32 / total as f32
        } else {
            0.0
        },
        if total > 0 { cycle_sum / total as u64 } else { 0 }
    );
    for d in 0..10 {
        if per_total[d] > 0 {
            rprintln!(
                "   数字 {}: {}/{} = {:.1}%",
                d,
                per_correct[d],
                per_total[d],
                100.0 * per_correct[d] as f32 / per_total[d] as f32
            );
        }
    }

    loop {
        cortex_m::asm::wfi();
    }
}
