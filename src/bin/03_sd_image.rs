//! 历程 3：从 SD 卡读 PNG → 解码 → 预处理 → int8 推理。
//!
//! 这一步把第 2b（图像采集）和第 4 步（推理）打通：SD 卡上放一些手写数字 PNG
//! （8.3 短名，如 `n0.png`…`n9.png`，白底黑字），固件挂 FAT、逐张解码、跑
//! `core_ml::preprocess` 成 28×28，再 `Model::predict`。日志全部走 RTT。
//!
//! ```text
//! cargo run --bin 03_sd_image
//! ```
//!
//! 依赖链：SDIO（`swm341_hal`）→ 自写只读 FAT16/32（`fat`）→ PNG 解码（`image`），
//! 三者都用落在 SDRAM 上的堆（`alloc`）。

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::heap::init_heap;
use swm341_ai::bench::CycleMeter;
use swm341_ai::core_ml::preprocess::{ascii_art, preprocess};
use swm341_ai::core_ml::{Model, N_CLASSES, N_PIXELS};
use swm341_ai::fat::{from_8_3, FatFs};
use swm341_ai::hal;
use swm341_ai::hal::sdramc::{ClkDiv, Config, Size, SdramPins, Sdramc};
use swm341_ai::image::decode_png;
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
    rprintln!("\n==== 03_sd_image: SD 卡读 PNG → 识别 ====");

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
    let sd = match SdCard::init(p.sdio(), p.PM2, p.PM4, p.PM5, p.PM6, p.PN0, p.PN1) {
        Ok(s) => s,
        Err(e) => {
            rprintln!("✗ SD 卡初始化失败：{:?}", e);
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    rprintln!("SD 卡就绪：{} 扇区", sd.num_blocks());

    // 4) 挂 FAT。
    let mut fs = match FatFs::mount(sd) {
        Ok(f) => f,
        Err(e) => {
            rprintln!("✗ FAT 挂载失败：{:?}（卡是否 FAT16/32 格式化？）", e);
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    rprintln!("FAT 挂载成功");

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

    // 6) 遍历根目录，挑出所有 .PNG 逐张识别。
    let entries = match fs.root_entries() {
        Ok(e) => e,
        Err(e) => {
            rprintln!("✗ 读根目录失败：{:?}", e);
            loop {
                cortex_m::asm::wfi();
            }
        }
    };
    let mut n_png = 0usize;
    let mut n_ok = 0usize;
    for info in &entries {
        if &info.name[8..11] != b"PNG" {
            continue;
        }
        n_png += 1;
        let name = from_8_3(&info.name);
        rprintln!("--- {} ---", core::str::from_utf8(&name).unwrap_or("?.png"));

        let bytes = match fs.read_file(info) {
            Ok(b) => b,
            Err(e) => {
                rprintln!("   读文件失败：{:?}", e);
                continue;
            }
        };
        let (w, h, gray) = match decode_png(&bytes) {
            Ok(d) => d,
            Err(e) => {
                rprintln!("   PNG 解码失败：{:?}", e);
                continue;
            }
        };
        rprintln!("   尺寸 {}x{} 灰度 {} 字节", w, h, gray.len());

        // 超出 preprocess 栈上限就先缩小到 ≤128。
        let (w, h, gray) = {
            let (nw, nh, g) = swm341_ai::image::downscale_to_max(&gray, w, h, MAX_DIM);
            (nw, nh, g)
        };

        let img28 = match preprocess(&gray, w, h, INVERT) {
            Some(i) => i,
            None => {
                rprintln!("   预处理失败（空图？）");
                continue;
            }
        };

        ascii_art(&img28, |line| rprintln!("{}", line));

        let (_, cyc) = meter.time(|| {
            let _ = model.predict(&img28);
        });
        let pred = model.predict(&img28);
        n_ok += 1;
        rprintln!(
            "   预测：{}   推理 {} cycles (~{:.1} MMAC/s)",
            pred,
            cyc,
            (N_PIXELS as f32 * N_CLASSES as f32) / (cyc as f32 / 150_000_000.0) / 1e6
        );
    }

    rprintln!(
        "\n扫描完成：根目录 {} 个 PNG，成功识别 {} 个",
        n_png,
        n_ok
    );

    loop {
        cortex_m::asm::wfi();
    }
}
