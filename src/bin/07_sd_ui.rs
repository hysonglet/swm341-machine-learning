//! 历程 7：从 SD 卡读手写数字 PNG，**流式**解码 + 预处理 + 推理，边显示边预测。
//!
//! 与 `06_digit_ui` 的区别：数据来源从"内置样本"换成"SD 卡上的真实图片"；
//! 与早期 `07` 的区别：不再启动时把所有图全读进内存（那样几百张 128×128 会撑爆
//! 4 MiB SDRAM 堆），而是**只扫描目录收集文件元数据**，之后在主循环里按帧
//! **逐张流式加载**——内存里永远只有"当前这一张"，绝不缓存全部。
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Handwritten Digits (SD Card)    SWM341 / int8 FC / classify  │
//! │ ┌────────────────────┐ ┌───────────────────────────────────┐ │
//! │ │ INPUT / 28 x 28    │ │ INFERENCE                         │ │
//! │ │  ┌──────────────┐  │ │   7        92.4%      7           │ │
//! │ │  │ 模型输入预览 │  │ │ PREDICT   CONFIDENCE  TRUTH       │ │
//! │ │  └──────────────┘  │ │ ─────────────────────────────────  │ │
//! │ │  auto cycling    │ │ │ CLASS PROBABILITIES               │ │
//! │ └────────────────────┘ │ >0 ▓▓░░░░░░░░░  12%  ... 10 行 │ │
//! │  SD CARD: test/digit  |  42 images  |  classify OK           │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! SD 卡目录约定：`test/digit/30/*.png` 是数字 0，`31/*.png` 是数字 1，……，
//! `39/*.png` 是数字 9；文件夹名 − 30 = 真实标签。把"一章"图片放进对应目录即可
//! （"一章"= 一个标签目录，或几个目录一起；程序流式轮播全部）。
//!
//! 预处理用 `core_ml::preprocess`（任意尺寸灰度 → 28×28，背景0/笔画255，与 MNIST
//! 一致），推理用 `core_ml::Model`（int8 全连接），再用真 f32 softmax 出 10 类概率。
//!
//! ```text
//! cargo run --bin 07_sd_ui
//! ```

#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;

use cortex_m::asm::delay;
use cortex_m_rt::entry;
use libm;
use rtt_target::{rprintln, rtt_init_print};

// 临时：把 panic 信息打到 RTT，区分“死循环”和“panic”（二者在裸机上表现一样——
// 都只剩清屏背景色）。调试定位完可换回 panic_halt。
use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    rprintln!("PANIC: {:?}", info);
    loop {
        cortex_m::asm::wfi();
    }
}

use hadris_fat::sync::{DirectoryEntry, FatDir, FatVolume, FatVolumeReadExt, FileEntry};

use swm341_ai::core_ml::preprocess::preprocess;
use swm341_ai::core_ml::{Model, N_CLASSES, N_PIXELS};
use swm341_ai::hal;
use swm341_ai::hal::gpio::{Input, Level, Output, OutputConfig, Pull};
use swm341_ai::hal::lcdc::{Config, HvTiming, LayerConfig, Lcd, LcdPins, PixelFormat};
use swm341_ai::hal::sdramc::{CasLatency, SdramPins, Sdramc};
use swm341_ai::heap::{self, HEAP_BASE, HEAP_SIZE};
use swm341_ai::image::decode_png;
use swm341_ai::sd::SdCard;

use oxivgl::display::{Buffers, LvglBuffers, COLOR_BUF_LINES};
use oxivgl::draw_buf::{ColorFormat, DrawBuf};
use oxivgl::fonts::{MONTSERRAT_14, MONTSERRAT_20, MONTSERRAT_36};
use oxivgl::mem::reserve_pool_raw;
use oxivgl::style::{color_make, Selector, Style};
use oxivgl::view::{NavAction, RenderConfig, Ui, View};
use oxivgl::widgets::{AsLvHandle, Bar, Canvas, Child, Label, Obj, Part, WidgetError};
use oxivgl_sys::{
    lv_area_t, lv_canvas_get_draw_buf, lv_color_format_t_LV_COLOR_FORMAT_RGB565,
    lv_display_flush_ready, lv_display_get_default, lv_display_set_color_format,
    lv_display_set_flush_cb, lv_display_t, lv_draw_buf_flush_cache, lv_draw_buf_goto_xy,
};

const HCLK_HZ: u32 = 150_000_000;
const WIDTH: usize = 800;
const HEIGHT: usize = 480;

// ── SDRAM (8 MiB @ 0x8000_0000) 分区，与 05_oxivgl / 06_digit_ui 一致 ──────────
const FB_BASE: usize = HEAP_BASE + HEAP_SIZE;
const FB_BYTES: usize = WIDTH * HEIGHT * 2;
const STRIPE_BYTES: usize = WIDTH * COLOR_BUF_LINES * 2;
const STRIPE_BASE: usize = FB_BASE + FB_BYTES;
const LV_HEAP_BASE: usize = 0x8050_0000;
const LV_HEAP_SIZE: usize = 1024 * 1024;

// ── 配色（RGB hex）────────────────────────────────────────────────────────────
const BG: u32 = 0x0B1220;
const CARD: u32 = 0x141E2E;
const BORDER: u32 = 0x2A3A52;
const TEXT: u32 = 0xE8F0FF;
const MUTED: u32 = 0x7E93B0;
const ACCENT: u32 = 0x22D3A7;
const HIGHLIGHT: u32 = 0xFFB547;
const CANVAS_BG_RGB: (u8, u8, u8) = (0x1A, 0x24, 0x36);
const BAR_TRACK: u32 = 0x1B2839;

/// 画布尺寸（像素）。
const CANVAS_PX: i32 = 260;
/// 每隔多少毫秒换一张图（约 0.5 s）。
const MS_PER_IMAGE: u32 = 500;

/// 由 host/ 工具生成（先跑 `cd host && cargo run --release`）。
const MODEL_BIN: &[u8] = include_bytes!("../../gen/model.bin");

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 07_sd_ui: SD 卡读图 + 流式推理 ====");

    // 启用 DWT 周期计数器，供识别耗时计时（与随机种子共用）。
    enable_dwt();

    let p = hal::init(hal::Config {
        clock: hal::clock::ClockConfig::hrc_150mhz(),
    });

    // 先调用所有 peripheral 方法（各 move 自己的字段）；GPIO 字段的 move 统一放最后。
    let lcd_peri = p.lcd();
    let sdram_peri = p.sdramc();
    let sdio = p.sdio();

    let pm2 = p.PM2;
    let pm4 = p.PM4;
    let pm5 = p.PM5;
    let pm6 = p.PM6;
    let pn0 = p.PN0;
    let pn1 = p.PN1;
    let pm3 = p.PM3;

    // 1. SDRAM 上电。
    let _sdram = Sdramc::new_with_pins(
        sdram_peri,
        SdramPins::default(),
        hal::sdramc::Config::default()
            .cas_latency(CasLatency::Latency3)
            .trc(8),
    );
    rprintln!("SDRAM: {} bytes", _sdram.size_bytes());

    // 2. Rust 堆必须在 SDRAM 可访问之后初始化。
    unsafe { heap::init_heap() };

    // 3. LCDC 初始化（与 04/05/06 完全一致，已真机验证）。
    let lcd = Lcd::new_with_pins(lcd_peri, LcdPins::default());
    let mut lcd_rst = Output::new(p.PD1, OutputConfig::default().initial_level(Level::Low));
    delay(HCLK_HZ / 1000);
    lcd_rst.set_high();
    let mut lcd_bl = Output::new(p.PD9, OutputConfig::default().initial_level(Level::High));
    let _ = &mut lcd_bl;

    lcd.configure(&Config {
        clock_div: 3,
        format: PixelFormat::Rgb565,
        invert_clock: true,
        invert_vsync: false,
        invert_hsync: false,
        serial: false,
        mpu_enable: false,
        auto_restart: true,
        h: HvTiming {
            sync: 5,
            back_porch: 46,
            pixels: 800,
            front_porch: 64,
        },
        v: HvTiming {
            sync: 5,
            back_porch: 23,
            pixels: 480,
            front_porch: 22,
        },
        bg_color: 0xFFFFFF,
    });
    fill_solid(FB_BASE as *mut u16, rgb565(BG));
    lcd.set_layer(
        0,
        &LayerConfig {
            enable: true,
            x_start: 0,
            x_stop: WIDTH as u16 - 1,
            y_start: 0,
            y_stop: HEIGHT as u16 - 1,
            framebuffer: FB_BASE as u32,
            stride_bytes: WIDTH as u32 * 2,
            alpha: 255,
            color_key: None,
        },
    );
    lcd.start_frame();
    rprintln!("LCDC scanning FB_BASE=0x{:08X}", FB_BASE);

    // 4. LVGL 的 TLSF 堆（必须在 lv_init 之前注册）。
    unsafe { reserve_pool_raw(LV_HEAP_BASE as *mut u8, LV_HEAP_SIZE) }
        .expect("register LVGL SDRAM heap pool");

    assert_eq!(
        STRIPE_BASE % 16,
        0,
        "stripe buffers must be 16-byte aligned"
    );
    let stripes: &'static mut LvglBuffers<STRIPE_BYTES> =
        unsafe { &mut *(STRIPE_BASE as *mut LvglBuffers<STRIPE_BYTES>) };
    let ui = Ui::init(WIDTH as i32, HEIGHT as i32, Buffers::partial(stripes));

    // 5. 覆盖 display 的颜色格式 + 注册 flush_cb（详见 05_oxivgl 的说明）。
    unsafe {
        let disp = lv_display_get_default();
        assert!(!disp.is_null(), "lv_display_create returned NULL");
        lv_display_set_color_format(disp, lv_color_format_t_LV_COLOR_FORMAT_RGB565);
        lv_display_set_flush_cb(disp, Some(swm341_flush_cb));
    }

    // 6. SD 卡上电 + 枚举 + 只**扫描目录收集元数据**（不读图、不解码）。
    let cd = Input::new(pm3, Pull::Up);
    rprintln!(
        "CD(PM3)={} -> 卡{}",
        if cd.is_high() { "HIGH" } else { "LOW " },
        if cd.is_low() {
            "已插入"
        } else {
            "未插入"
        }
    );

    // 让编译器推断具体生命周期（SdCard 带 'd，不能手写标注）。
    let fs = match SdCard::init(sdio, pm2, pm4, pm5, pm6, pn0, pn1) {
        Ok(sd) => match FatVolume::open(sd) {
            Ok(f) => {
                rprintln!("FAT 挂载成功：{:?}", f.fat_type());
                Some(f)
            }
            Err(e) => {
                rprintln!("✗ FAT 挂载失败：{:?}", e);
                None
            }
        },
        Err(e) => {
            rprintln!("✗ SD 卡初始化失败：{:?}", e);
            None
        }
    };

    // 7. 模型（在 flash，不占堆）。
    let model: &'static Model = Box::leak(Box::new(
        Model::from_bytes(MODEL_BIN).expect("invalid gen/model.bin"),
    ));
    let s_out = f32::from_le_bytes(MODEL_BIN[16..20].try_into().expect("model.bin too short"));
    rprintln!("model loaded, s_out={}", s_out);

    // 8. 页面。leak 到 SDRAM 堆，生命周期与 display 一致。
    let view: &'static mut SdView = Box::leak(Box::new(SdView::new()));
    ui.bind(view, &RenderConfig::default())
        .expect("create LVGL widgets");
    rprintln!("UI 已 bind，进入主循环");

    // 9. 流式枚举器：持有 test/digit 目录 + 一个轻量 PRNG，每次随机抽一张不同数字的
    //    测试图。**绝不缓存文件名**：单张抽取用蓄水池抽样一遍扫描目录即可，因此无论
    //    一章有几百还是几万张图都不会 OOM。
    let mut streamer = match &fs {
        Some(f) => Streamer::new(f),
        None => None,
    };
    rprintln!(
        "DEBUG streamer.is_some={}, fs.is_some={}",
        streamer.is_some(),
        fs.is_some()
    );
    if streamer.is_none() {
        view.show_empty();
        rprintln!("✗ 无法打开 test/digit 目录（需有 test/digit/30..39/）");
    }

    // 10. 主循环：按帧计时，每 ~1.5s 流式取下一张 → 解码 → 预处理 → 推理 → 显示。
    let mut ticks: u32 = 0;
    let mut first = true;
    let mut shown: usize = 0;
    let mut hb = 0u32;
    loop {
        let ms = ui.timer_handler().clamp(1, 50);
        hb = hb.wrapping_add(1);
        if hb % 10 == 0 {
            rprintln!("HB{} ms={} first={} shown={}", hb, ms, first, shown);
        }
        ticks = ticks.saturating_add(ms);
        if first || ticks >= MS_PER_IMAGE {
            first = false;
            ticks = 0;
            rprintln!("DEBUG load tick (shown={})", shown);
            if let Some(st) = &mut streamer {
                rprintln!("DEBUG before next_image");
                match st.next_image() {
                    Some((label, entry)) => {
                        let name = entry.name().to_string();
                        rprintln!("DEBUG got {} label={}", name, label);
                        load_and_show(
                            fs.as_ref().unwrap(),
                            &entry,
                            label,
                            &name,
                            model,
                            s_out,
                            &mut *view,
                            shown,
                            0,
                        );
                        rprintln!("DEBUG load_and_show done");
                        shown += 1;
                    }
                    None => {
                        // 10 个数字目录都没有 PNG（不是"一轮播完"，随机流式不会自然到头）。
                        rprintln!("DEBUG next_image=None (没有可展示的图)");
                        view.show_empty();
                    }
                }
            } else {
                rprintln!("DEBUG no streamer -> show_empty");
                view.show_empty();
            }
        }
        let _ = view.update();
        delay((HCLK_HZ / 1000) * ms);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// flush_cb / 帧缓冲（与 05_oxivgl / 06_digit_ui 相同）
// ───────────────────────────────────────────────────────────────────────────

unsafe extern "C" fn swm341_flush_cb(
    disp: *mut lv_display_t,
    area: *const lv_area_t,
    px_map: *mut u8,
) {
    if disp.is_null() {
        return;
    }
    if !area.is_null() && !px_map.is_null() {
        let a = unsafe { &*area };
        let x1 = a.x1.clamp(0, WIDTH as i32 - 1) as usize;
        let y1 = a.y1.clamp(0, HEIGHT as i32 - 1) as usize;
        let x2 = a.x2.clamp(0, WIDTH as i32 - 1) as usize;
        let y2 = a.y2.clamp(0, HEIGHT as i32 - 1) as usize;
        if x2 >= x1 && y2 >= y1 {
            let w = x2 - x1 + 1;
            let h = y2 - y1 + 1;
            let src = px_map.cast::<u16>();
            let fb = FB_BASE as *mut u16;
            for row in 0..h {
                let s = unsafe { src.add(row * w) };
                let d = unsafe { fb.add((y1 + row) * WIDTH + x1) };
                for col in 0..w {
                    let px = unsafe { s.add(col).read_volatile() };
                    unsafe { d.add(col).write_volatile(px) };
                }
            }
        }
    }
    unsafe { lv_display_flush_ready(disp) };
}

fn fill_solid(fb: *mut u16, color: u16) {
    for i in 0..WIDTH * HEIGHT {
        unsafe { fb.add(i).write_volatile(color) };
    }
}

/// 0xRRGGBB → 小端 RGB565（与 LCDC 的 `PixelFormat::Rgb565` 一致）。
const fn rgb565(hex: u32) -> u16 {
    let r = ((hex >> 16) & 0xFF) as u16;
    let g = ((hex >> 8) & 0xFF) as u16;
    let b = (hex & 0xFF) as u16;
    ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
}

// ───────────────────────────────────────────────────────────────────────────
// 流式枚举：持有 &fs + 当前目录迭代器，逐张产出 PNG 的（标签, 目录项）。
// 不在内存里缓存任何文件名 —— 这才是能扛几万张图而不 OOM 的关键。
// ───────────────────────────────────────────────────────────────────────────

/// xorshift32：无堆、无 std 的极简 PRNG。
fn next_rand(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// 启用 DWT 周期计数器（Cortex-M 标准做法；不支持时写无效，无副作用）。
/// 之后可用 `cyc()` 读取随 CPU 主频自增的 32 位计数，用来精确计时。
fn enable_dwt() {
    unsafe {
        let demcr = 0xE000_EDFC as *mut u32;
        let ctrl = 0xE000_1000 as *mut u32;
        // 打开 TRCENA 并启用 CYCCNT。
        let v = core::ptr::read_volatile(demcr);
        core::ptr::write_volatile(demcr, v | (1u32 << 24));
        let c = core::ptr::read_volatile(ctrl);
        core::ptr::write_volatile(ctrl, c | (1u32 << 0));
    }
}

/// 读 DWT 周期计数器（@ CPU 主频，见 `HCLK_HZ`）。两次相减即为经过的 CPU 周期数。
fn cyc() -> u32 {
    unsafe { core::ptr::read_volatile(0xE000_1004 as *const u32) }
}

/// 取一个随开机变化的种子：用 DWT 周期计数器，读不到就退回固定值
///（仍可用，只是每次开机顺序相同）。
fn make_seed() -> u32 {
    enable_dwt();
    cyc().wrapping_add(0x9E37_79B9) | 1
}

/// 目录遍历状态机：每次 `next_image` 随机挑一个数字目录，再在该目录里
/// 用蓄水池抽样随机抽一张 PNG。不缓存任何文件名 —— 单张抽取一遍扫描目录即可，
/// 内存恒定 = 一个临时 `FatDirIter`+ 当前这张。
struct Streamer<'a, 'b> {
    digit_dir: FatDir<'a, SdCard<'b>>,
    /// xorshift32 内部状态，非零即可。
    rng: u32,
}

impl<'a, 'b> Streamer<'a, 'b> {
    /// 打开 `test/digit` 目录，准备随机流式遍历其中的 30..39 子目录。
    fn new(fs: &'a FatVolume<SdCard<'b>>) -> Option<Self> {
        let root = fs.root_dir();
        let test = find_dir(&root, "TEST")?;
        let digit = find_dir(&test, "DIGIT")?;
        Some(Self {
            digit_dir: digit,
            rng: make_seed(),
        })
    }

    /// 取下一张 PNG 的（真实标签, 目录项）。返回 None 表示 10 个数字目录里都没有 PNG。
    /// 每次都随机选一个数字再随机选一张，保证展示顺序跨数字乱序。
    fn next_image(&mut self) -> Option<(u8, FileEntry)> {
        // 最多尝试 16 次随机数字，避开打开失败 / 为空的目录。
        for _ in 0..16 {
            let digit = (next_rand(&mut self.rng) % 10) as usize;
            if let Some(r) = self.random_png_in(digit) {
                return Some(r);
            }
        }
        None
    }

    /// 在 `digit` 对应的 3X 目录里用蓄水池抽样随机返回一张 PNG。
    fn random_png_in(&mut self, digit: usize) -> Option<(u8, FileEntry)> {
        let folder = format!("3{}", digit);
        let dir = self.digit_dir.open_dir(&folder).ok()?;
        let mut it = dir.entries();
        let mut chosen: Option<FileEntry> = None;
        let mut n: u32 = 0;
        while let Some(res) = it.next_entry() {
            let f = match res {
                Ok(DirectoryEntry::Entry(f)) => f,
                _ => continue,
            };
            let name = f.name().to_string();
            if f.is_directory()
                || name.starts_with('.')
                || !name.to_ascii_uppercase().ends_with(".PNG")
            {
                continue;
            }
            n += 1;
            // 蓄水池抽样：第 n 张以 1/n 概率替换当前选中 → 整体均匀随机。
            if next_rand(&mut self.rng) % n == 0 {
                chosen = Some(f.clone());
            }
        }
        chosen.map(|f| (digit as u8, f))
    }
}

/// 在 `parent` 目录里按名（忽略大小写）找一个子目录。
fn find_dir<'a, 'd>(parent: &FatDir<'a, SdCard<'d>>, want: &str) -> Option<FatDir<'a, SdCard<'d>>> {
    let mut it = parent.entries();
    while let Some(res) = it.next_entry() {
        if let Ok(DirectoryEntry::Entry(f)) = res {
            if f.is_directory() && f.name().eq_ignore_ascii_case(want) {
                return parent.open_entry(&f).ok();
            }
        }
    }
    None
}

// ───────────────────────────────────────────────────────────────────────────
// 推理 + 显示
// ───────────────────────────────────────────────────────────────────────────

/// 真 f32 softmax：`z = logits * s_out`，减最大值再 exp，得到可直接显示的 10 类概率。
fn probabilities(logits: &[i8; N_CLASSES], s_out: f32) -> [f32; N_CLASSES] {
    let mut z = [0.0f32; N_CLASSES];
    let mut zmax = f32::NEG_INFINITY;
    for k in 0..N_CLASSES {
        z[k] = logits[k] as f32 * s_out;
        if z[k] > zmax {
            zmax = z[k];
        }
    }
    let mut sum = 0.0f32;
    for k in 0..N_CLASSES {
        z[k] = libm::expf(z[k] - zmax);
        sum += z[k];
    }
    if sum > 0.0 {
        for k in 0..N_CLASSES {
            z[k] /= sum;
        }
    }
    z
}

/// 从 `fs` 读取 `entry` 指定的图像 → 解码 → 预处理 → 推理 → 交给 `view` 显示。
/// 任何一步失败只记日志返回，不阻塞轮播。
fn load_and_show(
    fs: &FatVolume<SdCard<'_>>,
    entry: &FileEntry,
    label: u8,
    name: &str,
    model: &Model,
    s_out: f32,
    view: &mut SdView,
    idx: usize,
    total: usize,
) {
    rprintln!("DEBUG las: read_file {}", name);
    let mut reader = match fs.read_file(entry) {
        Ok(r) => r,
        Err(e) => {
            rprintln!("   {} 打开失败：{:?}", name, e);
            return;
        }
    };
    rprintln!("DEBUG las: read_to_vec {}", name);
    let bytes = match reader.read_to_vec() {
        Ok(b) => b,
        Err(e) => {
            rprintln!("   {} 读取失败：{:?}", name, e);
            return;
        }
    };
    // 从这里开始计时：只统计「识别」耗时（解码 + 预处理 + 推理），不含上面的 SD 卡读取。
    let t0 = cyc();
    rprintln!("DEBUG las: decoded? len={}", bytes.len());
    let (w, h, gray) = match decode_png(&bytes) {
        Ok(d) => d,
        Err(e) => {
            rprintln!("   {} 解码失败：{:?}", name, e);
            return;
        }
    };
    if w == 0 || h == 0 {
        rprintln!("   {} 尺寸非法", name);
        return;
    }
    // 自动判断反色：平均亮度偏高说明是白底黑字，需要反色成 MNIST 风格（黑底白字）。
    let mut sum: u32 = 0;
    for &v in gray.iter() {
        sum += v as u32;
    }
    let avg = sum / (w * h) as u32;
    let inverse = avg > 128;

    let img28 = match preprocess(&gray, w, h, inverse) {
        Some(i) => i,
        None => {
            rprintln!("   {} 预处理失败（空图或超限）", name);
            return;
        }
    };

    let logits = model.infer(&img28);
    let probs = probabilities(&logits, s_out);
    // 识别耗时（不含 SD 卡读取）= 经过的 CPU 周期 / 主频(MHz) → 微秒。
    let rec_us = cyc().wrapping_sub(t0) / (HCLK_HZ / 1_000_000);
    if let Some(l) = &view.rec_l {
        let ms_x10 = rec_us / 100; // 0.1 ms 为单位，避免浮点格式化
        l.text(&format!("{}.{} ms", ms_x10 / 10, ms_x10 % 10));
    }
    rprintln!("DEBUG las: view.show {}", name);
    view.show(Some(label), &img28, &probs, name, idx, total);

    // 轻量确认：每张图打印一次预测结果（不会刷屏）。
    let mut pred = 0usize;
    for k in 1..N_CLASSES {
        if probs[k] > probs[pred] {
            pred = k;
        }
    }
    let p10 = (probs[pred] * 1000.0) as i32;
    rprintln!(
        "  ▶ {} 预测={} 置信度 {}.{}%",
        name,
        pred,
        p10 / 10,
        p10 % 10
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 页面
// ───────────────────────────────────────────────────────────────────────────

struct SdView {
    canvas: Option<Canvas<'static>>,
    pred: Option<Label<'static>>,
    conf: Option<Label<'static>>,
    truth: Option<Label<'static>>,
    file_l: Option<Label<'static>>,
    rec_l: Option<Label<'static>>,
    index_l: Option<Label<'static>>,
    status_l: Option<Label<'static>>,
    digit_labels: [Option<Label<'static>>; N_CLASSES],
    pct_labels: [Option<Label<'static>>; N_CLASSES],
    bars: [Option<Bar<'static>>; N_CLASSES],
}

impl SdView {
    fn new() -> Self {
        Self {
            canvas: None,
            pred: None,
            conf: None,
            truth: None,
            file_l: None,
            rec_l: None,
            index_l: None,
            status_l: None,
            digit_labels: core::array::from_fn(|_| None),
            pct_labels: core::array::from_fn(|_| None),
            bars: core::array::from_fn(|_| None),
        }
    }

    /// 把一张 28×28 模型输入图 + 10 类概率刷到页面。
    fn show(
        &mut self,
        truth: Option<u8>,
        img28: &[u8; N_PIXELS],
        probs: &[f32; N_CLASSES],
        name: &str,
        idx: usize,
        total: usize,
    ) {
        rprintln!("DBG show: canvas start");
        if let Some(c) = &self.canvas {
            draw_gray(c, &img28[..], 28, 28);
        }
        rprintln!("DBG show: canvas ok");

        let mut pred = 0usize;
        for k in 1..N_CLASSES {
            if probs[k] > probs[pred] {
                pred = k;
            }
        }
        let pct = |p: f32| (p * 100.0) as i32;

        if let Some(l) = &self.pred {
            l.text(&format!("{}", pred));
        }
        if let Some(l) = &self.conf {
            let p10 = (probs[pred] * 1000.0) as i32;
            l.text(&format!("{}.{}%", p10 / 10, p10 % 10));
        }
        if let Some(l) = &self.truth {
            let truth_txt = if let Some(t) = truth {
                format!("{}", t)
            } else {
                "-".to_string()
            };
            l.text(&truth_txt);
        }
        // 文件名可能很长，卡片宽度约 31 个 14pt 字符，超长截断。
        let shown = if name.len() > 30 {
            format!("{}…", &name[..29])
        } else {
            name.to_string()
        };
        if let Some(l) = &self.file_l {
            l.text(&shown);
        }
        if let Some(l) = &self.index_l {
            l.text(&if total == 0 {
                format!("{} / ?", idx)
            } else {
                format!("{}/{}", idx, total)
            });
        }
        if let Some(l) = &self.status_l {
            l.text(&format!(
                "SD CARD Directory: test/digit/3*/*.png | classify OK"
            ));
        }
        rprintln!("DBG show: labels ok");

        for k in 0..N_CLASSES {
            let marker = if k == pred { ">" } else { " " };
            if let Some(l) = &self.digit_labels[k] {
                l.text(&format!("{}{}", marker, k));
            }
            if let Some(l) = &self.pct_labels[k] {
                l.text(&format!("{}%", pct(probs[k])));
            }
            if let Some(b) = &self.bars[k] {
                b.set_value_raw(pct(probs[k]), false);
            }
        }
        rprintln!("DBG show: bars ok");
    }

    /// 无图 / 未插卡时的占位显示。
    fn show_empty(&mut self) {
        if let Some(c) = &self.canvas {
            clear_canvas(c);
        }
        if let Some(l) = &self.pred {
            l.text("-");
        }
        if let Some(l) = &self.conf {
            l.text("-");
        }
        if let Some(l) = &self.truth {
            l.text("-");
        }
        if let Some(l) = &self.file_l {
            l.text("NO IMAGES FOUND");
        }
        if let Some(l) = &self.rec_l {
            l.text("-");
        }
        if let Some(l) = &self.index_l {
            l.text("- / -");
        }
        if let Some(l) = &self.status_l {
            l.text("INSERT SD CARD with test/digit/3X/*.png");
        }
    }
}

impl View for SdView {
    fn create(&mut self, root: &Obj<'static>) -> Result<(), WidgetError> {
        let s_screen = Style::new(|s| {
            s.bg_color_hex(BG).bg_opa(255);
        });
        let s_card = Style::new(|s| {
            s.bg_color_hex(CARD)
                .bg_opa(255)
                .radius(10)
                .border_width(1)
                .pad_all(0)
                .border_opa(255)
                .border_color_hex(BORDER);
        });
        let s_rule = Style::new(|s| {
            s.bg_color_hex(BORDER).bg_opa(255).radius(1);
        });
        let s_title = Style::new(|s| {
            s.text_color_hex(TEXT).text_font(MONTSERRAT_20).bg_opa(0);
        });
        let s_head = Style::new(|s| {
            s.text_color_hex(ACCENT).text_font(MONTSERRAT_14).bg_opa(0);
        });
        let s_body = Style::new(|s| {
            s.text_color_hex(TEXT).text_font(MONTSERRAT_14).bg_opa(0);
        });
        let s_muted = Style::new(|s| {
            s.text_color_hex(MUTED).text_font(MONTSERRAT_14).bg_opa(0);
        });
        let s_big = Style::new(|s| {
            s.text_color_hex(HIGHLIGHT)
                .text_font(MONTSERRAT_36)
                .bg_opa(0);
        });
        let s_num = Style::new(|s| {
            s.text_color_hex(ACCENT).text_font(MONTSERRAT_20).bg_opa(0);
        });
        let s_bar_bg = Style::new(|s| {
            s.bg_color_hex(BAR_TRACK).bg_opa(255).radius(3);
        });
        let s_bar_ind = Style::new(|s| {
            s.bg_color_hex(ACCENT).bg_opa(255).radius(3);
        });

        root.add_style(&s_screen, Selector::DEFAULT);

        let card_l: Obj<'static> = Obj::new(root)?;
        card_l.pos(16, 56).size(320, 364);
        card_l.add_style(&s_card, Selector::DEFAULT);

        let card_r: Obj<'static> = Obj::new(root)?;
        card_r.pos(348, 56).size(436, 364);
        card_r.add_style(&s_card, Selector::DEFAULT);

        let card_b: Obj<'static> = Obj::new(root)?;
        card_b.pos(16, 430).size(768, 38);
        card_b.add_style(&s_card, Selector::DEFAULT);

        let rule_top: Obj<'static> = Obj::new(root)?;
        rule_top.pos(16, 46).size(768, 2);
        rule_top.add_style(&s_rule, Selector::DEFAULT);

        // 标题栏
        static_label(root, "Handwritten Digits (SD Card)", 20, 10, &s_title)?;
        static_label(root, "SWM341  /  int8 FC  /  classify", 470, 16, &s_muted)?;

        // 左卡片：模型输入预览（坐标相对 card_l 原点 (16,56)）
        static_label(&card_l, "INPUT / 28 x 28", 16, 12, &s_head)?;
        let buf = DrawBuf::create(CANVAS_PX as u32, CANVAS_PX as u32, ColorFormat::RGB565)
            .ok_or(WidgetError::LvglNullPointer)?;
        let canvas = Canvas::new(&card_l, buf)?;
        canvas.pos(20, 44).size(CANVAS_PX, CANVAS_PX);
        self.canvas = Some(canvas);
        static_label(&card_l, "auto cycling", 16, 320, &s_muted)?;

        // 右卡片：推理结果（坐标相对 card_r 原点 (348,56)）
        static_label(&card_r, "INFERENCE", 16, 12, &s_head)?;
        self.pred = Some(label(&card_r, "-", 22, 36, &s_big)?);
        static_label(&card_r, "PREDICT", 16, 80, &s_muted)?;
        self.conf = Some(label(&card_r, "-", 148, 40, &s_num)?);
        static_label(&card_r, "CONFIDENCE", 148, 80, &s_muted)?;
        self.truth = Some(label(&card_r, "-", 316, 40, &s_num)?);
        static_label(&card_r, "TRUTH", 316, 80, &s_muted)?;
        static_label(&card_r, "FILE:", 16, 98, &s_muted)?;
        self.file_l = Some(label(&card_r, "-", 16 + 48, 98, &s_body)?);
        static_label(&card_r, "REC(ms):", 16 + 48 + 128, 98, &s_muted)?;
        self.rec_l = Some(label(
            &card_r,
            "-",
            16 + 48 + 128 + 48 + 32 + 32,
            98,
            &s_body,
        )?);
        static_label(&card_r, "CLASS PROBABILITIES", 16, 120, &s_muted)?;

        // 10 行概率条：数字 / 条 / 百分比。
        for k in 0..N_CLASSES {
            let y = 155 + (k as i32) * 19;
            self.digit_labels[k] = Some(label(&card_r, &format!(" {}", k), 16, y, &s_body)?);
            let bar = Bar::new(&card_r)?;
            bar.pos(44, y + 2).size(296, 11);
            bar.add_style(&s_bar_bg, Selector::DEFAULT);
            bar.add_style(&s_bar_ind, Part::Indicator);
            bar.set_range_raw(0, 100);
            bar.set_value_raw(0, false);
            self.bars[k] = Some(bar);
            self.pct_labels[k] = Some(label(&card_r, "0%", 352, y, &s_body)?);
        }

        // 底部状态栏（坐标相对 card_b 原点 (16,430)）
        self.status_l = Some(label(&card_b, "initializing SD card…", 12, 10, &s_muted)?);

        let _ = (Child::new(card_l), Child::new(card_r), Child::new(card_b));
        let _ = (Child::new(rule_top),);

        // 首帧先画占位，主循环第一帧会立即加载第一张。
        self.show_empty();
        Ok(())
    }

    fn update(&mut self) -> Result<NavAction, WidgetError> {
        // 换图计时已移到主循环（因为加载需要从 SD 读图）。这里保持空操作即可：
        // 控件状态的变化（label/bars）在 show() 里已 invalidate，LVGL 会自动重绘。
        Ok(NavAction::None)
    }
}

/// 创建一个带样式的标签，交给调用方决定是否长期持有。
fn label(
    parent: &Obj<'static>,
    text: &str,
    x: i32,
    y: i32,
    style: &Style,
) -> Result<Label<'static>, WidgetError> {
    let l = Label::new(parent)?;
    l.text(text);
    l.pos(x, y);
    l.add_style(style, Selector::DEFAULT);
    Ok(l)
}

/// 纯装饰标签：由父级 LVGL 对象持有，抑制 Rust 侧 Drop。
fn static_label(
    parent: &Obj<'static>,
    text: &str,
    x: i32,
    y: i32,
    style: &Style,
) -> Result<(), WidgetError> {
    let _ = Child::new(label(parent, text, x, y, style)?);
    Ok(())
}

/// 把画板整块填成画布背景色（用于「无图」时清屏）。
/// `fill_bg` 是直接写 draw_buf 像素，不产生 draw task、不进 LVGL 的任务分发循环。
fn clear_canvas(canvas: &Canvas<'static>) {
    canvas.fill_bg(
        color_make(CANVAS_BG_RGB.0, CANVAS_BG_RGB.1, CANVAS_BG_RGB.2),
        255,
    );
}

/// 灰度 → RGB565（r = g = b）。
const fn gray565(v: u8) -> u16 {
    let v = v as u16;
    ((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3)
}

/// 把 28×28 灰度图（模型输入，背景0/笔画255）等比放大并居中画进 RGB565 canvas。
///
/// 旧实现按「同行连续相同灰阶」调 `CanvasLayer::draw_rect`：真实 PNG 反走样后
/// 一行里灰阶几乎各不相同，一张图会产生几百个 LVGL draw task，
/// `lv_canvas_finish_layer` 逐个排队/分发时会卡死在第一次 `view.show()`。
/// 现在直接拿 `lv_draw_buf_t` 的像素指针写 u16（整数最近邻放大）：零 draw task、
/// 零堆分配、零浮点，写完 flush cache + invalidate 一次即可被 LVGL 刷出。
fn draw_gray(canvas: &Canvas<'static>, gray: &[u8], w: u32, h: u32) {
    if w == 0 || h == 0 || gray.len() < (w * h) as usize {
        clear_canvas(canvas);
        return;
    }
    let buf = unsafe { lv_canvas_get_draw_buf(canvas.lv_handle()) };
    if buf.is_null() {
        return;
    }

    let side = CANVAS_PX as u32;
    let m = w.max(h);
    let dw = (w * side) / m;
    let dh = (h * side) / m;
    let ox = (side - dw) / 2;
    let oy = (side - dh) / 2;
    let bg = rgb565(
        ((CANVAS_BG_RGB.0 as u32) << 16) | ((CANVAS_BG_RGB.1 as u32) << 8) | CANVAS_BG_RGB.2 as u32,
    );

    // 每行取一次行首指针：stride 由 LVGL 决定，不能假设等于 side*2。
    for dy in 0..side {
        let row = unsafe { lv_draw_buf_goto_xy(buf, 0, dy) } as *mut u16;
        if row.is_null() {
            break;
        }
        for dx in 0..side {
            unsafe { row.add(dx as usize).write_volatile(bg) };
        }
    }

    for dy in 0..dh {
        let sy = (dy * h) / dh;
        if sy >= h {
            continue;
        }
        let row = unsafe { lv_draw_buf_goto_xy(buf, ox, oy + dy) } as *mut u16;
        if row.is_null() {
            break;
        }
        for dx in 0..dw {
            let sx = (dx * w) / dw;
            if sx >= w {
                continue;
            }
            let v = gray[(sy * w + sx) as usize];
            unsafe { row.add(dx as usize).write_volatile(gray565(v)) };
        }
    }

    unsafe { lv_draw_buf_flush_cache(buf, core::ptr::null()) };
    canvas.invalidate();
}
