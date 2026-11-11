//! 历程 6：手写数字识别页面（oxivgl / LVGL 9.5）。
//!
//! 在 `05_oxivgl` 已验证的 LCD + `flush_cb` 之上，做一个完整的识别页面：
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Handwritten Digit Recognition         SWM341 / int8 FC      │
//! │ ┌────────────────────┐ ┌───────────────────────────────────┐ │
//! │ │ INPUT / 28 x 28    │ │ INFERENCE                         │ │
//! │ │  ┌──────────────┐  │ │   7        92.4%      7           │ │
//! │ │  │              │  │ │ PREDICT   CONFIDENCE  TRUTH       │ │
//! │ │  │  28×28 → 10x │  │ │ ─────────────────────────────────  │ │
//! │ │  │              │  │ │ CLASS PROBABILITIES               │ │
//! │ │  └──────────────┘  │ │ >0 ▓▓░░░░░░░░░  12%  ... 10 行 │ │
//! │ │  sample #00 / 64   │ │                                   │ │
//! │ └────────────────────┘ └───────────────────────────────────┘ │
//! │  MODEL 784->10 int8  |  SOFTMAX 91.5%  |  LCD flush_cb OK     │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! 数据源是 `gen/samples.bin`（64 个 28×28 测试样本），推理用 `core_ml::Model`
//! 的 int8 全连接，再用真 f32 softmax 得到 10 类概率。页面每约 1.5 秒自动切
//! 到下一个样本：**当前板子没有触摸驱动**，所以先用轮播演示整条链路；接上
//! `PointerIndev` 后，把手写轨迹栅格化成 28×28，直接调
//! [`DigitView::show`] 就能复用同一套画板 + 推理 + 概率条刷新。
//!
//! 注意：LVGL 内置的 Montserrat 字体不含中文字形，页面文案一律用英文/数字。
//!
//! ```text
//! cargo run --bin 06_digit_ui
//! ```

#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use core::array::from_fn;
use cortex_m::asm::delay;
use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::core_ml::{Model, N_CLASSES, N_PIXELS};
use swm341_ai::hal;
use swm341_ai::hal::gpio::{Level, Output, OutputConfig};
use swm341_ai::hal::lcdc::{Config, HvTiming, LayerConfig, Lcd, LcdPins, PixelFormat};
use swm341_ai::hal::sdramc::{CasLatency, SdramPins, Sdramc};
use swm341_ai::heap::{self, HEAP_BASE, HEAP_SIZE};

use oxivgl::display::{Buffers, LvglBuffers, COLOR_BUF_LINES};
use oxivgl::draw::{Area, DrawRectDsc};
use oxivgl::draw_buf::{ColorFormat, DrawBuf};
// 只引用 3 套字号。LVGL 的 Montserrat 字体数据在 .rodata 里很占地方
// （48 号的 C 源码就 577 KB），每多引用一套就会把 512 KiB flash 吃掉一大块；
// 未被引用的字体靠 --gc-sections 去掉，所以这里刻意收敛字号种类。
use oxivgl::fonts::{MONTSERRAT_14, MONTSERRAT_20, MONTSERRAT_36};
use oxivgl::mem::reserve_pool_raw;
use oxivgl::style::{color_make, Selector, Style};
use oxivgl::view::{NavAction, RenderConfig, Ui, View};
use oxivgl::widgets::{Bar, Canvas, Child, Label, Obj, Part, WidgetError};
use oxivgl_sys::{
    lv_area_t, lv_color_format_t_LV_COLOR_FORMAT_RGB565, lv_display_flush_ready,
    lv_display_get_default, lv_display_set_color_format, lv_display_set_flush_cb, lv_display_t,
};

// 由 host/ 工具生成（先跑 `cd host && cargo run --release`）。
const MODEL_BIN: &[u8] = include_bytes!("../../gen/model.bin");
/// 每样本 = 1 字节 label + 784 字节图像。
/// 用 build.rs 从 `gen/samples.bin` 截出来的 12 样本精简版（约 9 KB），
/// 避免把整块 50 KB 样本塞进本就紧张的 flash。
const SAMPLES_BIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/samples_ui.bin"));
const SAMPLE_STRIDE: usize = 1 + N_PIXELS;
const N_SAMPLES: usize = SAMPLES_BIN.len() / SAMPLE_STRIDE;

const HCLK_HZ: u32 = 150_000_000;
const WIDTH: usize = 800;
const HEIGHT: usize = 480;

// ── SDRAM (8 MiB @ 0x8000_0000) 分区，与 05_oxivgl 一致 ─────────────────────
// 0x8000_0000 .. 0x8040_0000  Rust 堆（embedded-alloc，4 MiB）
// 0x8040_0000 .. 0x804B_B800  LCDC 扫描的帧缓冲（768 KiB）
// 0x804B_B800 .. 0x804D_AC00  LVGL 的两块 stripe draw buffer（2 * 40 行）
// 0x8050_0000 .. 0x8060_0000  LVGL TLSF 堆（1 MiB；canvas 的 157 KiB 从这里出）

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
const INK: (u8, u8, u8) = (0xF2, 0xF7, 0xFF);
const CANVAS_BG_RGB: (u8, u8, u8) = (0x1A, 0x24, 0x36);
const BAR_TRACK: u32 = 0x1B2839;

/// 画布尺寸：28×28 每个像素放大 10 倍。
const CANVAS_PX: i32 = 260;
const ZOOM: i32 = 10;
/// 灰度阈值：高于它就认为是笔画。
const INK_THRESHOLD: u8 = 96;
/// 每隔多少个 LVGL tick 换一个样本（约 1.5 s）。
const TICKS_PER_SAMPLE: u32 = 1;

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 06_digit_ui: LVGL 手写数字识别页面 ====");

    let p = hal::init(hal::Config {
        clock: hal::clock::ClockConfig::hrc_150mhz(),
    });

    let lcd_peri = p.lcd();
    let sdram_peri = p.sdramc();

    // 1. SDRAM 上电。
    let sdram = Sdramc::new_with_pins(
        sdram_peri,
        SdramPins::default(),
        hal::sdramc::Config::default()
            .cas_latency(CasLatency::Latency3)
            .trc(8),
    );
    rprintln!("SDRAM: {} bytes", sdram.size_bytes());

    // 2. Rust 堆必须在 SDRAM 可访问之后初始化（Style/Box/format! 都要用）。
    unsafe { heap::init_heap() };

    // 3. LCDC 初始化（与 04/05 完全一致，已真机验证）。
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

    // 6. 模型 / 页面。两者都 leak 到 SDRAM 堆，地址稳定、生命周期与 display 一致。
    let model: &'static Model = Box::leak(Box::new(
        Model::from_bytes(MODEL_BIN).expect("invalid gen/model.bin"),
    ));
    let s_out = f32::from_le_bytes(MODEL_BIN[16..20].try_into().expect("model.bin too short"));
    rprintln!("model loaded: {} samples, s_out={}", N_SAMPLES, s_out);

    let view: &'static mut DigitView = Box::leak(Box::new(DigitView::new(model, s_out)));
    ui.bind(view, &RenderConfig::default())
        .expect("create LVGL widgets");
    rprintln!("UI bound: canvas + 10 probability bars");

    loop {
        let _ = view.update();
        let ms = ui.timer_handler().clamp(1, 50);
        delay((HCLK_HZ / 1000) * ms);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// flush_cb / 帧缓冲（与 05_oxivgl 相同）
// ───────────────────────────────────────────────────────────────────────────

/// PARTIAL 模式：`px_map` 是紧凑的 `w*h` 像素（stride=`w`），目标帧缓冲
/// stride=`WIDTH`，必须逐行搬；拷完立刻 ready，因为 LCDC 在连续扫描这块内存。
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
// 推理数据
// ───────────────────────────────────────────────────────────────────────────

fn sample(i: usize) -> (u8, &'static [u8; N_PIXELS]) {
    let base = i * SAMPLE_STRIDE;
    let label = SAMPLES_BIN[base];
    let img: &'static [u8; N_PIXELS] = SAMPLES_BIN[base + 1..base + 1 + N_PIXELS]
        .try_into()
        .expect("bad sample stride");
    (label, img)
}

/// 真 f32 softmax：`z = logits * s_out`，减最大值再 exp，得到可直接显示的 10 类概率。
/// 绝不能用 `core_ml::softmax_s8`（它的标度不是概率，下限 0.882）。
fn probabilities(model: &Model, img: &[u8; N_PIXELS], s_out: f32) -> [f32; N_CLASSES] {
    let logits = model.infer(img);
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

// ───────────────────────────────────────────────────────────────────────────
// 页面
// ───────────────────────────────────────────────────────────────────────────

struct DigitView {
    model: &'static Model,
    s_out: f32,
    canvas: Option<Canvas<'static>>,
    pred: Option<Label<'static>>,
    conf: Option<Label<'static>>,
    truth: Option<Label<'static>>,
    sample_no: Option<Label<'static>>,
    digit_labels: [Option<Label<'static>>; N_CLASSES],
    pct_labels: [Option<Label<'static>>; N_CLASSES],
    bars: [Option<Bar<'static>>; N_CLASSES],
    idx: usize,
    ticks: u32,
}

impl DigitView {
    fn new(model: &'static Model, s_out: f32) -> Self {
        Self {
            model,
            s_out,
            canvas: None,
            pred: None,
            conf: None,
            truth: None,
            sample_no: None,
            digit_labels: from_fn(|_| None),
            pct_labels: from_fn(|_| None),
            bars: from_fn(|_| None),
            idx: 0,
            ticks: 0,
        }
    }

    /// 把一张 28×28 图（以及可选真值）喂给页面：重画画板 → 推理 → 刷新所有读数。
    /// 触摸输入接进来后也走这里，只是 `truth` 传 `None`。
    fn show(&mut self, truth: Option<u8>, img: &[u8; N_PIXELS]) {
        if let Some(c) = &self.canvas {
            draw_digit(c, img);
        }

        let probs = probabilities(self.model, img, self.s_out);
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
            // 一位小数（如 92.4%）：先放大 1000 倍再拆整数/小数部分。
            let p10 = (probs[pred] * 1000.0) as i32;
            l.text(&format!("{}.{}%", p10 / 10, p10 % 10));
        }
        if let Some(l) = &self.truth {
            let t = match truth {
                Some(t) => format!("{}", t),
                None => format!("-"),
            };
            l.text(&t);
        }
        if let Some(l) = &self.sample_no {
            l.text(&format!("sample #{:02} / {}", self.idx, N_SAMPLES));
        }

        for k in 0..N_CLASSES {
            // 预测类在数字前加 `>` 做高亮；其余行保持纯数字。Label::text 对
            // 相同字符串是 no-op，所以每轮只有两行会真正重绘。
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
    }
}

impl View for DigitView {
    fn create(&mut self, root: &Obj<'static>) -> Result<(), WidgetError> {
        // ── 样式（Style 是 Rc-backed，add_style 会 clone，局部变量可安全 drop）──
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

        // 静态装饰对象由父级 screen 拥有；用 Child 抑制 Rust 侧的 Drop，
        // 否则 Obj 析构会调 lv_obj_delete 把控件删掉。
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

        let rule_r: Obj<'static> = Obj::new(root)?;
        rule_r.pos(364, 186).size(404, 2);
        rule_r.add_style(&s_rule, Selector::DEFAULT);

        // 标题栏
        static_label(root, "Handwritten Digit Recognition", 20, 10, &s_title)?;
        static_label(root, "SWM341  /  int8 FC  /  91.5%", 540, 16, &s_muted)?;

        // 左卡片：输入 / 画板。
        // 注意：LVGL 的 pos() 是相对父对象的坐标，不是屏幕绝对坐标
        // （首次真机验证时这里全部按绝对坐标写，右卡片内容整体偏移了
        // 一个卡片原点跑出屏幕）。下面所有子控件坐标都相对所属卡片。
        static_label(&card_l, "INPUT / 28 x 28", 16, 12, &s_head)?;
        let buf = DrawBuf::create(CANVAS_PX as u32, CANVAS_PX as u32, ColorFormat::RGB565)
            .ok_or(WidgetError::LvglNullPointer)?;
        let canvas = Canvas::new(&card_l, buf)?;
        canvas.pos(20, 44).size(CANVAS_PX, CANVAS_PX);
        self.canvas = Some(canvas);
        self.sample_no = Some(label(&card_l, "sample #00 / 64", 16, 324, &s_body)?);
        static_label(
            &card_l,
            "DEMO - auto cycling (touch TBD)",
            16,
            340,
            &s_muted,
        )?;

        // 右卡片：推理结果（坐标相对 card_r 原点 (348,56)）
        static_label(&card_r, "INFERENCE", 16, 12, &s_head)?;
        self.pred = Some(label(&card_r, "-", 22, 36, &s_big)?);
        static_label(&card_r, "PREDICT", 22, 102, &s_muted)?;
        self.conf = Some(label(&card_r, "-", 148, 40, &s_num)?);
        static_label(&card_r, "CONFIDENCE", 148, 96, &s_muted)?;
        self.truth = Some(label(&card_r, "-", 316, 40, &s_num)?);
        static_label(&card_r, "TRUTH", 316, 96, &s_muted)?;
        static_label(&card_r, "CLASS PROBABILITIES", 16, 138, &s_muted)?;

        // 10 行概率条：数字 / 条 / 百分比。
        for k in 0..N_CLASSES {
            let y = 160 + (k as i32) * 19;
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
        static_label(
            &card_b,
            "MODEL 784->10 int8   |   F32 SOFTMAX   |   LCD flush_cb OK",
            12,
            10,
            &s_muted,
        )?;

        // 这些容器只被 LVGL 的父子关系持有，抑制 Rust 侧 Drop。
        let _ = (Child::new(card_l), Child::new(card_r), Child::new(card_b));
        let _ = (Child::new(rule_top), Child::new(rule_r));

        // 首帧立刻出图，不必等第一个轮播 tick。
        let (truth, img) = sample(self.idx);
        self.show(Some(truth), img);
        Ok(())
    }

    fn update(&mut self) -> Result<NavAction, WidgetError> {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % TICKS_PER_SAMPLE == 0 {
            self.idx = (self.idx + 1) % N_SAMPLES;
            let (truth, img) = sample(self.idx);
            self.show(Some(truth), img);
        }
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

/// 把 28×28 灰度图放大 10 倍画进 RGB565 canvas。
///
/// LVGL 9 的 `lv_canvas_set_px` 只对带 alpha 的格式有效，RGB565 必须用
/// `CanvasLayer::draw_rect`。这里把同一行里连续的笔画像素**合并成一个矩形**
/// （28 行通常只有几十段），避免产生几百个 draw task 撑爆 oxivgl 的内部
/// render scratch。
fn draw_digit(canvas: &Canvas<'static>, img: &[u8; N_PIXELS]) {
    let mut layer = canvas.init_layer();

    let mut bg = DrawRectDsc::new();
    bg.bg_color(color_make(
        CANVAS_BG_RGB.0,
        CANVAS_BG_RGB.1,
        CANVAS_BG_RGB.2,
    ));
    layer.draw_rect(
        &bg,
        Area {
            x1: 0,
            y1: 0,
            x2: CANVAS_PX - 1,
            y2: CANVAS_PX - 1,
        },
    );

    let mut ink = DrawRectDsc::new();
    ink.bg_color(color_make(INK.0, INK.1, INK.2));
    for row in 0..28i32 {
        let mut col = 0i32;
        while col < 28 {
            if img[(row * 28 + col) as usize] < INK_THRESHOLD {
                col += 1;
                continue;
            }
            let start = col;
            while col < 28 && img[(row * 28 + col) as usize] >= INK_THRESHOLD {
                col += 1;
            }
            layer.draw_rect(
                &ink,
                Area {
                    x1: start * ZOOM,
                    y1: row * ZOOM,
                    x2: col * ZOOM - 1,
                    y2: row * ZOOM + ZOOM - 1,
                },
            );
        }
    }
    drop(layer); // CanvasLayer 的 Drop 会 finish_layer 提交绘制
    canvas.invalidate();
}
