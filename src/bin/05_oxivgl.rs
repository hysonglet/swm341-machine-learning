//! 历程 5：接入 oxivgl（LVGL 9.5 的 safe Rust 绑定）并移植 `flush_cb`。
//!
//! 屏幕已在 `04_lcd` 点亮：LCDC 以 `auto_restart` 连续扫描 `FB_BASE` 处的
//! RGB565 帧缓冲。本历程保留这套硬件初始化，只把「谁往帧缓冲里写像素」从手写的
//! 渐变换成 LVGL：
//!
//! ```text
//! LVGL (PARTIAL 模式)  →  40 行 stripe buffer (SDRAM)
//!                          │
//!                          └─ flush_cb: 按脏矩形逐行 copy 到 FB_BASE
//!                                        │
//! LCDC layer0 (auto_restart) ────────────┴─→ 800x480 RGB 面板
//! ```
//!
//! 这是最经典的 flush_cb 移植：LVGL 只负责画，刷新完成立刻 `flush_ready`，
//! 不需要双缓冲切换、不需要中断、不需要改 LCD 层地址。代价是一次 SDRAM→SDRAM
//! 的行拷贝，对 800x480@30fps 完全够用。
//!
//! ```text
//! cargo run --bin 05_oxivgl
//! ```

#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use cortex_m::asm::delay;
use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::hal;
use swm341_ai::hal::gpio::{Level, Output, OutputConfig};
use swm341_ai::hal::lcdc::{Config, HvTiming, Lcd, LcdPins, LayerConfig, PixelFormat};
use swm341_ai::hal::sdramc::{CasLatency, SdramPins, Sdramc};
use swm341_ai::heap::{self, HEAP_BASE, HEAP_SIZE};

use oxivgl::display::{Buffers, LvglBuffers, COLOR_BUF_LINES};
use oxivgl::mem::reserve_pool_raw;
use oxivgl::style::{Selector, Style};
use oxivgl::view::{NavAction, RenderConfig, Ui, View};
use oxivgl::widgets::{Align, Label, Obj, WidgetError};
use oxivgl_sys::{
    lv_area_t, lv_color_format_t_LV_COLOR_FORMAT_RGB565, lv_display_flush_ready,
    lv_display_get_default, lv_display_set_color_format, lv_display_set_flush_cb, lv_display_t,
};

/// 核心时钟（与 `hrc_150mhz` 一致），用于复位脉冲与帧节奏。
const HCLK_HZ: u32 = 150_000_000;

const WIDTH: usize = 800;
const HEIGHT: usize = 480;

// ── SDRAM (8 MiB @ 0x8000_0000) 分区 ────────────────────────────────────────
// 0x8000_0000 .. 0x8040_0000  Rust 堆（embedded-alloc，4 MiB，供 Style/Box 等）
// 0x8040_0000 .. 0x804B_B800  LCDC 扫描的帧缓冲（800*480*2 = 768 KiB）
// 0x804B_B800 .. 0x804D_AC00  LVGL 的两块 stripe draw buffer（2 * 40 行）
// 0x8050_0000 .. 0x8060_0000  LVGL 自己的 TLSF 堆（1 MiB）
// 其余留作后续图像 / 模型权重。四段互不重叠，均在 8 MiB 内。

/// LCDC 帧缓冲：紧跟在 Rust 堆之后，避免被分配器覆盖。
const FB_BASE: usize = HEAP_BASE + HEAP_SIZE;
const FB_BYTES: usize = WIDTH * HEIGHT * 2;

/// LVGL 的 PARTIAL stripe buffer 放帧缓冲之后。`LvglBuffers` 含两块
/// `LvglBuf<BYTES>`（各 16 字节对齐），BYTES = WIDTH * COLOR_BUF_LINES * 2。
const STRIPE_BYTES: usize = WIDTH * COLOR_BUF_LINES * 2;
const STRIPE_BASE: usize = FB_BASE + FB_BYTES;

/// LVGL 的 C 堆。芯片内部只有 64 KiB SRAM，`lv_conf.h` 里的静态池只留了
/// 16 KiB，其余靠这里在运行时注册进 TLSF（`reserve_pool_raw`，必须在 lv_init 之前）。
/// 取 0x8050_0000 是为了同时满足 4 字节对齐和避开上面的 stripe buffer。
const LV_HEAP_BASE: usize = 0x8050_0000;
const LV_HEAP_SIZE: usize = 1024 * 1024;

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 05_oxivgl: LVGL + flush_cb ====");

    let p = hal::init(hal::Config {
        clock: hal::clock::ClockConfig::hrc_150mhz(),
    });

    // 先偷外设句柄，再移动 GPIO 字段。
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
    rprintln!("SDRAM: {} bytes @ 0x{:08X}", sdram.size_bytes(), sdram.base_address());

    // 2. Rust 堆必须在 SDRAM 可访问之后初始化（HEAP.init 会立刻写空闲链表头）。
    //    oxivgl 的 Style(Rc)、Box::leak 的 View 都要用它。
    unsafe { heap::init_heap() };

    // 3. RGB 并行总线 + 复位 / 背光（与 04_lcd 完全一致，已验证）。
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
        h: HvTiming { sync: 5, back_porch: 46, pixels: 800, front_porch: 64 },
        v: HvTiming { sync: 5, back_porch: 23, pixels: 480, front_porch: 22 },
        bg_color: 0xFFFFFF,
    });

    // 4. 先清一帧黑，别让面板短暂显示 SDRAM 上电后的随机内容。
    fill_solid(FB_BASE as *mut u16, 0x0000);
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
    // `lcd` / `lcd_rst` / `lcd_bl` 在 main 作用域内一直存活（main 不返回），
    // 因此 LCDC 时钟与背光会一直保持，无需额外持有句柄。
    rprintln!("LCDC scanning FB_BASE=0x{:08X}", FB_BASE);

    // 5. 把 SDRAM 的一段交给 LVGL 的 TLSF 堆。必须在 `lv_init()` 之前：
    //    oxivgl 在 driver init 里把 pending pool 注册进去。
    unsafe { reserve_pool_raw(LV_HEAP_BASE as *mut u8, LV_HEAP_SIZE) }
        .expect("register LVGL SDRAM heap pool");
    rprintln!("LVGL heap pool: {} bytes @ 0x{:08X}", LV_HEAP_SIZE, LV_HEAP_BASE);

    // 6. stripe draw buffers 直接落在 SDRAM 固定地址（内部 SRAM 只有 64 KiB，
    //    放不下两块 64 KB 的 buffer）。地址 16 字节对齐，满足 LvglBuf 的
    //    #[repr(align(16))]。LVGL 会覆写其内容，无需预先清零。
    assert_eq!(STRIPE_BASE % 16, 0, "stripe buffers must be 16-byte aligned");
    let stripes: &'static mut LvglBuffers<STRIPE_BYTES> =
        unsafe { &mut *(STRIPE_BASE as *mut LvglBuffers<STRIPE_BYTES>) };

    // 7. Ui::init = lv_init() + tick 源 + 创建 display + 注册 draw buffers。
    //    tick 源是 oxivgl 的 `get_tick_ms()`，它走 embassy-time；HAL 的
    //    `hal::init()` 已经装好 SysTick time driver，版本同为 embassy-time-driver 0.2。
    let ui = Ui::init(WIDTH as i32, HEIGHT as i32, Buffers::partial(stripes));

    // 8. 覆盖 display 的两项接线 —— 这是本次移植的核心。
    //    oxivgl 的 `lvgl_disp_init` 是按 ESP32(SPI 屏) 写的：PARTIAL 模式默认用
    //    RGB565_SWAPPED（SPI 大端），且只在 esp-hal / rtos-sem feature 下才注册
    //    flush_cb。我们既没开那些 feature，LCDC 又是原生小端 RGB565，所以：
    //      a) 颜色格式改回 RGB565；
    //      b) 注册自己的 flush_cb。
    unsafe {
        let disp = lv_display_get_default();
        assert!(!disp.is_null(), "lv_display_create returned NULL");
        lv_display_set_color_format(disp, lv_color_format_t_LV_COLOR_FORMAT_RGB565);
        lv_display_set_flush_cb(disp, Some(swm341_flush_cb));
    }
    rprintln!("flush_cb wired: PARTIAL -> FB_BASE");

    // 9. 建 UI。View 必须地址稳定（LVGL 把裸指针存进 event callback），
    //    Box::leak 到 SDRAM 堆上，生命周期与 display 一致。
    let view: &'static mut DemoView = Box::leak(Box::new(DemoView::default()));
    ui.bind(view, &RenderConfig::default())
        .expect("create LVGL widgets");

    // 10. 阻塞式渲染循环。不用 embassy executor：`timer_handler()` 直接驱动
    //     LVGL 的定时器/重绘，返回值是 LVGL 建议的下次调用间隔 (ms)。
    //     每次循环顺带 poll 一次 View::update，让界面上的计数动起来。
    loop {
        let _ = view.update();
        let ms = ui.timer_handler().clamp(1, 50);
        delay((HCLK_HZ / 1000) * ms);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// flush_cb：LVGL → LCDC 帧缓冲
// ───────────────────────────────────────────────────────────────────────────

/// LVGL 的刷新回调：把一块脏矩形从 PARTIAL stripe buffer 搬到 `FB_BASE`。
///
/// PARTIAL 模式下 `px_map` 是 `w*h` 个**紧凑排列**的 RGB565 像素（行 stride
/// 等于 `w`），而目标帧缓冲的行 stride 是 `WIDTH`，所以必须逐行拷贝，不能
/// 一次 `copy_nonoverlapping(w*h)`。
///
/// 本函数是同步的：拷贝完立刻 `lv_display_flush_ready(disp)`。因为 LCDC 在
/// `auto_restart` 下持续扫描同一块内存，拷贝本身就已经「上屏」，没有 DMA 完成
/// 中断要等。若以后改成双缓冲 + vblank 换地址，才需要把 ready 延后到中断里。
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
                // 帧缓冲由 LCDC 硬件异步读取，用 volatile 写避免编译器把整段
                // 拷贝当成「无人读取」而优化掉；源是普通内存，直接读即可。
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

/// 整屏填纯色（仅在 LVGL 接管之前清一帧，避免上电花屏）。
fn fill_solid(fb: *mut u16, color: u16) {
    for i in 0..WIDTH * HEIGHT {
        unsafe { fb.add(i).write_volatile(color) };
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 一个最小 GUI：深蓝底 + 居中的标题与一个会跳的计数
// ───────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct DemoView {
    title: Option<Label<'static>>,
    counter: Option<Label<'static>>,
    ticks: u32,
}

impl View for DemoView {
    fn create(&mut self, container: &Obj<'static>) -> Result<(), WidgetError> {
        let style = Style::new(|s| {
            s.bg_color_hex(0xff0000).bg_opa(255).text_color_hex(0xFFFFFF);
        });
        container.add_style(&style, Selector::DEFAULT);

        let title = Label::new(container)?;
        title.text("oxivgl on SWM341").align(Align::Center, 0, -40);

        let counter = Label::new(container)?;
        counter.text("flush_cb OK  tick 0").align(Align::Center, 0, 40);

        self.title = Some(title);
        self.counter = Some(counter);
        Ok(())
    }

    fn update(&mut self) -> Result<NavAction, WidgetError> {
        self.ticks = self.ticks.wrapping_add(1);
        // 每 16 个 tick 改一次文本：既证明渲染循环在跑，又不至于每帧全量重绘。
        if self.ticks % 16 == 0 {
            if let Some(label) = &self.counter {
                label.text(&format!("flush_cb OK  tick {}", self.ticks));
            }
        }
        Ok(NavAction::None)
    }
}
