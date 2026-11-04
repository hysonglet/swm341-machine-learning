//! 历程 4：TFT-LCD 点亮（裸帧缓冲，不依赖 LVGL）。
//!
//! 目标只有一句话：**让屏幕先亮起来**，确认 RGB 并行总线、SDRAM 帧缓冲、背光 /
//! 复位时序都正确。这里只往 SDRAM 帧缓冲里填色，由 LCD 控制器硬件自动扫描输出，
//! 不牵扯任何 GUI 库。
//!
//! 验证步骤：
//!
//! ```text
//! cargo run --bin 04_lcd
//! ```
//!
//! 点亮后，下一步再接入 `lvgl` 的 Rust 绑定，并最终换成 `oxivgl` 做 GUI。届时
//! LVGL 的 `flush_cb` 只要直接写这块帧缓冲（`FB_BASE`）即可，本文件的 LCD 初始化
//! 与图层配置可以原样复用（见文末「LVGL / oxivgl 接入点」）。
//!
//! 引脚与时序沿用 vendor `SimplLCD_RGB` 参考（24 位数据总线，PB2/3/4/5 =
//! VSYNC/HSYNC/DEN/DCLK）。复位 / 背光是板级 GPIO：这里 `PD1` 复位、`PD9` 背光，
//! 对应 SWM34SVE 100 脚 demo 板；64 脚板请把背光改成 `PB13`。

#![no_std]
#![no_main]

use cortex_m::asm::delay;
use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init_print};

use swm341_ai::hal;
use swm341_ai::hal::gpio::{Level, Output, OutputConfig};
use swm341_ai::hal::lcdc::{Config, HvTiming, Lcd, LcdPins, LayerConfig, PixelFormat};
use swm341_ai::hal::sdramc::{CasLatency, SdramPins, Sdramc};
use swm341_ai::heap::{HEAP_BASE, HEAP_SIZE};

/// 核心时钟（与 `hrc_150mhz` 一致），用于 1 ms 复位脉冲与帧节奏。
const HCLK_HZ: u32 = 150_000_000;

/// 面板几何（800x480 RGB565，vendor 参考时序）。
const WIDTH: usize = 800;
const HEIGHT: usize = 480;

/// 帧缓冲放在堆**之后**（`HEAP_BASE + HEAP_SIZE`），避免覆盖 `embedded-alloc`
/// 管理的 4 MiB 堆区。800*480*2 = 768 KiB，0x8040_0000 + 0xBB800 < 0x8080_0000
/// （8 MiB SDRAM 上界），安全。
const FB_BASE: usize = HEAP_BASE + HEAP_SIZE;

#[entry]
fn main() -> ! {
    rtt_init_print!();
    rprintln!("\n==== 04_lcd: TFT-LCD 点亮 ====");

    let p = hal::init(hal::Config {
        clock: hal::clock::ClockConfig::hrc_150mhz(),
    });
    rprintln!("HAL initialised");

    // 先偷出 lcd / sdramc 外设句柄，再移动 GPIO 字段（移动 `p.Pxy` 会借用 `p`）。
    let lcd_peri = p.lcd();
    let sdram_peri = p.sdramc();

    // 1. SDRAM：帧缓冲落在这一片。
    let sdram = Sdramc::new_with_pins(
        sdram_peri,
        SdramPins::default(),
        hal::sdramc::Config::default()
            .cas_latency(CasLatency::Latency3)
            .trc(8),
    );
    rprintln!(
        "SDRAM up: {:>9} bytes @ 0x{:08X}",
        sdram.size_bytes(),
        sdram.base_address()
    );

    // 2. 路由 28 线并行 RGB 总线，接管 LCD 控制器。
    let lcd = Lcd::new_with_pins(lcd_peri, LcdPins::default());

    // 3. 复位脉冲：拉低 ~1 ms 再释放。
    let mut lcd_rst = Output::new(p.PD1, OutputConfig::default().initial_level(Level::Low));
    delay(HCLK_HZ / 1000);
    lcd_rst.set_high();

    // 4. 背光打开（100 脚板 PD9；64 脚板 PB13）。
    let mut lcd_bl = Output::new(p.PD9, OutputConfig::default().initial_level(Level::High));
    let _ = &mut lcd_bl; // 保持背光常亮
    rprintln!("LCD reset released, backlight on");

    // 5. 800x480 RGB565 时序。DOTCLK = HCLK / (clock_div + 2) = 150 MHz / 5 = 30 MHz
    //    （clock_div = 3）。`invert_clock` 对应 vendor `SampleEdge = LCD_SAMPLE_FALL`。
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
    rprintln!("LCD configured: 800x480 RGB565, auto-restart on");

    let fb = FB_BASE as *mut u16;

    // 6. 先填一帧纯色，确保面板一上来显示的是确定的颜色（而不是 SDRAM 里的随机
    //    垃圾）。改 0xF800(红)/0x07E0(绿)/0x001F(蓝)/0xFFFF(白) 可快速排查总线。
    fill_solid(fb, 0x001F); // 纯蓝

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
    rprintln!("LCD scanning out -- solid blue");

    // 7. 动画：对角流动的彩虹渐变，证明帧缓冲可被持续改写并正确刷出。
    let mut frame: u32 = 0;
    loop {
        draw_gradient(fb, frame);
        frame = frame.wrapping_add(1);
        delay(HCLK_HZ / 120); // 约 60 fps 节奏（逐像素重绘本身也要几 ms）
        if frame % 60 == 0 {
            rprintln!("frame {}", frame);
        }
    }
}

/// 整屏填纯色（RGB565）。
#[inline(always)]
fn fill_solid(fb: *mut u16, color: u16) {
    let pixels = WIDTH * HEIGHT;
    for i in 0..pixels {
        unsafe { fb.add(i).write_volatile(color) };
    }
}

/// 对角流动 RGB565 彩虹渐变。三路相位错开三角波，平滑且不需要浮点 / 三角函数表。
#[inline(always)]
fn draw_gradient(fb: *mut u16, t: u32) {
    for y in 0..HEIGHT as u32 {
        for x in 0..WIDTH as u32 {
            let s = (y.wrapping_mul(256) / HEIGHT as u32)
                .wrapping_add(x.wrapping_mul(256) / WIDTH as u32)
                .wrapping_add(t)
                & 0xFF;
            let r = tri(s);
            let g = tri(s.wrapping_add(0x55));
            let b = tri(s.wrapping_add(0xAA));
            let rgb = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            unsafe {
                fb.add((y as usize) * WIDTH + (x as usize))
                    .write_volatile(rgb as u16);
            }
        }
    }
}

/// 8 位三角波 0..255..0（周期 256），调用方传入 `s.wrapping_add(offset)` 可能越界，
/// 先 `& 0xFF` 再算，避免 debug 下下溢 panic 把 CPU 挂死在涂屏之前。
#[inline(always)]
fn tri(v: u32) -> u32 {
    let v = v & 0xFF;
    if v < 128 {
        v * 2
    } else {
        (255 - v) * 2
    }
}

// ───────────────────────────────────────────────────────────────────────────
// LVGL / oxivgl 接入点（点亮之后再做）
// ───────────────────────────────────────────────────────────────────────────
// 1) 依赖：Cargo.toml 加 `lvgl`（或最终的 `oxivgl`）。两者都需要一个 `Display`
//    后端：实现其 `flush` 回调，把脏矩形直接 `copy_from_slice` 到 `FB_BASE` 处的
//    RGB565 帧缓冲即可 —— 本文件的 LCD 控制器初始化（`configure` / `set_layer` /
//    `start_frame`）原样保留，硬件扫描不变。
//
// 2) 帧缓冲即 `FB_BASE`（800x480 RGB565，stride = 800*2）。LVGL 的
//    `lv_disp_draw_buf` 可指向同一块内存，或另开双缓冲再 `set_layer` 切换 `ADDR`。
//
// 3) 时序 / 引脚若换板，只改上面的 `Config` 与 `PD1`/`PD9` 即可，GUI 代码不受影响。
