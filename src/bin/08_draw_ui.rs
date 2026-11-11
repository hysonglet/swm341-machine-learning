//! 历程 8：在屏幕上**手绘**一个数字，实时识别。
//!
//! 与 `07_sd_ui` 的区别：数据来源从「SD 卡上的真实 PNG」换成「用手指在画板上现画」。
//! 左侧 260×260 的 canvas 就是一个手写板——GT911 触摸控制器（I2C1，参考
//! `swm341-hal/examples/i2c_block.rs`）每次轮询把触点坐标报上来，我们把笔迹以
//! 112×112 灰度（背景 0 / 笔画 255）记到 `ink` 缓冲里，再：
//!
//! 1. **显示**：把 `ink` 等比放大画到 canvas（复用 07 的 `draw_gray`）；
//! 2. **识别**：把 `ink` 交给 `core_ml::preprocess`（裁剪包围盒 + 质心居中 + 缩放成
//!    28×28，与训练流水线完全一致），再 `model.infer` + 真 f32 softmax，实时刷新右侧
//!    的预测 / 置信度 / 10 类概率条。
//!
//! 画板下方有一个「CLEAR / 清空」按钮（屏幕坐标已硬编码），手指点一下即清屏重画。
//!
//! ```text
//! cargo run --bin 08_draw_ui
//! ```
//!
//! 触摸坐标映射：GT911 报的是它自己的分辨率（默认当 800×480 处理），线性映射到 LCD
//! 像素。若发现上下 / 左右镜像，把文件顶部 `TP_SWAP` / `TP_FLIP_X` / `TP_FLIP_Y`
//! 改成 `true` 即可；若整体偏移，按 RTT 首帧打印的 raw 坐标校准 `TP_X_MAX/TP_Y_MAX`。

#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;

use cortex_m::asm::delay;
use cortex_m_rt::entry;
use libm;
use rtt_target::{rprintln, rtt_init, set_print_channel, ChannelMode};

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

use swm341_ai::core_ml::preprocess::preprocess;
use swm341_ai::core_ml::{Model, N_CLASSES};
use swm341_ai::hal;
use swm341_ai::hal::gpio::{Level, Output, OutputConfig};
use swm341_ai::hal::lcdc::{Config, HvTiming, LayerConfig, Lcd, LcdPins, PixelFormat};
use swm341_ai::hal::sdramc::{CasLatency, SdramPins, Sdramc};
use swm341_ai::heap::{self, HEAP_BASE, HEAP_SIZE};

use oxivgl::display::{Buffers, LvglBuffers, COLOR_BUF_LINES};
use oxivgl::draw_buf::{ColorFormat, DrawBuf};
use oxivgl::fonts::{MONTSERRAT_14, MONTSERRAT_20, MONTSERRAT_36};
use oxivgl::indev::PointerIndev;
use oxivgl::mem::reserve_pool_raw;
use oxivgl::style::{color_make, Selector, Style};
use oxivgl::view::{NavAction, RenderConfig, Ui, View};
use oxivgl::widgets::{AsLvHandle, Bar, Canvas, Child, Label, Obj, Part, WidgetError};
use oxivgl_sys::{
    lv_area_t, lv_canvas_get_draw_buf, lv_color_format_t_LV_COLOR_FORMAT_RGB565,
    lv_display_flush_ready, lv_display_get_default, lv_display_set_color_format,
    lv_display_set_flush_cb, lv_display_t, lv_draw_buf_flush_cache, lv_draw_buf_goto_xy,
};

// 触摸控制器（GT911）走 I2C，方法与 `i2c_block.rs` 完全一致：
// I2C1，SCL=PC5，SDA=PC4，从地址 0x5D，寄存器地址大端。

const HCLK_HZ: u32 = 150_000_000;
const WIDTH: usize = 800;
const HEIGHT: usize = 480;

// ── SDRAM (8 MiB @ 0x8000_0000) 分区，与 05_oxivgl / 06_digit_ui / 07 一致 ──────
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

/// 画布（屏幕显示）边长，像素。
const CANVAS_PX: i32 = 260;
/// 手写缓冲（识别源）边长，像素。112 足够画顺滑笔迹，又不会让 preprocess 太重。
const INK_W: usize = 112;
const INK_H: usize = 112;
/// 画笔半径（以 `ink` 像素计）。≈ 4 * 260/112 ≈ 9px，粗细合适。
const PEN_R: i32 = 4;

/// 画板在屏幕上的左上角（card_l 在 (16,56)，canvas 在卡片内 (20,44)）。
const CANVAS_SCREEN_X: i32 = 16 + 20; // 36
const CANVAS_SCREEN_Y: i32 = 56 + 44; // 100

/// 「CLEAR / 清空」按钮的屏幕矩形（card_l 内 (16,316)，尺寸 288×34）。
const CLEAR_X: i32 = 16 + 16; // 32
const CLEAR_Y: i32 = 56 + 316; // 372
const CLEAR_W: i32 = 288;
const CLEAR_H: i32 = 48;

/// 识别节流：最短间隔（ms）。手指滑动时太频繁跑 preprocess 没必要。
const REC_INTERVAL_MS: u32 = 30;

/// 由 host/ 工具生成（先跑 `cd host && cargo run --release`）。
const MODEL_BIN: &[u8] = include_bytes!("../../gen/model.bin");

// ── GT911 触摸控制器（I2C）相关常量 ───────────────────────────────────────────
const TP_ADDR: u8 = 0x5D;
const TP_ALT_ADDR: u8 = 0x14;
const TP_FREQ: u32 = 400_000;
const REG_STATUS: u16 = 0x814E;
const REG_TP1: u16 = 0x8150;
const REG_PID: u16 = 0x8140;
/// 距最后一帧 READY 超过这么多次 indev 轮询，就强制抬笔（≈200ms 兜底释放，移植自
/// board_touch.c 的 TP_RELEASE_TIMEOUT_MS）。GT911 松手后有些固件不再产生新帧。
const TP_RELEASE_POLLS: u32 = 8;
/// GT911 报上来的分辨率（线性映射到 LCD）。若实测 raw 上限不同，照此改。
const TP_X_MAX: u32 = 800;
const TP_Y_MAX: u32 = 480;
/// 坐标变换开关：照搬 board_touch.c 的实测配置 —— TOUCH_SWAP_XY=0 / TOUCH_MIRROR_X=0 /
/// TOUCH_MIRROR_Y=0，即 GT911 原始坐标直接当屏幕坐标用（注释明确：该面板 Y 方向与
/// LCD 扫描一致，镜像反而会让触摸上下颠倒）。所以这里三个开关全部关闭。
const TP_SWAP: bool = false;
const TP_FLIP_X: bool = false;
const TP_FLIP_Y: bool = false;

type TouchBus<'d> = hal::i2c::I2c<'d, hal::I2c1, hal::gpio::PC5, hal::gpio::PC4>;

#[entry]
fn main() -> ! {
    // RTT 用非阻塞通道（缓冲区满就丢弃），避免日志洪流时卡住主循环。
    let channels = rtt_init! {
        up: {
            0: {
                size: 1024,
                mode: ChannelMode::NoBlockSkip,
            }
        }
    };
    set_print_channel(channels.up.0);
    rprintln!("\n==== 08_draw_ui: 手绘数字 + 实时识别 ====");

    // 启用 DWT 周期计数器，供识别耗时计时。
    enable_dwt();

    let p = hal::init(hal::Config {
        clock: hal::clock::ClockConfig::hrc_150mhz(),
    });

    let lcd_peri = p.lcd();
    let sdram_peri = p.sdramc();
    let i2c1 = p.i2c1();
    let pc5 = p.PC5;
    let pc4 = p.PC4;
    let pb11 = p.PB11; // GT911 RST

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

    // 3. LCDC 初始化（与 04/05/06/07 完全一致，已真机验证）。
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

    // 5. 覆盖 display 的颜色格式 + 注册 flush_cb。
    unsafe {
        let disp = lv_display_get_default();
        assert!(!disp.is_null(), "lv_display_create returned NULL");
        lv_display_set_color_format(disp, lv_color_format_t_LV_COLOR_FORMAT_RGB565);
        lv_display_set_flush_cb(disp, Some(swm341_flush_cb));
    }

    // 6. 触摸屏（GT911）：按已验证可用的 board_touch.c 时序复位 ——
    //    RST=PB11 拉低、INT=PD0 拉低（锁定从地址 0x5D），释放 RST 后把 INT 改回输入+下拉。
    //    RST=PB11 拉低、INT=PD0 拉低（锁从地址 0x5D）。引脚非 Copy，PD0 只能取用一次，
    //    故把它放在块里：复位期间驱动为低（定地址），块结束 drop 释放为浮空(high-Z)，
    //    之后由 GT911 的 open-drain INT 自由驱动（等价于 board_touch.c 的“输入+下拉”）。
    //    面板自带出厂配置，这里不上传配置、也不软复位（那两步在本项目里反而让芯片不报点）。
    let mut tp_rst = Output::new(pb11, OutputConfig::default().initial_level(Level::Low));
    {
        let _tp_int = Output::new(p.PD0, OutputConfig::default().initial_level(Level::Low));
        delay(HCLK_HZ / 100); // ~10ms：拉低保持复位
        tp_rst.set_high(); // 释放复位，GT911 开始初始化
        delay(HCLK_HZ / 100); // ~10ms
                              // _tp_int 在此块结束被 drop，释放 PD0 为 high-Z
    }
    delay(HCLK_HZ / 20); // ~50ms：等固件就绪

    let mut tp = hal::i2c::I2c::new(
        i2c1,
        pc5,
        pc4,
        hal::i2c::Config::default().frequency(TP_FREQ),
    );
    // 再留一点余量，确保首笔 I2C 不会撞在 GT911 启动尾巴上。
    delay(HCLK_HZ / 100);

    // 依次探测 0x5D / 0x14，记录命中地址。
    let mut tp_addr: u8 = TP_ADDR;
    let probed = if tp.blocking_write(TP_ADDR, &[]).is_ok() {
        tp_addr = TP_ADDR;
        Some(TP_ADDR)
    } else if tp.blocking_write(TP_ALT_ADDR, &[]).is_ok() {
        tp_addr = TP_ALT_ADDR;
        Some(TP_ALT_ADDR)
    } else {
        None
    };
    match probed {
        Some(a) => {
            // 顺手读一下产品 ID（"911"），确认不是别的从设备抢答。
            let mut pid = [0u8; 3];
            if tp
                .blocking_write_read(a, &reg_addr(REG_PID), &mut pid)
                .is_ok()
            {
                rprintln!(
                    "GT911 found @ 0x{:02X}, PID={}{}{}",
                    a,
                    pid[0] as char,
                    pid[1] as char,
                    pid[2] as char
                );
            } else {
                rprintln!("GT911 @ 0x{:02X} 探测到但读 PID 失败", a);
            }
        }
        None => rprintln!("✗ GT911 无响应（0x5D / 0x14 均无 ACK，检查 I2C 接线）"),
    }

    // 7. 模型（在 flash，不占堆）。
    let model: &'static Model = Box::leak(Box::new(
        Model::from_bytes(MODEL_BIN).expect("invalid gen/model.bin"),
    ));
    let s_out = f32::from_le_bytes(MODEL_BIN[16..20].try_into().expect("model.bin too short"));
    rprintln!("model loaded, s_out={}", s_out);

    // 8. 手写缓冲：背景 0 / 笔画 255。生命周期与 display 一致（'static 泄漏）。
    let ink: &'static mut [u8; INK_W * INK_H] = Box::leak(Box::new([0u8; INK_W * INK_H]));

    // 9. 页面。leak 到 SDRAM 堆，生命周期与 display 一致。
    let view: &'static mut DrawView = Box::leak(Box::new(DrawView::new()));
    ui.bind(view, &RenderConfig::default())
        .expect("create LVGL widgets");
    view.paint_canvas(ink);

    // 10. 把 GT911 注册成 LVGL 的 POINTER 输入设备：LVGL 在自己的读定时器上回调
    //     `read`，我们在这里轮询 GT911、把触点映射到画板/ink，并把屏幕坐标回给 LVGL。
    //     `tp`/`ink` 都是 'static 引用，通过裸指针在闭包与主循环间共享（单线程）。
    // SAFETY: `p` 在 main 中永不被 drop，把 I2C 句柄提升为 'static 引用是安全的。
    let tp: &'static mut TouchBus<'static> =
        unsafe { core::mem::transmute(Box::leak(Box::new(tp))) };
    let ts_ptr: *mut TouchState = Box::leak(Box::new(TouchState {
        tp,
        tp_addr,
        ink,
        last: None,
        mode: PenMode::Up,
        dirty: false,
        force: false,
        dbg_status: 0,
        coord_err: false,
        logged: false,
        down: false,
        last_tx: 0,
        last_ty: 0,
        since_frame: 0,
    })) as *mut TouchState;

    let _indev = PointerIndev::new_with(move || -> Option<(u16, u16)> {
        let st = unsafe { &mut *ts_ptr };
        // 读状态寄存器（不清 READY），记录原始值供心跳打印。这里**不再**打状态跳变日志：
        // 触摸时 0x80/0x00/0x81 之间每秒要跳变几十次，每次 rprintln 会拖慢甚至卡住
        // 主循环（RTT 已改为非阻塞，但省掉无谓打印更稳）。
        let raw = peek_status(&mut *st.tp, st.tp_addr).unwrap_or(0xFF);
        if raw == 0xFF {
            // 状态读取失败（总线异常）：原地复位 I2C 控制器，下一帧即可恢复，
            // 避免总线卡死把整个主循环拖垮（这是“写多了卡死”的根因之一）。
            (&mut *st.tp).reset();
        }
        st.dbg_status = raw;

        // —— 移植自 board_touch.c：以 BIT7(READY) 作为“有数据帧”判据 ——
        // GT911 空闲态也可能 bit7=1(且点数=0)，只看低 4 位点数会误判；正确做法是
        // bit7 置位才处理这一帧，否则保留上一帧坐标/状态。
        if raw != 0xFF && (raw & 0x80) != 0 {
            // 必须清 READY（写 0），芯片才会给下一帧。顺序与 C 驱动一致：先清后取坐标。
            let a = reg_addr(REG_STATUS);
            if (&mut *st.tp)
                .blocking_write(st.tp_addr, &[a[0], a[1], 0])
                .is_err()
            {
                (&mut *st.tp).reset();
            }
            st.since_frame = 0;

            let num = raw & 0x0F;
            if num == 0 || num >= 6 {
                // 松手那一帧：清掉按下态，但保留 ink（抬笔后再识别一次）。
                if st.down {
                    rprintln!("tp up");
                }
                st.down = false;
                st.mode = PenMode::Up;
                st.last = None;
                return None;
            }

            // 有触点：读第一个点坐标（0x8150 起 4 字节：xL xH yL yH）。
            let mut xy = [0u8; 4];
            if (&mut *st.tp)
                .blocking_write_read(st.tp_addr, &reg_addr(REG_TP1), &mut xy)
                .is_err()
            {
                if !st.coord_err {
                    rprintln!("GT911: status=0x{:02X} 但读坐标失败（I2C NAK?）", raw);
                    st.coord_err = true;
                }
                (&mut *st.tp).reset();
                if st.down {
                    rprintln!("tp up");
                }
                st.down = false;
                st.mode = PenMode::Up;
                st.last = None;
                return None;
            }
            // GT911 坐标为 12 位；高字节低 4 位为高位（高 4 位是 track id，忽略）。
            let mut mx = (((xy[1] & 0x0F) as u32) << 8) | xy[0] as u32;
            let mut my = (((xy[3] & 0x0F) as u32) << 8) | xy[2] as u32;

            // raw → LCD 像素，含可选的 swap/flip 校正。
            if TP_SWAP {
                core::mem::swap(&mut mx, &mut my);
            }
            if TP_FLIP_X {
                mx = TP_X_MAX - mx;
            }
            if TP_FLIP_Y {
                my = TP_Y_MAX - my;
            }
            let sx = (mx as i32).clamp(0, TP_X_MAX as i32 - 1) * WIDTH as i32 / TP_X_MAX as i32;
            let sy = (my as i32).clamp(0, TP_Y_MAX as i32 - 1) * HEIGHT as i32 / TP_Y_MAX as i32;

            if !st.logged {
                let region = if in_rect(sx, sy, CLEAR_X, CLEAR_Y, CLEAR_W, CLEAR_H) {
                    "CLEAR"
                } else if in_rect(sx, sy, CANVAS_SCREEN_X, CANVAS_SCREEN_Y, CANVAS_PX, CANVAS_PX) {
                    "canvas"
                } else {
                    "outside"
                };
                rprintln!(
                    "first touch: raw=({}, {})  mapped=({}, {})  region={}  [镜像改 TP_SWAP/TP_FLIP_X/TP_FLIP_Y]",
                    (((xy[1] & 0x0F) as u32) << 8) | xy[0] as u32,
                    (((xy[3] & 0x0F) as u32) << 8) | xy[2] as u32,
                    sx, sy, region
                );
                st.logged = true;
            }

            st.last_tx = sx;
            st.last_ty = sy;
            if !st.down {
                rprintln!("tp down ({}, {})", sx, sy);
            }
            st.down = true;

            // 落笔到 ink：先判 CLEAR 按钮，再判画板。
            if in_rect(sx, sy, CLEAR_X, CLEAR_Y, CLEAR_W, CLEAR_H) {
                for v in st.ink.iter_mut() {
                    *v = 0;
                }
                st.mode = PenMode::Clearing;
                st.dirty = true;
                st.force = true;
                rprintln!("CLEAR");
            } else if in_rect(sx, sy, CANVAS_SCREEN_X, CANVAS_SCREEN_Y, CANVAS_PX, CANVAS_PX) {
                let (ix, iy) = to_ink(sx, sy);
                match st.mode {
                    PenMode::Up => {
                        ink_stamp(st.ink, ix, iy, PEN_R);
                        st.mode = PenMode::Drawing;
                        st.last = Some((ix, iy));
                    }
                    PenMode::Drawing => {
                        if let Some((px, py)) = st.last {
                            ink_line(st.ink, px, py, ix, iy, PEN_R);
                        } else {
                            ink_stamp(st.ink, ix, iy, PEN_R);
                        }
                        st.last = Some((ix, iy));
                    }
                    PenMode::Clearing => {}
                }
                st.dirty = true;
            }
            return Some((sx as u16, sy as u16));
        }

        // —— 没有新数据帧：保留上一帧坐标，靠超时兜底释放 ——
        st.since_frame = st.since_frame.saturating_add(1);
        if st.down && st.since_frame > TP_RELEASE_POLLS {
            st.down = false;
            // 抬笔瞬间再跑一次识别，保证结果正确。
            if let PenMode::Drawing = st.mode {
                st.dirty = true;
                st.force = true;
            }
            st.mode = PenMode::Up;
            st.last = None;
        }
        if st.down {
            Some((st.last_tx as u16, st.last_ty as u16))
        } else {
            None
        }
    })
    .expect("PointerIndev init failed");

    rprintln!("UI 已 bind，进入主循环（在左侧画板写字）");

    // 11. 主循环：LVGL 在 timer_handler 内回调上面的 indev 读触摸；这里只负责
    //     把 ink 重绘到 canvas 并按需跑识别。
    let mut ticks: u32 = 0;
    let mut last_rec: u32 = 0;
    let mut hb = 0u32;
    loop {
        let ms = ui.timer_handler().clamp(1, 50);
        hb = hb.wrapping_add(1);
        if hb % 60 == 0 {
            let st = unsafe { &mut *ts_ptr };
            rprintln!("HB{} ms={} status=0x{:02X}", hb, ms, st.dbg_status);
        }
        ticks = ticks.saturating_add(ms);

        let st = unsafe { &mut *ts_ptr };
        if st.dirty {
            view.paint_canvas(st.ink);
            st.dirty = false;
            if st.force || ticks.saturating_sub(last_rec) >= REC_INTERVAL_MS {
                recognize_and_show(view, model, s_out, st.ink);
                last_rec = ticks;
                st.force = false;
            }
        }

        let _ = view.update();
        delay((HCLK_HZ / 1000) * ms);
    }
}

/// 屏幕坐标是否在矩形内。
fn in_rect(x: i32, y: i32, rx: i32, ry: i32, rw: i32, rh: i32) -> bool {
    x >= rx && x < rx + rw && y >= ry && y < ry + rh
}

/// 屏幕（画板内）坐标 → ink 像素坐标，并钳制在合法范围。
fn to_ink(sx: i32, sy: i32) -> (i32, i32) {
    let cx = (sx - CANVAS_SCREEN_X).clamp(0, CANVAS_PX - 1);
    let cy = (sy - CANVAS_SCREEN_Y).clamp(0, CANVAS_PX - 1);
    let ix = cx * INK_W as i32 / CANVAS_PX as i32;
    let iy = cy * INK_H as i32 / CANVAS_PX as i32;
    (ix.clamp(0, INK_W as i32 - 1), iy.clamp(0, INK_H as i32 - 1))
}

/// 在 `ink` 里以 (cx,cy) 为圆心、半径 r 盖一个实心圆（笔画 255）。
fn ink_stamp(ink: &mut [u8], cx: i32, cy: i32, r: i32) {
    let x0 = (cx - r).max(0);
    let x1 = (cx + r).min(INK_W as i32 - 1);
    let y0 = (cy - r).max(0);
    let y1 = (cy + r).min(INK_H as i32 - 1);
    let r2 = r * r;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r2 {
                ink[(y * INK_W as i32 + x) as usize] = 255;
            }
        }
    }
}

/// 在 `ink` 里画一条带笔宽的线段（用整数插值，避免引入浮点）。
fn ink_line(ink: &mut [u8], x0: i32, y0: i32, x1: i32, y1: i32, r: i32) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).max(1);
    for i in 0..=steps {
        let x = x0 + (dx * i) / steps;
        let y = y0 + (dy * i) / steps;
        ink_stamp(ink, x, y, r);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// GT911 触摸读取（I2C，参考 swm341-hal/examples/i2c_block.rs）
// ───────────────────────────────────────────────────────────────────────────

/// 16 位寄存器地址（大端）。
fn reg_addr(reg: u16) -> [u8; 2] {
    [(reg >> 8) as u8, reg as u8]
}

/// 仅读取状态寄存器（不清 READY 位），用于调试：看 GT911 是否在上报触摸。
/// 返回值 bit7=有数据、低 4 位=触点数。
fn peek_status(tp: &mut TouchBus, addr: u8) -> Option<u8> {
    let mut status = [0u8; 1];
    if tp
        .blocking_write_read(addr, &reg_addr(REG_STATUS), &mut status)
        .is_err()
    {
        return None;
    }
    Some(status[0])
}

// ───────────────────────────────────────────────────────────────────────────
// flush_cb / 帧缓冲（与 05_oxivgl / 06_digit_ui / 07 相同）
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
// 识别 + 显示
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

/// 把 `ink`（112×112）走 `preprocess`（裁剪+居中+28×28）后推理，结果刷到页面。
/// `ink` 全空时 `preprocess` 返回 None，显示占位 “-”。
fn recognize_and_show(view: &mut DrawView, model: &Model, s_out: f32, ink: &[u8; INK_W * INK_H]) {
    let t0 = cyc();
    match preprocess(ink, INK_W as u32, INK_H as u32, false) {
        None => {
            view.show_empty_pred();
            rprintln!("recognize: 空白画板");
        }
        Some(img28) => {
            let logits = model.infer(&img28);
            let probs = probabilities(&logits, s_out);
            // 识别耗时（不含触摸轮询）= 经过的 CPU 周期 / 主频(MHz) → 微秒。
            let rec_us = cyc().wrapping_sub(t0) / (HCLK_HZ / 1_000_000);
            if let Some(l) = &view.rec_l {
                let ms_x10 = rec_us / 100; // 0.1 ms 为单位，避免浮点格式化
                l.text(&format!("{}.{} ms", ms_x10 / 10, ms_x10 % 10));
            }
            view.show_pred(&probs);

            let mut pred = 0usize;
            for k in 1..N_CLASSES {
                if probs[k] > probs[pred] {
                    pred = k;
                }
            }
            let p10 = (probs[pred] * 1000.0) as i32;
            rprintln!("  ▶ 预测={} 置信度 {}.{}%", pred, p10 / 10, p10 % 10);
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 页面
// ───────────────────────────────────────────────────────────────────────────

/// 触摸笔状态。
enum PenMode {
    Up,
    Drawing,
    Clearing,
}

/// 触摸轮询（由 LVGL 的 pointer indev 在它的读定时器上调用）与渲染循环之间共享的状态。
/// 以 `Box::leak` 提升为 `'static`，再通过裸指针在闭包与主循环间访问（单线程、非并发，
/// 故安全）。`tp`/`ink` 也是 `'static` 引用。
struct TouchState {
    tp: &'static mut TouchBus<'static>,
    tp_addr: u8,
    ink: &'static mut [u8; INK_W * INK_H],
    last: Option<(i32, i32)>,
    mode: PenMode,
    dirty: bool,
    /// 抬起/清空瞬间置位，强制跑一次识别（不受 30ms 节流限制）。
    force: bool,
    /// 最近一次状态寄存器原始值，供心跳打印诊断。
    dbg_status: u8,
    /// 坐标读取失败时只告警一次。
    coord_err: bool,
    /// 首触日志只打一次。
    logged: bool,
    /// —— 以下三个字段移植自 board_touch.c：用 BIT7(READY) 作为“有数据帧”判据，
    ///     缓存上一帧坐标，靠轮询超时兜底释放，避免 LVGL 卡在按下态 / 丢抬笔。 ——
    /// 稳定的按下状态（每收到一帧 READY 置 1，超时清零）。
    down: bool,
    /// 最近一次有效坐标（LCD 像素空间，0..800 / 0..480）。
    last_tx: i32,
    last_ty: i32,
    /// 距上一帧 READY 已经过了多少次 indev 轮询；超过阈值即视为抬起。
    since_frame: u32,
}

struct DrawView {
    canvas: Option<Canvas<'static>>,
    pred: Option<Label<'static>>,
    conf: Option<Label<'static>>,
    rec_l: Option<Label<'static>>,
    status_l: Option<Label<'static>>,
    digit_labels: [Option<Label<'static>>; N_CLASSES],
    pct_labels: [Option<Label<'static>>; N_CLASSES],
    bars: [Option<Bar<'static>>; N_CLASSES],
}

impl DrawView {
    fn new() -> Self {
        Self {
            canvas: None,
            pred: None,
            conf: None,
            rec_l: None,
            status_l: None,
            digit_labels: core::array::from_fn(|_| None),
            pct_labels: core::array::from_fn(|_| None),
            bars: core::array::from_fn(|_| None),
        }
    }

    /// 把 `ink` 灰度图等比放大画到 canvas（复用 07 的 draw_gray）。
    fn paint_canvas(&self, ink: &[u8]) {
        if let Some(c) = &self.canvas {
            draw_gray(c, ink, INK_W as u32, INK_H as u32);
        }
    }

    /// 实时把 10 类概率刷到页面（预测大数字 + 置信度 + 概率条）。
    fn show_pred(&mut self, probs: &[f32; N_CLASSES]) {
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
    }

    /// 空白 / 无笔迹时的占位显示。
    fn show_empty_pred(&mut self) {
        if let Some(l) = &self.pred {
            l.text("-");
        }
        if let Some(l) = &self.conf {
            l.text("-");
        }
        for k in 0..N_CLASSES {
            if let Some(l) = &self.digit_labels[k] {
                l.text(&format!(" {}", k));
            }
            if let Some(l) = &self.pct_labels[k] {
                l.text("0%");
            }
            if let Some(b) = &self.bars[k] {
                b.set_value_raw(0, false);
            }
        }
    }
}

impl View for DrawView {
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
        // CLEAR 按钮样式。
        let s_clear = Style::new(|s| {
            s.bg_color_hex(0x2A3550)
                .bg_opa(255)
                .radius(6)
                .border_width(1)
                .border_opa(255)
                .border_color_hex(ACCENT);
        });
        let s_clear_txt = Style::new(|s| {
            s.text_color_hex(ACCENT).text_font(MONTSERRAT_14).bg_opa(0);
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
        static_label(root, "Handwritten Digits (Draw)", 20, 10, &s_title)?;
        static_label(
            root,
            "SWM341  /  int8 FC  /  live classify",
            470,
            16,
            &s_muted,
        )?;

        // 左卡片：手写板（坐标相对 card_l 原点 (16,56)）
        static_label(&card_l, "INPUT / DRAW HERE", 16, 12, &s_head)?;
        let buf = DrawBuf::create(CANVAS_PX as u32, CANVAS_PX as u32, ColorFormat::RGB565)
            .ok_or(WidgetError::LvglNullPointer)?;
        let canvas = Canvas::new(&card_l, buf)?;
        canvas.pos(20, 44).size(CANVAS_PX, CANVAS_PX);
        self.canvas = Some(canvas);

        // CLEAR 按钮（卡片内 (16,316)，屏幕 (32,372)，288×34）
        let clear_btn: Obj<'static> = Obj::new(&card_l)?;
        clear_btn.pos(16, 316).size(CLEAR_W, CLEAR_H);
        clear_btn.add_style(&s_clear, Selector::DEFAULT);
        static_label(&clear_btn, "CLEAR", 96, 8, &s_clear_txt)?;
        let _ = Child::new(clear_btn);

        // 右卡片：推理结果（坐标相对 card_r 原点 (348,56)）
        static_label(&card_r, "INFERENCE", 16, 12, &s_head)?;
        self.pred = Some(label(&card_r, "-", 22, 36, &s_big)?);
        static_label(&card_r, "PREDICT", 16, 80, &s_muted)?;
        self.conf = Some(label(&card_r, "-", 148, 40, &s_num)?);
        static_label(&card_r, "CONFIDENCE", 148, 80, &s_muted)?;
        static_label(&card_r, "REC(ms):", 16, 98, &s_muted)?;
        self.rec_l = Some(label(&card_r, "-", 16 + 64, 98, &s_body)?);
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
        self.status_l = Some(label(
            &card_b,
            "write something and press 'classify' button",
            12,
            10,
            &s_muted,
        )?);

        let _ = (Child::new(card_l), Child::new(card_r), Child::new(card_b));
        let _ = (Child::new(rule_top),);

        // 首帧先清屏 + 占位。
        self.show_empty_pred();
        Ok(())
    }

    fn update(&mut self) -> Result<NavAction, WidgetError> {
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

/// 把任意尺寸灰度图（背景0/笔画255）等比放大居中画进 RGB565 canvas。
///
/// 直接拿 `lv_draw_buf_t` 的像素指针写 u16（整数最近邻放大）：零 draw task、
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

// ───────────────────────────────────────────────────────────────────────────
// DWT 计时
// ───────────────────────────────────────────────────────────────────────────

/// 启用 DWT 周期计数器（Cortex-M 标准做法）。之后 `cyc()` 读随 CPU 主频自增的计数。
fn enable_dwt() {
    unsafe {
        let demcr = 0xE000_EDFC as *mut u32;
        let ctrl = 0xE000_1000 as *mut u32;
        let v = core::ptr::read_volatile(demcr);
        core::ptr::write_volatile(demcr, v | (1u32 << 24));
        let c = core::ptr::read_volatile(ctrl);
        core::ptr::write_volatile(ctrl, c | (1u32 << 0));
    }
}

/// 读 DWT 周期计数器（@ CPU 主频）。两次相减即为经过的 CPU 周期数。
fn cyc() -> u32 {
    unsafe { core::ptr::read_volatile(0xE000_1004 as *const u32) }
}
