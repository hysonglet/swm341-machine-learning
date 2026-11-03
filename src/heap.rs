//! 全局分配器：把 SDRAM 的一段当成堆。
//!
//! PNG 解码和 FAT 目录遍历都要用 `alloc`，而 SWM341 的 64K SRAM 装不下，
//! 所以堆必须放在板载 SDRAM 里。关键顺序：**先 `Sdramc::new_with_pins` 把 SDRAM 上电，
//! 再 `init_heap`**，否则 `HEAP.init` 往未上电的 SDRAM 写空闲链表头会写出垃圾，
//! 第一次分配就 HardFault。
//!
//! 注意模块名叫 `heap`（不是 `alloc`），以免和 `extern crate alloc;` 引入的 `alloc` crate 重名。

use embedded_alloc::Heap;

/// 在 `HEAP_BASE` 处划出 `HEAP_SIZE` 字节作为堆。必须在 SDRAM 已初始化之后调用。
///
/// 这里默认在 `0x8000_0000`（SDRAM 基址）起划 4 MiB；图像/解码的临时缓冲都落在里面，
/// 剩下的 SDRAM 仍可作帧缓冲（走裸指针，不用分配器）。
pub const HEAP_BASE: usize = 0x8000_0000;
pub const HEAP_SIZE: usize = 4 * 1024 * 1024;

// 全局分配器必须始终存在（哪怕当前 bin 不分配，依赖图也可能间接引用 `alloc` crate）。
// embedded-alloc 0.5 起没有 const `new()`，用 `empty()`；`init` 现在吃 `NonNull<u8>`。
#[global_allocator]
static HEAP: Heap = Heap::empty();

/// 初始化堆（一次性）。重复调用不安全，调用方保证只调一次。
///
/// # Safety
/// 必须在 SDRAM 控制器已初始化（已上电、可访问）之后调用，且 `HEAP_BASE` 起的
/// `HEAP_SIZE` 字节都不能被其它代码当作普通 RAM 使用。
pub unsafe fn init_heap() {
    // embedded-alloc 0.5：`init(start_addr: usize, size: usize)`。
    HEAP.init(HEAP_BASE, HEAP_SIZE);
}
