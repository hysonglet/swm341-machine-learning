/* SWM341 默认存储布局：512K flash @ 0x0，64K SRAM @ 0x2000_0000。
 *
 * 数值与 swm341-hal 自带的 memory.x 保持一致——两边都往链接搜索路径里放同名
 * 文件，内容不同的话谁先被找到就成了运气问题。
 *
 * 后续要在 SDRAM 里放图像/权重时，再像 swm341_slint 那样在这里补窗口，
 * 但运行时初始化的内存不要声明进 MEMORY（链接器会把 .bss 塞进去）。
 */
PROVIDE(FLASH_BASE = 0x00000000);
PROVIDE(FLASH_SIZE = 512K);
PROVIDE(RAM_BASE = 0x20000000);
PROVIDE(RAM_SIZE = 64K);

MEMORY
{
  FLASH : ORIGIN = FLASH_BASE, LENGTH = FLASH_SIZE
  RAM   : ORIGIN = RAM_BASE,   LENGTH = RAM_SIZE
}
