//! 周期计数用的 DWT 封装。
//!
//! CMSIS-NN 的 s8 内核在 DSP 路径下每个 SMLAD 算 2 个乘加，所以"每 MAC 花多少
//! 周期"能直观看出 SIMD 有没有生效：纯 C 实现一般 1.x cycles/MAC，DSP 内核
//! 通常低于 1。

use crate::hal;

pub struct CycleMeter<'d> {
    dwt: hal::dwt::Dwt<'d>,
}

impl<'d> CycleMeter<'d> {
    /// 拿走 DWT 单例，并立刻启动自由运行的周期计数器。
    pub fn new(peri: impl hal::Peripheral<P = hal::Dwt> + 'd) -> Self {
        let dwt = hal::dwt::Dwt::new_cycle_counter(peri);
        Self { dwt }
    }

    /// 跑 `f`，返回结果和耗时周期数。
    pub fn time<R>(&mut self, f: impl FnOnce() -> R) -> (R, u32) {
        self.dwt.measure(f)
    }

    pub fn has_cycle_counter(&self) -> bool {
        self.dwt.has_cycle_counter()
    }
}
