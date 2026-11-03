//! 极简板端断言：把每个用例的 PASS/FAIL 打到 RTT。
//!
//! `no_std` 目标上没有 test harness，所以自检就是"跑一遍 + 跟期望值比 + 计数"。
//! 每个历程最后打印一行汇总，失败也会照样跑完（看清楚到底哪几个算子不对），
//! 而不是在第一个 assert 上 halt。

use crate::nn::Result;
use rtt_target::rprintln;

#[derive(Default)]
pub struct Checker {
    passed: u32,
    failed: u32,
    section: &'static str,
}

impl Checker {
    pub const fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            section: "",
        }
    }

    /// 换一个分组标题（打印一行分隔，便于在 RTT 里定位）。
    pub fn section(&mut self, title: &'static str) {
        self.section = title;
        rprintln!("\n--- {} ---", title);
    }

    pub fn pass(&mut self, what: &str) {
        self.passed += 1;
        rprintln!("  [ ok ] {}", what);
    }

    pub fn fail(&mut self, what: &str) {
        self.failed += 1;
        rprintln!("  [FAIL] {} ({})", what, self.section);
    }

    pub fn expect(&mut self, what: &str, condition: bool) {
        if condition {
            self.pass(what);
        } else {
            self.fail(what);
        }
    }

    /// 算子返回 `Ok(())`（embedded-nn 的 `Result`，参数错会返回 `ArgumentError`）。
    pub fn expect_ok(&mut self, what: &str, result: Result<()>) {
        match result {
            Ok(()) => self.pass(what),
            Err(err) => {
                self.failed += 1;
                rprintln!("  [FAIL] {} (返回 {:?})", what, err);
            }
        }
    }

    /// 逐元素比较；不一致时把期望/实际整段打出来。
    pub fn expect_i8(&mut self, what: &str, got: &[i8], want: &[i8]) {
        if got == want {
            self.pass(what);
        } else {
            self.failed += 1;
            rprintln!("  [FAIL] {}", what);
            rprintln!("         want {:?}", want);
            rprintln!("         got  {:?}", got);
        }
    }

    pub fn expect_i32_slice(&mut self, what: &str, got: &[i32], want: &[i32]) {
        if got == want {
            self.pass(what);
        } else {
            self.failed += 1;
            rprintln!("  [FAIL] {}", what);
            rprintln!("         want {:?}", want);
            rprintln!("         got  {:?}", got);
        }
    }

    pub fn expect_i32(&mut self, what: &str, got: i32, want: i32) {
        if got == want {
            self.pass(what);
        } else {
            self.failed += 1;
            rprintln!("  [FAIL] {}: want {}, got {}", what, want, got);
        }
    }

    pub fn passed(&self) -> u32 {
        self.passed
    }

    pub fn failed(&self) -> u32 {
        self.failed
    }

    /// 打印汇总并返回"是否全绿"。
    pub fn summary(&self) -> bool {
        rprintln!(
            "\n==== {} passed, {} failed ====",
            self.passed,
            self.failed
        );
        self.failed == 0
    }
}
