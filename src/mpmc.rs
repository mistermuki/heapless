//! A fixed capacity Multiple-Producer Multiple-Consumer (MPMC) lock-free queue.
//!
//! NOTE: This module requires atomic CAS operations. On targets where they're not natively available,
//! they are emulated by the [`portable-atomic`](https://crates.io/crates/portable-atomic) crate.
//!
//! # Example
//!
//! This queue can be constructed in "const context". Placing it in a `static` variable lets *all*
//! contexts (interrupts/threads/`main`) safely enqueue and dequeue items from it.
//!
//! ``` ignore
//! #![no_main]
//! #![no_std]
//!
//! use panic_semihosting as _;
//!
//! use cortex_m::{asm, peripheral::syst::SystClkSource};
//! use cortex_m_rt::{entry, exception};
//! use cortex_m_semihosting::hprintln;
//! use heapless::mpmc::Q2;
//!
//! static Q: Q2<u8> = Q2::new();
//!
//! #[entry]
//! fn main() -> ! {
//!     if let Some(p) = cortex_m::Peripherals::take() {
//!         let mut syst = p.SYST;
//!
//!         // configures the system timer to trigger a SysTick exception every second
//!         syst.set_clock_source(SystClkSource::Core);
//!         syst.set_reload(12_000_000);
//!         syst.enable_counter();
//!         syst.enable_interrupt();
//!     }
//!
//!     loop {
//!         if let Some(x) = Q.dequeue() {
//!             hprintln!("{}", x).ok();
//!         } else {
//!             asm::wfi();
//!         }
//!     }
//! }
//!
//! #[exception]
//! fn SysTick() {
//!     static mut COUNT: u8 = 0;
//!
//!     Q.enqueue(*COUNT).ok();
//!     *COUNT += 1;
//! }
//! ```
//!
//! # Benchmark
//!
//! Measured on a ARM Cortex-M3 core running at 8 MHz and with zero Flash wait cycles
//!
//! N| `Q8::<u8>::enqueue().ok()` (`z`) | `Q8::<u8>::dequeue()` (`z`) |
//! -|----------------------------------|-----------------------------|
//! 0|34                                |35                           |
//! 1|52                                |53                           |
//! 2|69                                |71                           |
//!
//! - `N` denotes the number of *interruptions*. On Cortex-M, an interruption consists of an
//!   interrupt handler preempting the would-be atomic section of the `enqueue`/`dequeue`
//!   operation. Note that it does *not* matter if the higher priority handler uses the queue or
//!   not.
//! - All execution times are in clock cycles. 1 clock cycle = 125 ns.
//! - Execution time is *dependent* of `mem::size_of::<T>()`. Both operations include one
//! `memcpy(T)` in their successful path.
//! - The optimization level is indicated in parentheses.
//! - The numbers reported correspond to the successful path (i.e. `Some` is returned by `dequeue`
//! and `Ok` is returned by `enqueue`).
//!
//! # Portability
//!
//! This module requires CAS atomic instructions which are not available on all architectures
//! (e.g.  ARMv6-M (`thumbv6m-none-eabi`) and MSP430 (`msp430-none-elf`)). These atomics can be
//! emulated however with [`portable-atomic`](https://crates.io/crates/portable-atomic), which is
//! enabled with the `cas` feature and is enabled by default for `thumbv6m-none-eabi` and `riscv32`
//! targets.
//!
//! # References
//!
//! This is an implementation of Dmitry Vyukov's ["Bounded MPMC queue"][0] minus the cache padding.
//!
//! [0]: http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue

use core::{cell::UnsafeCell, mem::MaybeUninit};

#[cfg(not(feature = "portable-atomic"))]
use core::sync::atomic;
#[cfg(feature = "portable-atomic")]
use portable_atomic as atomic;

use atomic::Ordering;

#[cfg(feature = "mpmc_large")]
type AtomicTargetSize = atomic::AtomicUsize;
#[cfg(not(feature = "mpmc_large"))]
type AtomicTargetSize = atomic::AtomicU8;

#[cfg(feature = "mpmc_large")]
type IntSize = usize;
#[cfg(not(feature = "mpmc_large"))]
type IntSize = u8;

/// MPMC queue with a capability for (2-1) 1 elements.
pub type Q2<T> = MpMcQueue<T, 2>;

/// MPMC queue with a capability for (4-1) 3 elements.
pub type Q4<T> = MpMcQueue<T, 4>;

/// MPMC queue with a capability for (8-1) 7 elements.
pub type Q8<T> = MpMcQueue<T, 8>;

/// MPMC queue with a capability for (16-1) 15 elements.
pub type Q16<T> = MpMcQueue<T, 16>;

/// MPMC queue with a capability for (32-1) 31 elements.
pub type Q32<T> = MpMcQueue<T, 32>;

/// MPMC queue with a capability for (64-1) 63 elements.
pub type Q64<T> = MpMcQueue<T, 64>;

/// MPMC queue with a capacity for N-1 elements
/// N must be a power of 2
/// The max value of N is u8::MAX - 1 if `mpmc_large` feature is not enabled.
pub struct MpMcQueue<T, const N: usize> {
    buffer: UnsafeCell<[Cell<T>; N]>,
    dequeue_pos: AtomicTargetSize,
    enqueue_pos: AtomicTargetSize,
}

impl<T, const N: usize> MpMcQueue<T, N> {
    const EMPTY_CELL: Cell<T> = Cell::new();
    const ASSERT: [(); 1] = [()];

    /// Creates an empty queue
    #[inline]
    pub const fn new() -> Self {
        // Const assert
        crate::sealed::greater_than_1::<N>();
        crate::sealed::power_of_two::<N>();

        // Const assert on size.
        #[allow(clippy::no_effect)]
        Self::ASSERT[(N >= (IntSize::MAX as usize)) as usize];

        let mut cell_count = 0;

        let mut result_cells: [Cell<T>; N] = [Self::EMPTY_CELL; N];
        while cell_count != N {
            result_cells[cell_count] = Cell::new();
            cell_count += 1;
        }

        Self {
            buffer: UnsafeCell::new(result_cells),
            dequeue_pos: AtomicTargetSize::new(0),
            enqueue_pos: AtomicTargetSize::new(0),
        }
    }

    /// Returns the numbers of elements in the queue
    #[inline]
    pub fn len(&self) -> usize {
        let front = self.enqueue_pos.load(Ordering::Relaxed);
        let back = self.dequeue_pos.load(Ordering::Relaxed);

        if front >= back {
            (front - back) as usize
        } else {
            (N - (back - front) as usize) % N
        }
    }

    /// Returns true if the queue contains N-1 elements.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == N - 1
    }

    /// Returns true if the queue contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the item in the front of the queue, or `None` if the queue is empty
    #[inline]
    pub fn dequeue(&self) -> Option<T> {
        unsafe {
            dequeue(
                self.buffer.get() as *mut _,
                &self.dequeue_pos,
                N,
                self.len(),
            )
        }
    }

    /// Adds an `item` to the end of the queue
    ///
    /// Returns back the `item` if the queue is full
    #[inline]
    pub fn enqueue(&self, item: T) -> Result<(), T> {
        unsafe {
            enqueue(
                self.buffer.get() as *mut _,
                &self.enqueue_pos,
                item,
                N,
                self.len(),
            )
        }
    }
}

impl<T, const N: usize> Default for MpMcQueue<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T, const N: usize> Sync for MpMcQueue<T, N> where T: Send {}

struct Cell<T> {
    data: MaybeUninit<T>,
}

impl<T> Cell<T> {
    const fn new() -> Self {
        Self {
            data: MaybeUninit::uninit(),
        }
    }
}

unsafe fn dequeue<T>(
    buffer: *mut Cell<T>,
    dequeue_pos: &AtomicTargetSize,
    capacity: usize,
    len: usize,
) -> Option<T> {
    let pos = dequeue_pos.load(Ordering::Relaxed);

    if len < 1 {
        return None;
    }

    let cell = buffer.add(usize::from(pos));

    dequeue_pos
        .fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
            Some((usize::from(v + 1) % capacity) as IntSize)
        })
        .unwrap();

    let data = (*cell).data.as_ptr().read();
    Some(data)
}

unsafe fn enqueue<T>(
    buffer: *mut Cell<T>,
    enqueue_pos: &AtomicTargetSize,
    item: T,
    capacity: usize,
    len: usize,
) -> Result<(), T> {
    // Loads the enqueue position from the atomic.
    let pos = enqueue_pos.load(Ordering::Relaxed);

    // Check remaining space.
    if (capacity - len - 1) < 1 {
        return Err(item);
    }

    let cell = buffer.add(usize::from(pos));
    (*cell).data.as_mut_ptr().write(item);

    enqueue_pos
        .fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
            Some((usize::from(v + 1) % capacity) as IntSize)
        })
        .unwrap();
    return Ok(());
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;

    use super::{MpMcQueue, Q32, Q4};

    // Ensure a `MpMcQueue` containing `!Send` values stays `!Send` itself.
    assert_not_impl_any!(MpMcQueue<*const (), 4>: Send);

    #[test]
    fn sanity() {
        let q = Q4::new();
        q.enqueue(0).unwrap();
        assert_eq!(q.len(), 1);
        q.enqueue(1).unwrap();
        assert_eq!(q.len(), 2);
        q.enqueue(2).unwrap();
        assert_eq!(q.len(), 3);

        assert!(q.is_full());
        assert!(q.enqueue(1).is_err());

        assert_eq!(q.dequeue(), Some(0));
        assert_eq!(q.len(), 2);
        assert!(!q.is_full());

        assert_eq!(q.dequeue(), Some(1));
        assert_eq!(q.len(), 1);
        assert!(!q.is_full());

        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.len(), 0);
        assert!(!q.is_full());

        assert_eq!(q.dequeue(), None);
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }

    #[test]
    fn drain_at_pos255() {
        let q = Q4::new();
        for _ in 0..255 {
            assert!(q.enqueue(0).is_ok());
            assert_eq!(q.len(), 1);
            assert_eq!(q.dequeue(), Some(0));
            assert!(q.is_empty());
        }
        // this should not block forever
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn full_at_wrapped_pos0() {
        let q = Q4::new();
        for _ in 0..254 {
            assert!(q.enqueue(0).is_ok());
            assert_eq!(q.len(), 1);
            assert_eq!(q.dequeue(), Some(0));
            assert!(q.is_empty());
        }
        assert!(q.enqueue(0).is_ok());
        assert_eq!(q.len(), 1);
        assert!(q.enqueue(1).is_ok());
        assert_eq!(q.len(), 2);
        assert!(q.enqueue(2).is_ok());
        assert_eq!(q.len(), 3);
        assert!(q.is_full());
        // this should not block forever
        assert!(q.enqueue(0).is_err());
    }

    #[test]
    fn len_all_the_way() {
        let q = Q32::new();

        // Count all the way up.
        for i in 0..31 {
            assert_eq!(q.len(), i);
            assert!(q.enqueue(i).is_ok());
        }
        assert!(q.is_full());

        // Count all the way down.
        for i in 0..31 {
            assert_eq!(q.len(), 31 - i);
            assert_eq!(q.dequeue(), Some(i));
        }
        assert!(q.is_empty());
    }
}
