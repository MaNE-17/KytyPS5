#ifndef KYTY_COMMON_SPINWAIT_H_
#define KYTY_COMMON_SPINWAIT_H_

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#include <thread>

namespace Common {

// The waits of a spin lock. A waiter pauses for a short time, then gives its CPU to another ready
// thread at every round. A lock holder can be preempted where the process has more busy threads
// than the host has logical CPUs (an 8-core CPU with the game's threads, the emulator's and the
// driver's), and a waiter that only pauses then spins out its whole time slice while the holder
// waits for one; on a CPU with many more threads that hardly happens. The pause phase is measured
// in time stamp counter ticks, not in pauses: a pause takes about 140 cycles on recent Intel
// cores and far fewer on others, so a count of them is a different wait on every CPU.
class SpinWait final {
public:
	void Wait() {
#if defined(__x86_64__) || defined(_M_X64)
		if (m_start == 0) {
			m_start = __rdtsc();
		}
		if (m_yielded || __rdtsc() - m_start > PauseTicks) {
			m_yielded = true;
			YieldCpu();
			return;
		}
		_mm_pause();
#else
		if (++m_rounds > PauseRounds) {
			YieldCpu();
		}
#endif
	}

private:
	// About 20 to 40 microseconds at the counter rates of desktop CPUs (2 to 4 GHz).
#if defined(__x86_64__) || defined(_M_X64)
	static constexpr uint64_t PauseTicks = 80000;
#else
	static constexpr uint32_t PauseRounds = 4096;
#endif

	// SwitchToThread on Windows (MSVC's library), sched_yield elsewhere.
	static void YieldCpu() { std::this_thread::yield(); }

#if defined(__x86_64__) || defined(_M_X64)
	uint64_t m_start   = 0;
	bool     m_yielded = false;
#else
	uint32_t m_rounds = 0;
#endif
};

} // namespace Common

#endif // KYTY_COMMON_SPINWAIT_H_
