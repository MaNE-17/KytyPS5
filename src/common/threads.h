#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <condition_variable>
#include <mutex>
#endif

namespace Common {

void InitializeThreads();

using wait_poll_func_t = void (*)();

namespace Thread {

void SleepMicro(uint32_t micros);
void SleepNano(uint64_t nanos);
bool IsMainThread();
// Unique per thread; never reused by another thread.
int GetThreadIdUnique();
// Lets the calling thread yield the CPU to normal-priority threads (background work such as
// pipeline builds). Best effort: does nothing where the platform refuses.
void LowerCurrentThreadPriority();
// Lets the calling thread run ahead of normal-priority threads (the thread that translates guest
// GPU commands: the guest's own threads otherwise hold its CPUs for whole time slices while they
// load). Best effort.
void RaiseCurrentThreadPriority();
// The logical CPUs the process may use: those of its affinity mask where the platform has one,
// all of the host's otherwise. At least 1.
uint32_t AvailableProcessors();
// The CPUs the calling thread may run on, from a list such as "2,3,8-15" (logical CPUs 0-63 of
// its processor group); other threads are not kept off them. "auto" names the performance cores
// without the core of CPU 0, which serves most interrupts. False when nothing was changed (an
// empty or unusable list, a CPU without core classes, or a platform that refuses).
bool PinCurrentThreadToCpus(const char* list);

// The threads the host CPU layout places (KYTY_CPU_LAYOUT, below).
enum class CpuRole {
	// The thread that translates the guest's GPU commands (the frame's serial thread).
	GpuFeed,
	// The thread that replays its Vulkan commands to the driver.
	GpuRecord,
};

// Host CPU layout for CPUs of one core class with SMT (Ryzen and Intel CPUs without efficiency
// cores). On those, every busy thread of the process lands on the same few cores, and a thread
// whose SMT sibling is busy runs at about two thirds of its speed whatever its priority: priority
// orders threads that wait for one logical CPU, not two threads that share a core. The thread
// that feeds the GPU holds the frame rate where the CPU is the limit, so the layout gives it a
// physical core of its own (both of its logical CPUs, the second left idle), gives the Vulkan
// recording thread one logical CPU of a second core, and keeps every other thread of the process
// (the game's, helpers, the driver's) on the remaining logical CPUs, through Windows' CPU sets.
// KYTY_CPU_LAYOUT=0 leaves the scheduler as it was; =1 applies it on any CPU of 6 logical CPUs
// or more. Unset ("auto"): on a CPU of one core class with 8 logical CPUs or more. A hybrid CPU
// is left alone (KYTY_RENDER_CPUS=auto is the option there), and so is an explicit
// KYTY_RENDER_CPUS. Windows only.
//
// Places the calling thread in its role; the first call also sets the process' default CPUs.
// False when nothing was changed.
bool PlaceCurrentThread(CpuRole role);
// No layout for this process (the precompile pass uses every CPU). Before the first placement.
void DisableCpuLayout();
// One line that says what the layout does on this host, for the log.
const char* CpuLayoutText();

} // namespace Thread

// Recursive.
class Mutex {
public:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();

	KYTY_CLASS_NO_COPY(Mutex);

private:
	// A CRITICAL_SECTION.
	alignas(8) unsigned char m_cs[40];
#else
	void Lock() { m_mutex.lock(); }
	void Unlock() { m_mutex.unlock(); }

private:
	std::recursive_mutex m_mutex;
#endif

	friend class CondVar;
};

class CondVar {
public:
	CondVar() = default;

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void Signal();
	void SignalAll();
#else
	void Signal() { m_cv.notify_one(); }
	void SignalAll() { m_cv.notify_all(); }
#endif

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A CONDITION_VARIABLE, whose initial state is zero.
	void* m_cv = nullptr;
#else
	std::condition_variable_any m_cv;
#endif
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
