#include "common/threads.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono> // IWYU pragma: keep
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/resource.h>
#include <unistd.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// IWYU pragma: no_include <winbase.h>
constexpr DWORD KYTY_CS_SPIN_COUNT = 4000;

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

// High-resolution waitable timers need Windows 10 1803, the floor MapViewOfFile3 already sets.
static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}
	// Closed when the thread exits.
	thread_local const std::unique_ptr<void, decltype(&CloseHandle)> timer {
	    CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
		                       TIMER_ALL_ACCESS),
	    &CloseHandle};
	LARGE_INTEGER due_time {};
	due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
	if (SetWaitableTimerEx(timer.get(), &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
		WaitForSingleObject(timer.get(), INFINITE);
	}
}

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <synchapi.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <winerror.h>

static CRITICAL_SECTION* ToCs(unsigned char* storage) {
	return reinterpret_cast<CRITICAL_SECTION*>(storage);
}

static CONDITION_VARIABLE* ToCv(void** storage) {
	return reinterpret_cast<CONDITION_VARIABLE*>(storage);
}

#endif

namespace Common {

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

static std::thread::id g_main_thread;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Windows runs the threads of a process whose window is not in the foreground at reduced speed
// where it can: on a CPU with performance and efficiency cores it places them on the efficiency
// cores, and it may coarsen their timers. The emulator's work is the same whether or not its
// window has the focus (a second monitor, a capture, an overlay in front): with the window in
// the background Saros ran at 24.7 flips/s on an i9-13900K and at 37.6 once its threads were
// kept on the performance cores. Opting out of both leaves the scheduler free to use every core
// as for a foreground process. KYTY_POWER_THROTTLING=1 keeps the system's default.
static void OptOutOfPowerThrottling() {
	if (const char* keep = std::getenv("KYTY_POWER_THROTTLING");
	    keep != nullptr && std::strcmp(keep, "1") == 0) {
		return;
	}
#ifndef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
#define PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION 0x4
#endif
	PROCESS_POWER_THROTTLING_STATE state {};
	state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
	state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED |
	                    PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
	state.StateMask   = 0;
	if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state)) ==
	    0) {
		// Before Windows 11 the timer resolution control is unknown: the execution speed alone.
		state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
		(void)SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
		                            sizeof(state));
	}
}
#endif

void InitializeThreads() {
	g_main_thread = std::this_thread::get_id();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	OptOutOfPowerThrottling();
#endif
}

void Thread::SleepMicro(uint32_t micros) {
	SleepNano(uint64_t {micros} * 1000);
}

void Thread::SleepNano(uint64_t nanos) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(__APPLE__) // no clock_nanosleep
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#else
	if (nanos == 0) {
		return;
	}
	constexpr uint64_t NANOS_PER_SEC = 1000000000;
	// An absolute deadline preserves the requested sleep across signal interruptions.
	timespec deadline {};
	clock_gettime(CLOCK_MONOTONIC, &deadline);
	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

void Thread::LowerCurrentThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// Linux applies nice values per thread.
	(void)setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 5);
#endif
}

uint32_t Thread::AvailableProcessors() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD_PTR process = 0;
	DWORD_PTR system  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) != 0 && process != 0) {
		uint32_t count = 0;
		for (; process != 0; process &= process - 1) {
			count++;
		}
		// (A mask covers one processor group; the host may have more CPUs than it names.)
		return count;
	}
#endif
	return std::max(std::thread::hardware_concurrency(), 1u);
}

void Thread::RaiseCurrentThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// Needs CAP_SYS_NICE or a nice rlimit; refused silently otherwise.
	(void)setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), -5);
#endif
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The logical CPUs (bit n: CPU n) of the fastest core class the process may use, without the
// core that CPU 0 belongs to; 0 when the CPU has one class only or the classes are unknown.
static DWORD_PTR PerformanceCpuMask() {
	DWORD_PTR process = 0;
	DWORD_PTR system  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) == 0) {
		return 0;
	}
	ULONG length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) {
		return 0;
	}
	const auto buffer = std::make_unique<unsigned char[]>(length);
	auto*      first  = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.get());
	if (GetSystemCpuSetInformation(first, length, &length, GetCurrentProcess(), 0) == 0) {
		return 0;
	}
	unsigned  fastest         = 0;
	unsigned  slowest         = 255;
	int       interrupt_core  = -1;
	const auto for_each = [&](const auto& visit) {
		for (ULONG offset = 0; offset < length;) {
			const auto* info =
			    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.get() + offset);
			if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 &&
			    info->CpuSet.LogicalProcessorIndex < 64 &&
			    ((process >> info->CpuSet.LogicalProcessorIndex) & 1u) != 0) {
				visit(info->CpuSet);
			}
			offset += info->Size;
		}
	};
	for_each([&](const auto& cpu) {
		fastest = std::max<unsigned>(fastest, cpu.EfficiencyClass);
		slowest = std::min<unsigned>(slowest, cpu.EfficiencyClass);
		if (cpu.LogicalProcessorIndex == 0) {
			interrupt_core = cpu.CoreIndex;
		}
	});
	if (fastest == slowest) {
		return 0;
	}
	DWORD_PTR mask = 0;
	for_each([&](const auto& cpu) {
		if (cpu.EfficiencyClass == fastest && cpu.CoreIndex != interrupt_core) {
			mask |= DWORD_PTR {1} << cpu.LogicalProcessorIndex;
		}
	});
	return mask;
}
#endif

bool Thread::PinCurrentThreadToCpus(const char* list) {
	if (list == nullptr || *list == '\0') {
		return false;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD_PTR mask = 0;
	if (std::strcmp(list, "auto") == 0) {
		mask = PerformanceCpuMask();
	} else {
		// "a,b,c-d".
		for (const char* text = list; *text != '\0';) {
			char*      end   = nullptr;
			const auto first = std::strtoul(text, &end, 10);
			if (end == text) {
				return false;
			}
			auto last = first;
			if (*end == '-') {
				const char* range = end + 1;
				last              = std::strtoul(range, &end, 10);
				if (end == range) {
					return false;
				}
			}
			for (auto cpu = first; cpu <= last && cpu < 64; cpu++) {
				mask |= DWORD_PTR {1} << cpu;
			}
			text = *end == ',' ? end + 1 : end;
			if (*end != ',' && *end != '\0') {
				return false;
			}
		}
	}
	return mask != 0 && SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
	return false;
#endif
}

namespace {

struct CpuLayout {
	bool        active = false;
	std::string text;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::vector<ULONG> feed; // CPU set ids
	std::vector<ULONG> record;
	std::vector<ULONG> others;
#endif
};

std::atomic<bool> g_cpu_layout_disabled {false};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// SYSTEM_CPU_SET_INFORMATION's flags (AllFlags), spelled out: older SDKs lack the field names.
constexpr BYTE CPU_SET_PARKED                      = 0x1;
constexpr BYTE CPU_SET_ALLOCATED                   = 0x2;
constexpr BYTE CPU_SET_ALLOCATED_TO_TARGET_PROCESS = 0x4;
constexpr BYTE CPU_SET_REAL_TIME                   = 0x8;

std::string CpuList(const std::vector<BYTE>& cpus) {
	std::string text;
	for (const auto cpu: cpus) {
		text += (text.empty() ? "" : ",") + std::to_string(cpu);
	}
	return text;
}

CpuLayout BuildCpuLayout() {
	CpuLayout   layout;
	const char* mode = std::getenv("KYTY_CPU_LAYOUT");
	const bool  forced =
	    mode != nullptr && (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "on") == 0);
	if (mode != nullptr && (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0)) {
		layout.text = "off (KYTY_CPU_LAYOUT=0)";
		return layout;
	}
	if (g_cpu_layout_disabled.load()) {
		layout.text = "off (not for this run)";
		return layout;
	}
	if (const char* cpus = std::getenv("KYTY_RENDER_CPUS"); cpus != nullptr && *cpus != '\0') {
		layout.text = "off (KYTY_RENDER_CPUS is set)";
		return layout;
	}
	DWORD_PTR process = 0;
	DWORD_PTR system  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) == 0) {
		layout.text = "off (no affinity mask)";
		return layout;
	}
	ULONG length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) {
		layout.text = "off (no CPU sets)";
		return layout;
	}
	const auto buffer = std::make_unique<unsigned char[]>(length);
	if (GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.get()),
	                               length, &length, GetCurrentProcess(), 0) == 0) {
		layout.text = "off (no CPU sets)";
		return layout;
	}
	struct Cpu {
		ULONG id;
		BYTE  index;
		BYTE  core;
		BYTE  efficiency;
	};
	std::vector<Cpu> cpus;
	for (ULONG offset = 0; offset < length;) {
		const auto* info =
		    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.get() + offset);
		offset += info->Size;
		if (info->Type != CpuSetInformation || info->CpuSet.Group != 0 ||
		    info->CpuSet.LogicalProcessorIndex >= 64 ||
		    ((process >> info->CpuSet.LogicalProcessorIndex) & 1u) == 0) {
			continue;
		}
		const BYTE flags = info->CpuSet.AllFlags;
		// Another process' reserved CPUs and real-time ones are not the process' to give out.
		if (((flags & CPU_SET_ALLOCATED) != 0 &&
		     (flags & CPU_SET_ALLOCATED_TO_TARGET_PROCESS) == 0) ||
		    (flags & CPU_SET_REAL_TIME) != 0) {
			continue;
		}
		(void)CPU_SET_PARKED;
		cpus.push_back({info->CpuSet.Id, info->CpuSet.LogicalProcessorIndex, info->CpuSet.CoreIndex,
		                info->CpuSet.EfficiencyClass});
	}
	if (cpus.size() < (forced ? 6u : 8u)) {
		layout.text =
		    "off (" + std::to_string(cpus.size()) + " logical CPUs: too few to give one core away)";
		return layout;
	}
	struct Core {
		BYTE             index       = 0;
		BYTE             efficiency  = 0;
		bool             serves_cpu0 = false;
		std::vector<Cpu> cpus;
	};
	std::vector<Core> cores;
	for (const auto& cpu: cpus) {
		auto it = std::find_if(cores.begin(), cores.end(),
		                       [&](const Core& c) { return c.index == cpu.core; });
		if (it == cores.end()) {
			cores.push_back({cpu.core, cpu.efficiency, false, {}});
			it = std::prev(cores.end());
		}
		it->efficiency  = std::max(it->efficiency, cpu.efficiency);
		it->serves_cpu0 = it->serves_cpu0 || cpu.index == 0;
		it->cpus.push_back(cpu);
	}
	// A hybrid CPU: classes of core where the slower class has no SMT (Intel's efficiency cores).
	// Ryzen CPUs with "preferred cores" may report two classes too, all of them with SMT.
	BYTE fastest = 0;
	BYTE slowest = 255;
	for (const auto& core: cores) {
		fastest = std::max(fastest, core.efficiency);
		slowest = std::min(slowest, core.efficiency);
	}
	bool hybrid = false;
	if (fastest != slowest) {
		for (const auto& core: cores) {
			hybrid = hybrid || (core.efficiency == slowest && core.cpus.size() == 1);
		}
	}
	if (hybrid && !forced) {
		layout.text = "off (hybrid CPU: KYTY_RENDER_CPUS=auto places the GPU thread there)";
		return layout;
	}
	std::vector<const Core*> candidates;
	for (const auto& core: cores) {
		if (!core.serves_cpu0) {
			candidates.push_back(&core);
		}
	}
	// The fastest class first (Windows' preferred cores, where it names them), then the order of
	// the cores.
	std::stable_sort(candidates.begin(), candidates.end(),
	                 [](const Core* a, const Core* b) { return a->efficiency > b->efficiency; });
	if (candidates.size() < 2) {
		layout.text = "off (one core)";
		return layout;
	}
	const Core*       feed_core = candidates[0];
	std::vector<BYTE> feed_cpus;
	for (const auto& cpu: feed_core->cpus) {
		layout.feed.push_back(cpu.id);
		feed_cpus.push_back(cpu.index);
	}
	// The recording thread gets a CPU of its own where enough are left: 12 logical CPUs or more.
	std::vector<BYTE> record_cpus;
	if (cpus.size() >= 12) {
		const auto& cpu = candidates[1]->cpus.front();
		layout.record.push_back(cpu.id);
		record_cpus.push_back(cpu.index);
	}
	std::vector<BYTE> other_cpus;
	for (const auto& cpu: cpus) {
		if (std::find(layout.feed.begin(), layout.feed.end(), cpu.id) == layout.feed.end() &&
		    std::find(layout.record.begin(), layout.record.end(), cpu.id) == layout.record.end()) {
			layout.others.push_back(cpu.id);
			other_cpus.push_back(cpu.index);
		}
	}
	if (layout.others.size() < 4) {
		layout.text = "off (too few CPUs left for the game)";
		layout.feed.clear();
		layout.record.clear();
		layout.others.clear();
		return layout;
	}
	layout.active = true;
	layout.text   = "GPU thread on CPUs " + CpuList(feed_cpus) + " (a core of its own)" +
	                (record_cpus.empty() ? std::string {}
	                                     : ", recording thread on CPU " + CpuList(record_cpus)) +
	                ", the game and the rest on " + std::to_string(other_cpus.size()) + " CPUs (" +
	                CpuList(other_cpus) + "); KYTY_CPU_LAYOUT=0 turns it off";
	return layout;
}
#endif

CpuLayout& TheCpuLayout() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static CpuLayout layout = BuildCpuLayout();
#else
	static CpuLayout layout {false, "off (Windows only)"};
#endif
	return layout;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
std::once_flag g_cpu_layout_applied;
#endif

} // namespace

void Thread::DisableCpuLayout() {
	g_cpu_layout_disabled.store(true);
}

const char* Thread::CpuLayoutText() {
	return TheCpuLayout().text.c_str();
}

bool Thread::PlaceCurrentThread(CpuRole role) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	auto& layout = TheCpuLayout();
	if (!layout.active) {
		return false;
	}
	std::call_once(g_cpu_layout_applied, [&] {
		if (SetProcessDefaultCpuSets(GetCurrentProcess(), layout.others.data(),
		                             static_cast<ULONG>(layout.others.size())) == 0) {
			layout.active = false;
			layout.text   = "off (Windows refused the process' CPU sets, error " +
			                std::to_string(GetLastError()) + ")";
		}
	});
	if (!layout.active) {
		return false;
	}
	const auto& ids = role == CpuRole::GpuFeed ? layout.feed : layout.record;
	if (ids.empty()) {
		return false;
	}
	return SetThreadSelectedCpuSets(GetCurrentThread(), ids.data(),
	                                static_cast<ULONG>(ids.size())) != 0;
#else
	(void)role;
	return false;
#endif
}

int Thread::GetThreadIdUnique() {
	static std::atomic<int> counter = 0;
	static thread_local int tid     = ++counter;
	return tid;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
Mutex::Mutex() {
	static_assert(sizeof(CRITICAL_SECTION) == sizeof(m_cs) &&
	              alignof(CRITICAL_SECTION) <= alignof(Mutex));
	InitializeCriticalSectionAndSpinCount(ToCs(m_cs), KYTY_CS_SPIN_COUNT);
}

Mutex::~Mutex() {
	DeleteCriticalSection(ToCs(m_cs));
}

void Mutex::Lock() {
	EnterCriticalSection(ToCs(m_cs));
}

void Mutex::Unlock() {
	LeaveCriticalSection(ToCs(m_cs));
}

void CondVar::Signal() {
	WakeConditionVariable(ToCv(&m_cv));
}

void CondVar::SignalAll() {
	WakeAllConditionVariable(ToCv(&m_cv));
}
#endif

void CondVar::Wait(Mutex* mutex) {
	auto* callback = g_cond_wait_poll_callback;
	if (callback == nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs), INFINITE);
#else
		m_cv.wait(mutex->m_mutex);
#endif
	} else if (!WaitFor(mutex, 10000)) {
		mutex->Unlock();
		callback();
		mutex->Lock();
	}
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs),
	                                micros < 1000 ? 1 : micros / 1000) != 0 ||
	       GetLastError() != ERROR_TIMEOUT;
#else
	return m_cv.wait_for(mutex->m_mutex, std::chrono::microseconds(micros)) ==
	       std::cv_status::no_timeout;
#endif
}

} // namespace Common
