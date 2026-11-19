#include "graphics/host_gpu/vulkanRecording.h"

#include "common/assert.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>

namespace Libs::Graphics::VulkanRecording {
namespace {

// 1: a producer's commands are queued for its recording thread; 0: they go to the driver from
// the producer as without a recording thread (after what is queued).
Common::LiveControl::Switch g_recording("vulkan_recording", 1);
// Pauses the idle recording thread spins before it blocks: a blocked thread costs the producer
// a wake-up system call per batch.
Common::LiveControl::Switch g_recording_spin("vulkan_recording_spin", 4000);
// Diagnostics: the number of waits of the producer for the recording thread still to be logged,
// with the call that waited.
Common::LiveControl::Switch g_recording_log("vulkan_recording_log", 0);

// Per frame in `measure`: commands queued, batches handed over, direct calls of the producer
// that found queued commands and waited for them, and the times the producer waited for room.
Common::LiveControl::Counter g_queued_commands("record_cmds");
Common::LiveControl::Counter g_published_chunks("record_chunks");
Common::LiveControl::Counter g_direct_waits("record_waits");
Common::LiveControl::Counter g_room_waits("record_full");
Common::LiveControl::Counter g_sleeps("record_sleeps");

constexpr size_t   CHUNK_BYTES = 64 * 1024;
// Deep enough that a burst of commands does not block the producer in WaitForRoom.
constexpr size_t   CHUNK_COUNT = 64;
constexpr uint32_t END         = std::numeric_limits<uint32_t>::max();

// The driver's entry points (the dispatcher's entries before the thunks replaced them).
vk::detail::DispatchLoaderDynamic original;

struct Header {
	void (*execute)(void*) = nullptr;
	void (*destroy)(void*) = nullptr;
	uint32_t payload       = 0;
	uint32_t next          = END;
};

struct Chunk {
	alignas(64) std::array<std::byte, CHUNK_BYTES> data;
	size_t                                         used  = 0;
	uint32_t                                       first = END;
	uint32_t                                       last  = END;
	uint32_t                                       count = 0;
};

// One command and everything its arguments point to, in one chunk. A command that cannot be
// copied (valid false) publishes nothing of what was copied for it.
class Writer {
public:
	explicit Writer(Chunk& chunk): data(chunk.data.data()), used(chunk.used) {}

	bool       valid    = true;
	bool       overflow = false;
	std::byte* data;
	size_t     used;
	uint32_t   command = END;

	void* Allocate(size_t bytes, size_t alignment) {
		if (!valid) {
			return nullptr;
		}
		const size_t offset = (used + alignment - 1) & ~(alignment - 1);
		if (bytes > CHUNK_BYTES || offset > CHUNK_BYTES - bytes) {
			valid    = false;
			overflow = true;
			return nullptr;
		}
		used = offset + bytes;
		return data + offset;
	}

	const void* Copy(const void* source, size_t bytes) {
		if (source == nullptr || bytes == 0) {
			return nullptr;
		}
		auto* target = Allocate(bytes, alignof(std::max_align_t));
		if (target != nullptr) {
			std::memcpy(target, source, bytes);
		}
		return target;
	}

	template <typename T>
	const T* Copy(const T* source, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (source == nullptr || count == 0) {
			return nullptr;
		}
		if (count > CHUNK_BYTES / sizeof(T)) {
			valid    = false;
			overflow = true;
			return nullptr;
		}
		auto* target = static_cast<T*>(Allocate(count * sizeof(T), alignof(T)));
		if (target == nullptr) {
			return nullptr;
		}
		std::memcpy(target, source, count * sizeof(T));
		for (size_t i = 0; i < count && valid; ++i) {
			auto& value = target[i];
			if constexpr (requires { value.pNext; }) {
				// An extension chain is not followed: such a call goes to the driver at once.
				if (value.pNext != nullptr) {
					valid = false;
					break;
				}
			}
			if constexpr (std::is_same_v<T, VkDependencyInfo>) {
				value.pMemoryBarriers = Copy(value.pMemoryBarriers, value.memoryBarrierCount);
				value.pBufferMemoryBarriers =
				    Copy(value.pBufferMemoryBarriers, value.bufferMemoryBarrierCount);
				value.pImageMemoryBarriers =
				    Copy(value.pImageMemoryBarriers, value.imageMemoryBarrierCount);
			} else if constexpr (std::is_same_v<T, VkRenderingInfo>) {
				value.pColorAttachments  = Copy(value.pColorAttachments, value.colorAttachmentCount);
				value.pDepthAttachment   = Copy(value.pDepthAttachment, 1);
				value.pStencilAttachment = Copy(value.pStencilAttachment, 1);
			} else if constexpr (std::is_same_v<T, VkCommandBufferBeginInfo>) {
				// Primary command buffers only.
				if (value.pInheritanceInfo != nullptr) {
					valid = false;
				}
			} else if constexpr (std::is_same_v<T, VkWriteDescriptorSet>) {
				// Fields the descriptor type does not select need not point to readable
				// memory: only the selected payload is copied.
				value.pImageInfo       = nullptr;
				value.pBufferInfo      = nullptr;
				value.pTexelBufferView = nullptr;
				switch (value.descriptorType) {
					case VK_DESCRIPTOR_TYPE_SAMPLER:
					case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
					case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
					case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
					case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
						value.pImageInfo = Copy(source[i].pImageInfo, value.descriptorCount);
						break;
					case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
					case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
					case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
					case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
						value.pBufferInfo = Copy(source[i].pBufferInfo, value.descriptorCount);
						break;
					case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
					case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
						value.pTexelBufferView =
						    Copy(source[i].pTexelBufferView, value.descriptorCount);
						break;
					default: valid = false; break;
				}
			}
		}
		return target;
	}

	template <typename Function>
	void Command(Function function) {
		static_assert(std::is_nothrow_move_constructible_v<Function>);
		static_assert(std::is_nothrow_destructible_v<Function>);
		auto* header  = static_cast<Header*>(Allocate(sizeof(Header), alignof(Header)));
		void* payload = Allocate(sizeof(Function), alignof(Function));
		if (!valid) {
			return;
		}
		new (payload) Function(std::move(function));
		new (header) Header {
		    .execute = [](void* pointer) { (*static_cast<Function*>(pointer))(); },
		    .destroy = nullptr,
		    .payload = static_cast<uint32_t>(static_cast<std::byte*>(payload) - data),
		};
		if constexpr (!std::is_trivially_destructible_v<Function>) {
			header->destroy = [](void* pointer) { static_cast<Function*>(pointer)->~Function(); };
		}
		command = static_cast<uint32_t>(reinterpret_cast<std::byte*>(header) - data);
	}
};

// A ring of chunks between one producer and its recording thread. Each side touches a chunk
// only while it owns it: the producer fills the chunk at `sequence` and hands it over by
// publishing; the recording thread replays published chunks in order and gives each back by
// completing it.
class Stream {
public:
	Stream(): m_chunks(std::make_unique<std::array<Chunk, CHUNK_COUNT>>()) {
		m_worker = std::thread([this] { Run(); });
	}

	~Stream() {
		Drain();
		m_stopping.store(true, std::memory_order_release);
		// atomic::wait ends on a changed value, not on a notification alone.
		m_published.fetch_add(1, std::memory_order_release);
		m_published.notify_one();
		m_worker.join();
	}

	Stream(const Stream&)            = delete;
	Stream& operator=(const Stream&) = delete;

	template <typename Capture>
	bool Enqueue(Capture capture) {
		for (int attempt = 0; attempt < 2; ++attempt) {
			WaitForRoom();
			auto&  chunk = (*m_chunks)[m_sequence % CHUNK_COUNT];
			Writer writer(chunk);
			capture(writer);
			if (writer.valid && writer.command != END) {
				if (chunk.last != END) {
					reinterpret_cast<Header*>(chunk.data.data() + chunk.last)->next = writer.command;
				} else {
					chunk.first = writer.command;
				}
				chunk.last = writer.command;
				chunk.used = writer.used;
				++chunk.count;
				g_queued_commands.Add();
				if (chunk.count >= 128 || chunk.used >= CHUNK_BYTES * 3 / 4) {
					Publish();
				}
				return true;
			}
			// A command that did not fit gets a chunk of its own; one that cannot be copied
			// (or is larger than a chunk) goes to the driver at once.
			if (!writer.overflow || chunk.count == 0) {
				break;
			}
			Publish();
		}
		return false;
	}

	void Flush() { Publish(); }

	// Commands queued and not yet replayed (a direct call would wait for them).
	[[nodiscard]] bool Pending() const {
		return m_completed.load(std::memory_order_acquire) != m_sequence ||
		       (*m_chunks)[m_sequence % CHUNK_COUNT].count != 0;
	}

	void Drain() {
		Publish();
		for (auto done = m_completed.load(std::memory_order_acquire); done != m_sequence;
		     done      = m_completed.load(std::memory_order_acquire)) {
			m_completed.wait(done, std::memory_order_acquire);
		}
	}

private:
	void WaitForRoom() {
		auto done = m_completed.load(std::memory_order_acquire);
		if (m_sequence - done < CHUNK_COUNT) {
			return;
		}
		g_room_waits.Add();
		for (; m_sequence - done >= CHUNK_COUNT; done = m_completed.load(std::memory_order_acquire)) {
			m_completed.wait(done, std::memory_order_acquire);
		}
	}

	void Publish() {
		WaitForRoom();
		const auto& chunk = (*m_chunks)[m_sequence % CHUNK_COUNT];
		if (chunk.count == 0) {
			return;
		}
		g_published_chunks.Add();
		m_published.store(++m_sequence, std::memory_order_seq_cst);
		// The recording thread is woken only when it said it blocks (both sides seq_cst): a
		// wake-up is a system call, and the thread is usually busy or still spinning.
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			m_published.notify_one();
		}
	}

	void Run() {
		KYTY_PROFILER_THREAD("Thread_GpuRecord");
		Common::LiveControl::RegisterThread("record");
		// Its latency is in every wait of the producer for the GPU.
		Common::Thread::RaiseCurrentThreadPriority();
		// A logical CPU of its own where the host CPU layout gives one (threads.h).
		(void)Common::Thread::PlaceCurrentThread(Common::Thread::CpuRole::GpuRecord);
		uint64_t current = 0;
		for (;;) {
			for (auto ready = m_published.load(std::memory_order_acquire); current == ready;
			     ready      = m_published.load(std::memory_order_acquire)) {
				const auto spins = g_recording_spin.Get();
				for (uint32_t spin = 0; spin < spins && ready == current; ++spin) {
					_mm_pause();
					ready = m_published.load(std::memory_order_acquire);
				}
				if (ready != current) {
					break;
				}
				g_sleeps.Add();
				m_sleeping.store(true, std::memory_order_seq_cst);
				if (m_published.load(std::memory_order_seq_cst) == current) {
					m_published.wait(current, std::memory_order_seq_cst);
				}
				m_sleeping.store(false, std::memory_order_relaxed);
			}
			if (m_stopping.load(std::memory_order_acquire)) {
				return;
			}
			auto& chunk = (*m_chunks)[current % CHUNK_COUNT];
			for (auto offset = chunk.first; offset != END;) {
				const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + offset);
				header->execute(chunk.data.data() + header->payload);
				if (header->destroy != nullptr) {
					header->destroy(chunk.data.data() + header->payload);
				}
				offset = header->next;
			}
			chunk.used  = 0;
			chunk.count = 0;
			chunk.first = chunk.last = END;
			m_completed.store(++current, std::memory_order_release);
			m_completed.notify_one();
		}
	}

	std::unique_ptr<std::array<Chunk, CHUNK_COUNT>> m_chunks;
	// The producer's: chunks published so far (the chunk it fills is the next one).
	uint64_t                          m_sequence = 0;
	alignas(64) std::atomic<uint64_t> m_published {0};
	alignas(64) std::atomic<uint64_t> m_completed {0};
	alignas(64) std::atomic<bool>     m_sleeping {false};
	std::atomic<bool>                 m_stopping {false};
	std::thread                       m_worker;
};

thread_local std::unique_ptr<Stream> t_producer;

Stream* RecordingStream() {
	return t_producer != nullptr && g_recording.On() ? t_producer.get() : nullptr;
}

// A call of the producer that goes to the driver itself (defined below, after the thunks' names).
void BeforeDirect(uint32_t thunk);
// The driver's answer to a queued call whose thunk answered VK_SUCCESS.
void CheckReplayed(VkResult result, uint32_t thunk);

#include "vulkanRecordingThunks.inc"

// A call of the producer that goes to the driver itself: everything queued is replayed first.
[[gnu::noinline]] void BeforeDirect(uint32_t thunk) {
	if (t_producer == nullptr || !t_producer->Pending()) {
		return;
	}
	g_direct_waits.Add();
	if (const auto left = g_recording_log.Get(); left != 0) {
		g_recording_log.Set(left - 1u);
		LOGF("LIVE vulkan recording: %s waits for the recording thread\n", g_thunk_names[thunk]);
	}
	t_producer->Drain();
}

// The driver's answer to a queued call whose thunk answered VK_SUCCESS.
void CheckReplayed(VkResult result, uint32_t thunk) {
	if (result != VK_SUCCESS) {
		EXIT("Vulkan recording: %s failed on the recording thread: %s\n", g_thunk_names[thunk],
		     vk::to_string(static_cast<vk::Result>(result)).c_str());
	}
}

} // namespace

void Install() {
	InstallThunks();
}

bool Recording() {
	return RecordingStream() != nullptr;
}

void Drain() {
	if (t_producer != nullptr) {
		t_producer->Drain();
	}
}

bool EnqueueOrdered(void (*function)(const void*), const void* data, size_t size, bool publish) {
	auto* stream = RecordingStream();
	if (stream == nullptr || function == nullptr) {
		return false;
	}
	const bool queued = stream->Enqueue([&](Writer& writer) {
		const void* copied = writer.Copy(data, size);
		writer.Command([function, copied] { function(copied); });
	});
	if (queued && publish) {
		stream->Flush();
	}
	return queued;
}

ProducerScope::ProducerScope() {
	EXIT_IF(t_producer != nullptr);
	t_producer = std::make_unique<Stream>();
}

ProducerScope::~ProducerScope() {
	t_producer.reset();
}

} // namespace Libs::Graphics::VulkanRecording
