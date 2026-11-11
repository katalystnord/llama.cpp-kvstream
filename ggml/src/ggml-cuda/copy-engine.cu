#include "copy-engine.cuh"

#include "ggml-backend-impl.h"
#include "ggml-copy-engine.h"

#include <cstdlib>
#include <vector>

#if defined(GGML_USE_HIP)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#define HSA_CHECK(x)                                                                    \
    do {                                                                                \
        const hsa_status_t s_ = (x);                                                    \
        if (s_ != HSA_STATUS_SUCCESS) {                                                 \
            const char * m_ = nullptr;                                                  \
            hsa_status_string(s_, &m_);                                                 \
            GGML_ABORT("%s failed: %s", #x, m_ ? m_ : "unknown HSA error");             \
        }                                                                               \
    } while (0)

// On AMD the copies go straight to an SDMA engine through ROCr, with no dependencies. HIP sends every stream's copies to
// one engine, and a copy that waits for its stream's kernels (e.g. reading back a graph's output) blocks the engine
// until they finish; if those kernels wait for this queue's copies (ggml_flash_attn_ext_set_sync), nothing moves. So
// this queue takes another engine: the highest H2D engine (HIP uses the first), or GGML_CUDA_COPY_ENGINE_SDMA (a mask).
//
// Events: the copies between two record() calls form a batch that counts down one signal. The signal starts at 1 (a
// guard, so that the copies that finish early cannot reach 0 first), each copy adds 1 and decrements it when done, and
// record() drops the guard.
struct ggml_backend_copy_engine {
    int                       device;
    hsa_agent_t               gpu {};
    hsa_agent_t               cpu {};
    hsa_amd_sdma_engine_id_t  engine {};
    std::vector<hsa_signal_t> events;  // the batch of each recorded event
    hsa_signal_t              batch {}; // the batch being filled
    std::vector<void *>       shared;
};

static ggml_backend_copy_engine_t ggml_cuda_copy_engine_init(ggml_backend_dev_t dev, int n_events) {
    auto * ce = new ggml_backend_copy_engine;
    ce->device = ggml_cuda_dev_index(dev);
    ggml_cuda_set_device(ce->device);

    // the GPU agent owns the device's memory
    void * probe = nullptr;
    CUDA_CHECK(cudaMalloc(&probe, 256));
    hsa_amd_pointer_info_t info = {};
    info.size = sizeof(info);
    HSA_CHECK(hsa_amd_pointer_info(probe, &info, nullptr, nullptr, nullptr));
    ce->gpu = info.agentOwner;
    CUDA_CHECK(cudaFree(probe));

    HSA_CHECK(hsa_iterate_agents([](hsa_agent_t a, void * data) {
        hsa_device_type_t t;
        hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
        auto * cpu = (hsa_agent_t *) data;
        if (t == HSA_DEVICE_TYPE_CPU && !cpu->handle) {
            *cpu = a;
        }
        return HSA_STATUS_SUCCESS;
    }, &ce->cpu));

    uint32_t mask = 0;
    HSA_CHECK(hsa_amd_memory_copy_engine_status(ce->gpu, ce->cpu, &mask));
    if (const char * env = getenv("GGML_CUDA_COPY_ENGINE_SDMA")) {
        mask = (uint32_t) strtoul(env, nullptr, 0);
    }
    GGML_ASSERT(mask != 0 && "no SDMA engine for host to device copies");
    uint32_t engine = 1u << (31 - __builtin_clz(mask));
    if (engine == 1) {
        GGML_LOG_WARN("%s: only SDMA engine 0x1 is available, which HIP also uses: copies may stall\n", __func__);
    }
    ce->engine = (hsa_amd_sdma_engine_id_t) engine;

    ce->events.resize(n_events);
    for (auto & s : ce->events) {
        HSA_CHECK(hsa_signal_create(0, 0, nullptr, &s));
    }
    HSA_CHECK(hsa_signal_create(1, 0, nullptr, &ce->batch));

    GGML_LOG_INFO("%s: host to device copies on SDMA engine 0x%x (available: 0x%x)\n", __func__, engine, mask);
    return ce;
}

static void ggml_cuda_copy_engine_free(ggml_backend_copy_engine_t ce) {
    for (auto & s : ce->events) {
        hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        hsa_signal_destroy(s);
    }
    // the open batch still holds its guard
    hsa_signal_wait_scacquire(ce->batch, HSA_SIGNAL_CONDITION_EQ, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    hsa_signal_destroy(ce->batch);
    ggml_cuda_set_device(ce->device);
    for (void * p : ce->shared) {
        CUDA_CHECK(cudaFreeHost(p));
    }
    delete ce;
}

static void ggml_cuda_copy_engine_copy_to_device(ggml_backend_copy_engine_t ce, void * dst, const void * src, size_t size) {
    hsa_signal_add_relaxed(ce->batch, 1);
    HSA_CHECK(hsa_amd_memory_async_copy_on_engine(dst, ce->gpu, src, ce->cpu, size, 0, nullptr, ce->batch, ce->engine, true));
}

static void ggml_cuda_copy_engine_record(ggml_backend_copy_engine_t ce, int i) {
    // the caller only records an event again once it is done, so its old batch signal is free to reuse
    std::swap(ce->events[i], ce->batch);
    hsa_signal_subtract_screlease(ce->events[i], 1);
    hsa_signal_store_relaxed(ce->batch, 1);
}

static bool ggml_cuda_copy_engine_done(ggml_backend_copy_engine_t ce, int i) {
    return hsa_signal_load_scacquire(ce->events[i]) == 0;
}

#else

// A non-blocking stream for host -> device copies.
struct ggml_backend_copy_engine {
    int                      device;
    cudaStream_t             stream;
    std::vector<cudaEvent_t> events;
    std::vector<void *>      shared;
};

static ggml_backend_copy_engine_t ggml_cuda_copy_engine_init(ggml_backend_dev_t dev, int n_events) {
    auto * ce = new ggml_backend_copy_engine;
    ce->device = ggml_cuda_dev_index(dev);
    ggml_cuda_set_device(ce->device);
    CUDA_CHECK(cudaStreamCreateWithFlags(&ce->stream, cudaStreamNonBlocking));
    ce->events.resize(n_events);
    for (auto & e : ce->events) {
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
    return ce;
}

static void ggml_cuda_copy_engine_free(ggml_backend_copy_engine_t ce) {
    ggml_cuda_set_device(ce->device);
    CUDA_CHECK(cudaStreamSynchronize(ce->stream));
    for (auto & e : ce->events) {
        CUDA_CHECK(cudaEventDestroy(e));
    }
    CUDA_CHECK(cudaStreamDestroy(ce->stream));
    for (void * p : ce->shared) {
        CUDA_CHECK(cudaFreeHost(p));
    }
    delete ce;
}

static void ggml_cuda_copy_engine_copy_to_device(ggml_backend_copy_engine_t ce, void * dst, const void * src, size_t size) {
    ggml_cuda_set_device(ce->device);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, ce->stream));
}

static void ggml_cuda_copy_engine_record(ggml_backend_copy_engine_t ce, int i) {
    ggml_cuda_set_device(ce->device);
    CUDA_CHECK(cudaEventRecord(ce->events[i], ce->stream));
}

static bool ggml_cuda_copy_engine_done(ggml_backend_copy_engine_t ce, int i) {
    const cudaError_t err = cudaEventQuery(ce->events[i]);
    if (err == cudaErrorNotReady) {
        (void) cudaGetLastError();
        return false;
    }
    CUDA_CHECK(err);
    return true;
}

#endif // defined(GGML_USE_HIP)

static void * ggml_cuda_copy_engine_alloc_shared(ggml_backend_copy_engine_t ce, size_t size) {
    ggml_cuda_set_device(ce->device);
    void * p = nullptr;
#if defined(GGML_USE_HIP)
    // the default (HIP_HOST_COHERENT unset) is non-coherent memory, which device kernels cannot poll
    CUDA_CHECK(hipHostMalloc(&p, size, hipHostMallocCoherent | hipHostMallocMapped));
#else
    CUDA_CHECK(cudaHostAlloc(&p, size, cudaHostAllocMapped));
#endif
    memset(p, 0, size);
    ce->shared.push_back(p);
    return p;
}

template <typename T>
static __global__ void k_copy_to_host(const T * __restrict__ src, T * __restrict__ dst, const size_t n) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        dst[i] = src[i];
    }
}

// On AMD, HIP can put a device -> host copy that waits for earlier kernels on the SDMA queue of the copy engine, and the
// kernels it waits for can wait for copies behind it in that queue. A kernel on the compute stream uses no SDMA queue.
static bool ggml_cuda_copy_engine_copy_to_host(ggml_backend_t backend, void * dst, const void * src, size_t size) {
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);

    void * dst_dev = nullptr;
    if (cudaHostGetDevicePointer(&dst_dev, dst, 0) != cudaSuccess) {
        (void) cudaGetLastError(); // dst is not pinned
        return false;
    }

    const int    block    = 256;
    const size_t max_grid = 1024;
    cudaStream_t stream   = ctx->stream();
    if ((((uintptr_t) src | (uintptr_t) dst_dev | size) % sizeof(int4)) == 0) {
        const size_t n = size/sizeof(int4);
        k_copy_to_host<<<std::min((n + block - 1)/block, max_grid), block, 0, stream>>>((const int4 *) src, (int4 *) dst_dev, n);
    } else {
        k_copy_to_host<<<std::min((size + block - 1)/block, max_grid), block, 0, stream>>>((const char *) src, (char *) dst_dev, size);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static const ggml_backend_copy_engine_i ggml_cuda_copy_engine_iface = {
    /* .init           = */ ggml_cuda_copy_engine_init,
    /* .free           = */ ggml_cuda_copy_engine_free,
    /* .alloc_shared   = */ ggml_cuda_copy_engine_alloc_shared,
    /* .copy_to_device = */ ggml_cuda_copy_engine_copy_to_device,
    /* .record         = */ ggml_cuda_copy_engine_record,
    /* .done           = */ ggml_cuda_copy_engine_done,
    /* .copy_to_host   = */ ggml_cuda_copy_engine_copy_to_host,
};

const ggml_backend_copy_engine_i * ggml_cuda_copy_engine_get_iface(void) {
    return &ggml_cuda_copy_engine_iface;
}
