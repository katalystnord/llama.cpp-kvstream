#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Copy engine [EXPERIMENTAL]: a queue for host -> device copies that runs beside a device's compute stream, and host
// memory that the host and device kernels can both poll (e.g. the flags of ggml_flash_attn_ext_set_sync).
// Backends that have one return a ggml_backend_copy_engine_get_iface_t from get_proc_address("ggml_backend_copy_engine").
// The functions of one engine may be called from different host threads, but each one from one thread at a time.

typedef struct ggml_backend_copy_engine * ggml_backend_copy_engine_t;

struct ggml_backend_copy_engine_i {
    // n_events: events that record() and done() address by index
    ggml_backend_copy_engine_t (*init)(ggml_backend_dev_t dev, int n_events);
    void (*free)(ggml_backend_copy_engine_t ce);

    // zero-initialized host memory that is coherent with device kernels and has the same address on the device;
    // freed with the engine
    void * (*alloc_shared)(ggml_backend_copy_engine_t ce, size_t size);

    // queue a host -> device copy; src must be pinned (e.g. in a host buffer of the device)
    void (*copy_to_device)(ggml_backend_copy_engine_t ce, void * dst, const void * src, size_t size);

    // event i marks the current end of the queue; done(i) is true once the queue has passed it
    void (*record)(ggml_backend_copy_engine_t ce, int i);
    bool (*done)(ggml_backend_copy_engine_t ce, int i);

    // queue a device -> host copy as a kernel on the backend's compute stream, so no DMA queue holds it (see
    // kvstream/design/copy-stall.md); dst must be pinned; false if not possible (then use ggml_backend_tensor_get_async)
    bool (*copy_to_host)(ggml_backend_t backend, void * dst, const void * src, size_t size);
};

typedef const struct ggml_backend_copy_engine_i * (*ggml_backend_copy_engine_get_iface_t)(void);

#ifdef  __cplusplus
}
#endif
