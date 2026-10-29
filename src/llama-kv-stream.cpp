#include "llama-kv-stream.h"

#include "llama-impl.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

llama_kv_stream::llama_kv_stream(const std::vector<layer> & layers, uint32_t n_slots, uint32_t n_slots_decode, uint32_t kv_size,
        ggml_backend_dev_t dev, bool no_alloc) :
    layers(layers), n_slots(n_slots), kv_size(kv_size), n_pages((kv_size + page_size - 1)/page_size), dev(dev) {
    GGML_ASSERT(!layers.empty());

    for (const auto & l : layers) {
        // the device slots share one row layout
        if (l.k->ne[0] != layers[0].k->ne[0] || l.v->ne[0] != layers[0].v->ne[0]) {
            throw std::runtime_error("KV streaming requires the same K/V width in every layer");
        }
    }

    size_t size = 0;
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        host_offs.push_back(size);
        size += (size_t) n_pages*page_size*(row_size(ikv, false) + row_size(ikv, true));
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (!buft) {
        buft = ggml_backend_cpu_buffer_type();
    }
    host_buf.reset(ggml_backend_buft_alloc_buffer(buft, no_alloc ? 0 : size));
    if (!host_buf) {
        throw std::runtime_error("failed to allocate the KV stream host store");
    }

    // two device slots, each the size of a layer until set_backend knows whether prefill reads the pages in place
    alloc_slots((int64_t) n_pages*page_size, no_alloc);

    // pages stream when a phase's pool cannot hold the context: the prefill pool, or the decode budget, which holds the
    // pool and the slots (at least one ring page)
    const bool streams = n_slots < n_pages || (n_slots_decode > 0 && n_slots_decode <= n_pages);
    if (streams) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_iface = (ggml_backend_copy_engine_get_iface_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_engine");
        if (!get_iface) {
            throw std::runtime_error(format("KV streaming: the resident budgets cannot hold all %u pages, and %s has no "
                                            "copy engine to stream the rest", n_pages, ggml_backend_dev_name(dev)));
        }
        ce_iface = get_iface();
        if (!no_alloc) {
            ce    = ce_iface->init(dev, n_events);
            flags = (uint32_t *) ce_iface->alloc_shared(ce, 2*layers.size()*sizeof(uint32_t));
        }
    }

    page_slot.assign(n_pages, -1);
    slot_page.assign(n_slots, -1);
    page_ring.assign(n_pages, -1);
    dirty.assign(n_pages, 0);
    saved.assign(n_pages, 0);

    const char * debug = getenv("LLAMA_KV_STREAM_DEBUG");
    debug_rotate = debug && strcmp(debug, "rotate") == 0;

    if (const char * timeout = getenv("LLAMA_KV_STREAM_TIMEOUT")) {
        timeout_us = (int64_t) (atof(timeout)*1e6);
    }
    if (const char * frac = getenv("LLAMA_KV_SPLIT")) {
        if (strcmp(frac, "auto") == 0) {
            tuner.on   = true;
            split_frac = split_step; // only to enable the split below; the tuner starts at 0
        } else {
            split_frac = std::clamp((float) atof(frac), 0.0f, 1.0f);
        }
    }
    split_threads = std::max(1, (int) std::thread::hardware_concurrency()*2/3);
    if (const char * threads = getenv("LLAMA_KV_SPLIT_THREADS")) {
        split_threads = std::max(1, atoi(threads));
    }
    split_desc.resize(layers.size());
    if (debug_rotate && n_slots < n_pages) {
        LLAMA_LOG_WARN("%s: LLAMA_KV_STREAM_DEBUG=rotate needs a pool that holds the context, ignored\n", __func__);
        debug_rotate = false;
    }

    LLAMA_LOG_INFO("%s: %u of %u pages resident (%u tokens), %10s host store = %8.2f MiB, slots = %8.2f MiB%s\n", __func__,
            n_slots, n_pages, n_slots*page_size, ggml_backend_buffer_name(host_buf.get()), size/1024.0/1024.0,
            ggml_backend_buffer_get_size(slots_buf.get())/1024.0/1024.0, debug_rotate ? " (debug: rotate)" : "");
    if (n_slots_decode > 0) {
        // the decode budget also holds the slots of the pages that stream, so its pool shrinks as more of them stream
        LLAMA_LOG_INFO("%s: between prefills, the decode budget holds up to %u of %u pages (%u tokens)\n", __func__,
                std::min(n_slots_decode, n_pages), n_pages, std::min(n_slots_decode, n_pages)*page_size);
    }
    if (streams) {
        LLAMA_LOG_INFO("%s: the other pages stream from the host store into the ubatches that attend them\n", __func__);
    }
}

llama_kv_stream::~llama_kv_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    split_cv.notify_all();
    if (copier.joinable()) {
        copier.join();
    }
    if (publisher.joinable()) {
        publisher.join();
    }
    if (splitter.joinable()) {
        splitter.join();
    }
    if (split_pool) {
        ggml_threadpool_free(split_pool);
    }
    for (auto & j : jobs) {
        if (j.wb) {
            ggml_backend_event_free(j.wb);
        }
    }
    if (wb_pending) {
        ggml_backend_event_free(wb_pending);
    }
    if (ce) {
        ce_iface->free(ce);
    }
}

uint32_t llama_kv_stream::slots_for(size_t bytes, size_t bytes_per_cell) {
    return (uint32_t) std::min<size_t>(bytes/(bytes_per_cell*page_size), UINT32_MAX);
}

void llama_kv_stream::alloc_slots(int64_t rows, bool no_alloc) {
    slots_buf.reset();
    ggml_init_params params = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    slots_ctx.reset(ggml_init(params));

    for (int s = 0; s < 2; ++s) {
        slots[s].k = ggml_new_tensor_2d(slots_ctx.get(), layers[0].k->type, layers[0].k->ne[0], rows);
        slots[s].v = ggml_new_tensor_2d(slots_ctx.get(), layers[0].v->type, layers[0].v->ne[0], rows);
        ggml_format_name(slots[s].k, "kv_stream_slot_k%d", s);
        ggml_format_name(slots[s].v, "kv_stream_slot_v%d", s);
    }

    ggml_backend_buffer_type_t buft_dev = ggml_backend_buffer_get_type(layers[0].k->buffer);
    if (no_alloc) {
        // dummy buffer, so that the scheduler does not try to allocate the slots
        slots_buf.reset(ggml_backend_buft_alloc_buffer(buft_dev, 0));
        for (ggml_tensor * t = ggml_get_first_tensor(slots_ctx.get()); t; t = ggml_get_next_tensor(slots_ctx.get(), t)) {
            t->buffer = slots_buf.get();
        }
    } else {
        slots_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(slots_ctx.get(), buft_dev));
    }
    if (!slots_buf) {
        throw std::runtime_error("failed to allocate the KV stream device slots");
    }
    ggml_backend_buffer_clear(slots_buf.get(), 0);
}

void llama_kv_stream::set_backend(ggml_backend_t backend, int64_t n_head) {
    this->backend = backend;

    // does the backend's attention take paged K/V for a prefill ubatch too? then prefill needs no gathers
    ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };

    const ggml_tensor * pk = layers[0].k;
    const ggml_tensor * pv = layers[0].v;
    const int64_t d    = 256;
    const int64_t n_kv = pk->ne[0]/d;
    const int64_t nb   = 512;

    auto heads = [&](const ggml_tensor * pool) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), pool->type, pool->ne[0], pool->ne[1]);
        return ggml_view_4d(ctx.get(), t, d, t->ne[1], n_kv, 1, t->nb[1], ggml_row_size(t->type, d), ggml_nbytes(t), 0);
    };
    ggml_tensor * q     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, nb, n_head, 1);
    ggml_tensor * mask  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, page_size, nb, 1, 1);
    ggml_tensor * pages = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    ggml_tensor * fa    = ggml_flash_attn_ext(ctx.get(), q, heads(pk), heads(pv), mask, 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_pages(fa, pages, 0, page_size, nullptr, nullptr);
    ggml_fa_page_run  probe_run  = { 0, 0, 1, 1 };
    ggml_fa_page_runs probe_runs = { 1, &probe_run };
    ggml_flash_attn_ext_set_page_runs(fa, &probe_runs);
    ggml_prec_set_acc(fa, GGML_PREC_F32);

    prefill_paged = ggml_backend_supports_op(backend, fa);
    LLAMA_LOG_INFO("%s: prefill attends %s\n", __func__, prefill_paged ? "the pages directly" : "gathers of the pages");

    // split attention: does the backend's decode attention merge a part from the CPU?
    if (split_frac > 0.0f && !split_q) {
        bool ok = ce && prefill_paged && !debug_rotate;
        if (ok) {
            ggml_tensor * q1   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, 1, n_head, 1);
            ggml_tensor * m1   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, page_size, 1, 1, 1);
            ggml_tensor * fa1  = ggml_flash_attn_ext(ctx.get(), q1, heads(pk), heads(pv), m1, 1.0f, 0.0f, 0.0f);
            ggml_tensor * tag2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
            float    probe_f = 0.0f;
            uint32_t probe_u = 0;
            const ggml_fa_split probe_split = { &probe_f, &probe_u, &probe_f, &probe_f, &probe_u, 1.0f };
            ggml_flash_attn_ext_set_pages(fa1, pages, 0, page_size, nullptr, nullptr);
            ggml_flash_attn_ext_set_split(fa1, tag2, &probe_split);
            ggml_prec_set_acc(fa1, GGML_PREC_F32);
            ok = ggml_backend_supports_op(backend, fa1);
        }
        if (ok) {
            const size_t n_layer = layers.size();
            const size_t rows    = (size_t) decode_max_tokens*n_head; // (query token, head) rows per layer
            split_n_head = n_head;
            split_q     = (float *)    ce_iface->alloc_shared(ce, n_layer*rows*d*sizeof(float));
            split_out   = (float *)    ce_iface->alloc_shared(ce, n_layer*rows*d*sizeof(float));
            split_lse   = (float *)    ce_iface->alloc_shared(ce, n_layer*rows*sizeof(float));
            split_flags = (uint32_t *) ce_iface->alloc_shared(ce, 2*n_layer*sizeof(uint32_t));
            for (size_t ikv = 0; ikv < n_layer; ++ikv) {
                split_desc[ikv] = { split_q + ikv*rows*d, split_flags + ikv, split_out + ikv*rows*d, split_lse + ikv*rows,
                    split_flags + n_layer + ikv, split_desc[ikv].scale };
            }
            if (tuner.on) {
                split_frac = 0.0f;
                LLAMA_LOG_INFO("%s: split attention: the CPU (%d threads) attends a measured share of the pages that "
                        "would stream (auto), shared buffers %.2f MiB\n", __func__, split_threads,
                        n_layer*rows*(2*d + 1)*sizeof(float)/1024.0/1024.0);
            } else {
                LLAMA_LOG_INFO("%s: split attention: the CPU (%d threads) attends %.0f%% of the pages that would stream, "
                        "shared buffers %.2f MiB\n", __func__, split_threads, 100.0*split_frac,
                        n_layer*rows*(2*d + 1)*sizeof(float)/1024.0/1024.0);
            }
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_KV_SPLIT is set, but %s; split attention is off\n", __func__,
                    !ce ? "no pages stream" : !prefill_paged ? "the backend does not attend pages directly" :
                    debug_rotate ? "LLAMA_KV_STREAM_DEBUG=rotate is set" : "the backend's attention does not merge a CPU part");
            split_frac = 0.0f;
            tuner.on   = false;
        }
    }

    // Without gathers, a slot only holds the pages that stream, packed: the attended pages that are not resident. The
    // pool is full of attended pages whenever any page streams (free slots take attended pages first, and pages outside
    // the attended range leave fewer to attend), so at most n_pages - n_slots stream per ubatch.
    if (prefill_paged && ggml_backend_buffer_get_size(slots_buf.get()) > 0 && !debug_rotate) {
        const uint32_t n_ring = std::max<uint32_t>(1, n_pages > n_slots ? n_pages - n_slots : 1);
        alloc_slots((int64_t) n_ring*page_size, false);
        LLAMA_LOG_INFO("%s: slots hold %u streamed pages, %8.2f MiB\n", __func__, n_ring,
                ggml_backend_buffer_get_size(slots_buf.get())/1024.0/1024.0);
    }
}

size_t llama_kv_stream::row_size(int32_t ikv, bool is_v) const {
    const ggml_tensor * t = is_v ? layers[ikv].v : layers[ikv].k;
    return ggml_row_size(t->type, t->ne[0]);
}

char * llama_kv_stream::host(int32_t ikv, bool is_v, uint32_t page) const {
    char * base = (char *) ggml_backend_buffer_get_base(host_buf.get()) + host_offs[ikv];
    if (is_v) {
        base += (size_t) n_pages*page_size*row_size(ikv, false);
    }
    return base + (size_t) page*page_size*row_size(ikv, is_v);
}

void llama_kv_stream::zero_host(uint32_t page) {
    // rows nobody wrote are masked, but must not hold NaNs (0*NaN in P*V)
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        memset(host(ikv, false, page), 0, page_size*row_size(ikv, false));
        memset(host(ikv, true,  page), 0, page_size*row_size(ikv, true));
    }
    saved[page] = 1;
}

void llama_kv_stream::assign(uint32_t page, uint32_t slot) {
    GGML_ASSERT(page_slot[page] < 0 && slot_page[slot] < 0);

    page_slot[page] = slot;
    slot_page[slot] = page;
    dirty[page] = 0;
    layout_id++;

    if (saved[page]) {
        GGML_ASSERT(backend && "llama_kv_stream: no backend for page moves");
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (bool is_v : { false, true }) {
                const size_t bytes = page_size*row_size(ikv, is_v);
                ggml_backend_tensor_set_async(backend, is_v ? layers[ikv].v : layers[ikv].k, host(ikv, is_v, page), slot*bytes, bytes);
            }
        }
    }
}

void llama_kv_stream::evict(uint32_t page) {
    const int32_t slot = page_slot[page];
    GGML_ASSERT(slot >= 0);

    if (dirty[page]) {
        GGML_ASSERT(backend && "llama_kv_stream: no backend for page moves");
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (bool is_v : { false, true }) {
                const size_t bytes = page_size*row_size(ikv, is_v);
                ggml_backend_tensor_get_async(backend, is_v ? layers[ikv].v : layers[ikv].k, host(ikv, is_v, page), slot*bytes, bytes);
            }
        }
        saved[page] = 1;
        dirty[page] = 0;

        // the copies that stream this page must wait for the write-back
        if (ce) {
            if (!wb_pending) {
                wb_pending = ggml_backend_event_new(dev);
            }
            ggml_backend_event_record(wb_pending, backend);
        }
    }

    page_slot[page] = -1;
    slot_page[slot] = -1;
    layout_id++;
}

void llama_kv_stream::prepare(const std::vector<uint32_t> & cells, uint32_t n_kv) {
    GGML_ASSERT(n_kv % page_size == 0);

    n_prepared++;

    const uint32_t n_att = std::min(n_kv/page_size, n_pages);

    std::vector<uint8_t> written(n_pages, 0);
    for (uint32_t c : cells) {
        written[c/page_size] = 1;
    }

    if (debug_rotate) {
        // every page moves: write back the resident ones, then bring the needed ones into rotated slots
        rotation++;
        for (uint32_t p = 0; p < n_pages; ++p) {
            if (page_slot[p] >= 0) {
                evict(p);
            }
        }
        for (uint32_t p = 0; p < n_pages; ++p) {
            if (p < n_att || written[p]) {
                assign(p, (p + rotation) % n_slots);
            }
        }
    } else {
        uint32_t free_slot = 0;
        auto next_free = [&]() {
            while (free_slot < n_slots && slot_page[free_slot] >= 0) {
                free_slot++;
            }
            return free_slot;
        };

        // the written pages are resident: evict the oldest other page if there is no free slot
        for (uint32_t p = 0; p < n_pages; ++p) {
            if (!written[p] || page_slot[p] >= 0) {
                continue;
            }
            uint32_t slot = next_free();
            if (slot == n_slots) {
                uint32_t victim = 0;
                while (victim < n_pages && (page_slot[victim] < 0 || written[victim])) {
                    victim++;
                }
                GGML_ASSERT(victim < n_pages && "llama_kv_stream: the pool is smaller than a ubatch");
                slot = page_slot[victim];
                evict(victim);
            }
            assign(p, slot);
        }

        // free slots take the newest attended pages
        for (uint32_t p = n_att; p-- > 0 && next_free() < n_slots; ) {
            if (page_slot[p] < 0) {
                assign(p, next_free());
            }
        }
    }

    for (uint32_t c : cells) {
        dirty[c/page_size] = 1;
    }

    // the attended pages that are not resident stream: packed for the decode kernel, in place for a gather
    const bool decode = cells.size() <= decode_max_tokens || prefill_paged;

    std::fill(page_ring.begin(), page_ring.end(), -1);
    n_streamed = 0;

    // split attention: the CPU takes its share of the pages that would stream, the oldest ones, while they form one
    // block from the first page (the oldest pages are the first to leave the pool, so they usually do)
    n_split = 0;
    if (split_q) {
        const bool decode_ub = cells.size() <= decode_max_tokens;
        uint32_t n_lead = 0;
        while (n_lead < n_att && page_slot[n_lead] < 0) {
            n_lead++;
        }
        uint32_t n_out = 0;
        for (uint32_t p = 0; p < n_att; ++p) {
            n_out += page_slot[p] < 0;
        }
        const float share = tuner.on ? split_tune(n_out, decode_ub, *std::min_element(cells.begin(), cells.end())) : split_frac;
        if (decode_ub) {
            n_split = std::min(n_lead, (uint32_t) (share*n_out));
        }
    }

    std::vector<run> copies;
    for (uint32_t p = 0; p < n_att; ++p) {
        if (page_slot[p] >= 0) {
            continue;
        }
        if (!saved[p]) {
            zero_host(p);
        }
        if (p < n_split) {
            continue; // the CPU reads it in the host store
        }
        page_ring[p] = n_streamed++;
        const uint32_t dst = decode ? page_ring[p] : p;
        if (!copies.empty() && copies.back().page + copies.back().n == p && copies.back().slot + copies.back().n == dst) {
            copies.back().n++;
        } else {
            copies.push_back({ p, dst, 1 });
        }
    }

    // packed, the streamed pages must fit the slots (sized in set_backend)
    GGML_ASSERT(!decode || (int64_t) n_streamed*page_size <= slots[0].k->ne[1]);

    run_list.clear();
    for (uint32_t p = 0; p < n_att; ++p) {
        const int32_t pool = page_slot[p] >= 0;
        const int32_t pos  = pool ? page_slot[p] : page_ring[p];
        if (!run_list.empty() && run_list.back().pool == pool && run_list.back().pos + run_list.back().n == pos) {
            run_list.back().n++;
        } else {
            run_list.push_back({ (int32_t) p, pos, 1, pool });
        }
    }
    runs_desc = { (int32_t) run_list.size(), run_list.data() };

    if (n_split > 0) {
        split_tag++;
        {
            std::lock_guard<std::mutex> lock(mtx);
            split_jobs.push_back({ split_tag, n_split*page_size, (uint32_t) cells.size() });
        }
        split_cv.notify_all();
        if (!splitter.joinable()) {
            splitter = std::thread([this]() { splitter_main(); });
        }
    }

    if (n_streamed == 0) {
        return;
    }

    GGML_ASSERT(ce && "llama_kv_stream: no copy engine");

    tag++;
    {
        std::lock_guard<std::mutex> lock(mtx);
        jobs.push_back({ tag, std::move(copies), wb_pending, cells.size() <= decode_max_tokens, n_prepared });
        n_jobs++;
    }
    wb_pending = nullptr;
    cv.notify_all();

    if (!copier.joinable()) {
        copier    = std::thread([this]() { copier_main(); });
        publisher = std::thread([this]() { publisher_main(); });
    }
}

template <typename F>
bool llama_kv_stream::wait_for(F cond, const char * what, int32_t ikv, uint32_t wait_tag) {
    const int64_t t0 = ggml_time_us();
    for (uint32_t i = 0; !cond(); ++i) {
        if (stop) {
            return false;
        }
        cpu_relax();
        if (i % 1024 == 0 && ggml_time_us() - t0 > timeout_us) {
            LLAMA_LOG_ERROR("%s: stalled for %.1f s waiting for %s (layer %d, tag %u)\n", __func__,
                    (ggml_time_us() - t0)/1e6, what, ikv, wait_tag);
            log_state();
            GGML_ABORT("KV streaming stalled");
        }
    }
    return true;
}

void llama_kv_stream::log_state() const {
    LLAMA_LOG_ERROR("%s: tag %u, jobs %llu queued / %llu done, layer copies %llu queued / %llu published\n", __func__, tag,
            (unsigned long long) n_jobs, (unsigned long long) n_jobs_done.load(),
            (unsigned long long) n_enqueued.load(), (unsigned long long) n_published.load());
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        LLAMA_LOG_ERROR("%s:   layer %2zu: ready %u consumed %u\n", __func__, ikv,
                __atomic_load_n(ready(ikv), __ATOMIC_ACQUIRE), __atomic_load_n(consumed(ikv), __ATOMIC_ACQUIRE));
    }
}

void llama_kv_stream::copier_main() {
    uint64_t g = 0;

    // the last layer that read each device slot, and its tag
    span reader[2] = { { -1, 0 }, { -1, 0 } };

    prefetch pf;

    // queue the copies of layer ikv into its slot once the slot's previous reader released it; false if stopping
    auto queue_layer = [&](int32_t ikv, uint32_t t, const std::vector<run> & copies, bool is_prefetch) {
        const int s = ikv % 2;

        if (reader[s].ikv >= 0) {
            const span r = reader[s];
            if (!wait_for([&]() { return __atomic_load_n(consumed(r.ikv), __ATOMIC_ACQUIRE) >= r.tag; },
                          "the attention to release a slot", r.ikv, r.tag)) {
                return false;
            }
        }
        if (!wait_for([&]() { return g - n_published.load(std::memory_order_acquire) < (uint64_t) n_events; },
                      "a free copy event", ikv, t)) {
            return false;
        }

        for (const auto & r : copies) {
            for (bool is_v : { false, true }) {
                ggml_tensor * dst = is_v ? slots[s].v : slots[s].k;
                const size_t bytes = (size_t) page_size*row_size(ikv, is_v);
                ce_iface->copy_to_device(ce, (char *) dst->data + r.slot*bytes, host(ikv, is_v, r.page), r.n*bytes);
            }
        }
        ce_iface->record(ce, g % n_events);
        spans[g % n_events] = { ikv, t, is_prefetch };
        {
            // under the lock, so that a publisher about to sleep cannot miss it
            std::lock_guard<std::mutex> lock(mtx);
            n_enqueued.store(++g, std::memory_order_release);
        }
        cv.notify_all();
        // a prefetched slot has no reader until the next job claims it
        if (!is_prefetch) {
            reader[s] = { ikv, t };
        }
        return true;
    };

    auto same_copies = [](const std::vector<run> & a, const std::vector<run> & b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const run & x, const run & y) {
            return x.page == y.page && x.slot == y.slot && x.n == y.n;
        });
    };

    const int32_t n_layer = (int32_t) layers.size();

    while (true) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return stop || !jobs.empty(); });
            if (stop) {
                return;
            }
            j = std::move(jobs.front());
            jobs.pop_front();
        }

        if (j.wb) {
            ggml_backend_event_synchronize(j.wb);
            ggml_backend_event_free(j.wb);
        }

        // the prefetched layers hold this job's pages if it is the ubatch right after the prefetch's job and streams the
        // same copies: publish them once they landed. A ubatch in between that streamed nothing does not advance the tag,
        // and may have evicted pages that the prefetch copied before their write-back.
        int32_t first = 0;
        if (pf.valid && pf.ubatch == j.ubatch && pf.epoch == epoch.load(std::memory_order_acquire) &&
            same_copies(pf.copies, j.copies)) {
            if (!wait_for([&]() { return n_published.load(std::memory_order_acquire) >= pf.g_end; },
                          "the prefetched copies to land", 0, j.tag)) {
                return;
            }
            first = std::min(n_prefetch, n_layer);
            for (int32_t ikv = 0; ikv < first; ++ikv) {
                __atomic_store_n(flags + ikv, j.tag, __ATOMIC_RELEASE);
                reader[ikv % 2] = { ikv, j.tag };
            }
        }
        pf.valid = false;

        for (int32_t ikv = first; ikv < n_layer; ++ikv) {
            if (!queue_layer(ikv, j.tag, j.copies, false)) {
                return;
            }
        }

        // nothing queued behind a decode ubatch: copy the first layers of the next one while the host works
        bool idle = false;
        if (j.small) {
            std::lock_guard<std::mutex> lock(mtx);
            idle = jobs.empty() && !stop;
        }
        if (idle) {
            pf.epoch = epoch.load(std::memory_order_acquire);
            for (int32_t ikv = 0; ikv < std::min(n_prefetch, n_layer); ++ikv) {
                if (!queue_layer(ikv, j.tag + 1, j.copies, true)) {
                    return;
                }
            }
            pf.valid  = true;
            pf.ubatch = j.ubatch + 1;
            pf.g_end  = g;
            pf.copies = std::move(j.copies);
        }

        n_jobs_done.fetch_add(1, std::memory_order_release);
    }
}

void llama_kv_stream::publisher_main() {
    uint64_t g = 0;

    while (true) {
        // spin briefly for the next copy, then sleep until the copier queues one
        for (int i = 0; i < 4096 && n_enqueued.load(std::memory_order_acquire) <= g; ++i) {
            cpu_relax();
        }
        if (n_enqueued.load(std::memory_order_acquire) <= g) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return stop || n_enqueued.load(std::memory_order_acquire) > g; });
            if (stop) {
                return;
            }
        }

        const span sp = spans[g % n_events];
        if (!wait_for([&]() { return ce_iface->done(ce, g % n_events); }, "copies to land", sp.ikv, sp.tag)) {
            return;
        }

        if (!sp.prefetch) {
            __atomic_store_n(flags + sp.ikv, sp.tag, __ATOMIC_RELEASE);
        }
        n_published.store(++g, std::memory_order_release);
    }
}

// LLAMA_KV_SPLIT=auto: the share for the coming ubatch. The time between two consecutive decode prepares is the previous
// ubatch's (with the host's sampling), and the distance between their first cells is the tokens it kept (the accepted
// drafts plus one when verifying drafts: rejected cells are written again). Both count for the share that ubatch
// used. Windows of split_window ubatches after the first two (graph rebuild); a window's time per token is its total
// time over its total tokens, without ubatches slower than three times the median.
float llama_kv_stream::split_tune(uint32_t n_out, bool decode, uint32_t cell0) {
    auto & t = tuner;
    const int64_t now = ggml_time_us();
    const bool counted = decode && t.n_last + 1 == n_prepared && n_out > 0 &&
        cell0 > t.c_last && cell0 - t.c_last <= decode_max_tokens;
    if (counted) {
        if (t.skip > 0) {
            t.skip--;
        } else {
            t.samples.push_back({ now - t.t_last, cell0 - t.c_last });
        }
    }
    if (decode) {
        t.t_last = now;
        t.n_last = n_prepared;
        t.c_last = cell0;
    }
    if (n_out == 0) {
        return 0.0f; // every attended page is resident: nothing to split
    }

    const float cur = t.phase == 1 ? t.probe : t.base;
    if (t.phase == 2) {
        // settled: look again after a while, or when the streamed part has changed by a quarter
        if (counted && ++t.settled >= 512) {
            t.phase = 0;
        } else if (n_out > t.n_out + t.n_out/4 || n_out + n_out/4 < t.n_out) {
            t.phase = 0;
        }
        if (t.phase == 0) {
            t.samples.clear();
            t.skip = 2;
            t.turned = false;
            t.leaped = false;
        }
        return t.base;
    }
    if ((int) t.samples.size() < split_window) {
        return cur;
    }

    std::vector<int64_t> dts;
    for (const auto & x : t.samples) {
        dts.push_back(x.first);
    }
    std::nth_element(dts.begin(), dts.begin() + dts.size()/2, dts.end());
    const int64_t median = dts[dts.size()/2];
    int64_t  sum_us  = 0;
    uint64_t sum_tok = 0;
    for (const auto & x : t.samples) {
        if (x.first <= 3*median) {
            sum_us  += x.first;
            sum_tok += x.second;
        }
    }
    const double mean = (double) sum_us/std::max<uint64_t>(sum_tok, 1); // us per token
    t.samples.clear();
    t.skip = 2;

    auto settle = [&]() {
        t.phase   = 2;
        t.settled = 0;
        t.n_out   = n_out;
        if (getenv("LLAMA_KV_SPLIT_DEBUG")) {
            fprintf(stderr, "llama_kv_stream: split auto: settled at %.2f (%.1f ms per token, %u streamed pages)\n",
                    t.base, t.t_base/1e3, n_out);
        }
        return t.base;
    };
    auto try_probe = [&]() {
        const float p = std::clamp(t.base + t.dir*split_step, 0.0f, 0.9f);
        if (fabsf(p - t.base) < 1e-3f) {
            return false;
        }
        t.probe = p;
        t.phase = 1;
        return true;
    };

    if (getenv("LLAMA_KV_SPLIT_DEBUG")) {
        fprintf(stderr, "llama_kv_stream: split auto: share %.2f: %.1f ms per token\n", cur, mean/1e3);
    }
    if (t.phase == 0) {
        t.t_base = mean;
        if (try_probe()) {
            return t.probe;
        }
        t.dir = -t.dir;
        return try_probe() ? t.probe : settle();
    }
    // a probe: keep it if it is at least 2% faster, then go on in the same direction
    if (mean < 0.98*t.t_base) {
        t.base   = t.probe;
        t.t_base = mean;
        t.turned = true; // having improved, do not turn back
        return try_probe() ? t.probe : settle();
    }
    // the first step from no split may not cover the split's fixed cost while larger shares pay (found on the second
    // machine: 0.1 under 2%, 0.2 to 0.5 about 4% each): from 0, try one step further once before settling
    if (t.base == 0.0f && t.dir > 0.0f && !t.leaped && t.probe < 2*split_step - 1e-3f) {
        t.leaped = true;
        t.probe  = 2*split_step;
        return t.probe;
    }
    if (!t.turned) {
        t.turned = true;
        t.dir = -t.dir;
        if (try_probe()) {
            return t.probe;
        }
    }
    t.phase = 0;
    return settle();
}

void llama_kv_stream::splitter_main() {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(split_threads);
    split_pool = ggml_threadpool_new(&tpp);

    // LLAMA_KV_SPLIT_DEBUG: per 32 ubatches, the time the worker waited for Q and the time it attended, per ubatch
    const bool debug = getenv("LLAMA_KV_SPLIT_DEBUG") != nullptr;
    // LLAMA_KV_SPLIT_NOOP, for measuring only: the CPU's part is skipped and empty, so the output is WRONG; the speed
    // is the most the split could give at that share if the CPU part took no time
    const bool noop = getenv("LLAMA_KV_SPLIT_NOOP") != nullptr;
    if (noop) {
        fprintf(stderr, "llama_kv_stream: LLAMA_KV_SPLIT_NOOP: the CPU's part is skipped, results are wrong (measuring only)\n");
    }
    int64_t t_wait = 0, t_attend = 0, n_done = 0;

    const int32_t n_layer = (int32_t) layers.size();
    for (;;) {
        split_job job;
        {
            std::unique_lock<std::mutex> lock(mtx);
            split_cv.wait(lock, [&]() { return stop || !split_jobs.empty(); });
            if (stop) {
                return;
            }
            job = split_jobs.front();
            split_jobs.pop_front();
        }
        // the layers in graph order: each starts when the GPU has handed over its Q
        for (int32_t ikv = 0; ikv < n_layer; ++ikv) {
            const int64_t t0 = ggml_time_us();
            if (!wait_for([&]() { return __atomic_load_n(split_flags + ikv, __ATOMIC_ACQUIRE) >= job.tag; },
                    "Q for split attention", ikv, job.tag)) {
                return;
            }
            const int64_t t1 = ggml_time_us();
            if (noop) {
                // the CPU's part as if it took no time: empty (LSE -inf), so the merge keeps the GPU's part only
                float * lse = split_lse + (size_t) ikv*decode_max_tokens*split_n_head;
                std::fill(lse, lse + (size_t) job.n_tok*split_n_head, -INFINITY);
            } else {
                split_layer(ikv, job.n_rows, job.n_tok);
            }
            __atomic_store_n(split_flags + n_layer + ikv, job.tag, __ATOMIC_RELEASE);
            // the wait for layer 0 includes the time between ubatches
            t_wait   += ikv > 0 ? t1 - t0 : 0;
            t_attend += ggml_time_us() - t1;
        }
        if (debug && ++n_done % 32 == 0) {
            fprintf(stderr, "llama_kv_stream: split: %u rows, %u tokens: waited %.1f ms for Q (layers 1+), attended "
                    "%.1f ms (%.1f GB/s), per ubatch\n", job.n_rows, job.n_tok, t_wait/32e3, t_attend/32e3,
                    (double) job.n_rows*n_layer*(row_size(0, false) + row_size(0, true))/(t_attend/32e6)/1e9);
            t_wait = t_attend = 0;
        }
    }
}

// attention of Q over the layer's first n_rows K/V rows, in place in the host store, into the shared buffers
void llama_kv_stream::split_layer(int32_t ikv, uint32_t n_rows, uint32_t n_tok) {
    const int64_t d    = 256;
    const size_t  rows = (size_t) decode_max_tokens*split_n_head;

    ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };

    auto heads = [&](bool is_v) {
        const ggml_tensor * pool = is_v ? layers[ikv].v : layers[ikv].k;
        ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), pool->type, pool->ne[0], n_rows);
        t->data = host(ikv, is_v, 0);
        return ggml_view_3d(ctx.get(), t, d, n_rows, pool->ne[0]/d, t->nb[1], ggml_row_size(t->type, d), 0);
    };
    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d, n_tok, split_n_head);
    q->data = split_q + ikv*rows*d;

    ggml_tensor * fa = ggml_flash_attn_ext(ctx.get(), q, heads(false), heads(true), nullptr, split_desc[ikv].scale, 0.0f, 0.0f);
    fa->data = split_out + ikv*rows*d;
    ggml_flash_attn_ext_set_lse(fa, split_lse + ikv*rows);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_cplan plan = ggml_graph_plan(gf, split_threads, split_pool);
    if (plan.work_size > split_work.size()) {
        split_work.resize(plan.work_size);
    }
    plan.work_data = split_work.data();
    ggml_graph_compute(gf, &plan);
}

void llama_kv_stream::drain() {
    if (!copier.joinable()) {
        return;
    }
    wait_for([&]() {
        return n_jobs_done.load(std::memory_order_acquire) == n_jobs &&
               n_published.load(std::memory_order_acquire) == n_enqueued.load(std::memory_order_acquire);
    }, "the copy queue to drain", -1, tag);
    // after the wait: a job still running could take the new epoch for its prefetch
    epoch.fetch_add(1, std::memory_order_acq_rel);
}

int64_t llama_kv_stream::row(uint32_t cell) const {
    const int32_t slot = page_slot[cell/page_size];
    GGML_ASSERT(slot >= 0);
    return (int64_t) slot*page_size + cell % page_size;
}

void llama_kv_stream::set_input_pages(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && dst->ne[1] == (int64_t) layers.size());

    // the CPU's pages come first and are not in the table
    int32_t * data = (int32_t *) dst->data;
    for (int64_t i = 0; i < dst->ne[0]; ++i) {
        const int64_t p = i + n_split;
        GGML_ASSERT(page_slot[p] >= 0 || page_ring[p] >= 0);
        data[i] = page_slot[p] >= 0 ? page_slot[p] : -1 - page_ring[p];
    }
    for (int64_t l = 1; l < dst->ne[1]; ++l) {
        std::copy(data, data + dst->ne[0], data + l*dst->ne[0]);
    }
}

void llama_kv_stream::set_input_tag(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(ggml_nelements(dst) >= 2);
    ((int32_t *) dst->data)[0] = (int32_t) tag;
    ((int32_t *) dst->data)[1] = (int32_t) split_tag;
}

std::vector<llama_kv_stream::run> llama_kv_stream::runs(uint32_t n_kv) const {
    std::vector<run> res;
    for (uint32_t p = 0; p < std::min(n_kv/page_size, n_pages); ++p) {
        const int32_t slot = page_slot[p];
        if (slot < 0) {
            continue; // streamed, or a graph reserved before anything is resident
        }
        if (!res.empty() && res.back().page + res.back().n == p && res.back().slot + res.back().n == (uint32_t) slot) {
            res.back().n++;
        } else {
            res.push_back({ p, (uint32_t) slot, 1 });
        }
    }
    return res;
}

void llama_kv_stream::clear() {
    drain();

    page_slot.assign(n_pages, -1);
    slot_page.assign(n_slots, -1);
    page_ring.assign(n_pages, -1);
    dirty.assign(n_pages, 0);
    saved.assign(n_pages, 0);
    n_streamed = 0;
    layout_id++;
}

void llama_kv_stream::release_pool() {
    drain();
    for (uint32_t p = 0; p < n_pages; ++p) {
        if (page_slot[p] >= 0) {
            evict(p);
        }
    }
    if (backend) {
        ggml_backend_synchronize(backend);
    }
    page_ring.assign(n_pages, -1);
    n_streamed = 0;
    slots_buf.reset();
}

void llama_kv_stream::set_pool(uint32_t n_slots, uint32_t n_ring) {
    for (uint32_t p = 0; p < n_pages; ++p) {
        GGML_ASSERT(page_slot[p] < 0 && "llama_kv_stream: set_pool without release_pool");
    }
    this->n_slots = n_slots;
    slot_page.assign(n_slots, -1);
    layout_id++;

    // the slots hold the pages that stream, packed (see set_backend), or a whole layer for gathers
    if (!ring_packed()) {
        n_ring = n_pages;
    } else if (n_ring == 0) {
        n_ring = std::max<uint32_t>(1, n_pages > n_slots ? n_pages - n_slots : 1);
    }
    alloc_slots((int64_t) n_ring*page_size, false);

    LLAMA_LOG_DEBUG("%s: %u of %u pages resident, slots = %8.2f MiB\n", __func__, n_slots, n_pages,
            ggml_backend_buffer_get_size(slots_buf.get())/1024.0/1024.0);
}

void llama_kv_stream::drop_pool() {
    for (uint32_t p = 0; p < n_pages; ++p) {
        GGML_ASSERT(page_slot[p] < 0 && "llama_kv_stream: drop_pool without release_pool");
    }
    n_slots = 0;
    slot_page.clear();
    slots_buf.reset();
    layout_id++;
}

void llama_kv_stream::get_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, void * dst) {
    drain();
    if (backend) {
        ggml_backend_synchronize(backend);
    }

    const size_t row = row_size(ikv, is_v);
    ggml_tensor * pool = is_v ? layers[ikv].v : layers[ikv].k;

    for (uint32_t c = c0; c < c1; ) {
        const uint32_t page = c/page_size;
        const uint32_t n    = std::min(c1, (page + 1)*page_size) - c;
        char * out = (char *) dst + (size_t) (c - c0)*row;

        if (page_slot[page] >= 0) {
            ggml_backend_tensor_get(pool, out, (size_t) (page_slot[page]*page_size + c % page_size)*row, n*row);
        } else if (saved[page]) {
            memcpy(out, host(ikv, is_v, page) + (size_t) (c % page_size)*row, n*row);
        } else {
            memset(out, 0, n*row);
        }
        c += n;
    }
}

void llama_kv_stream::set_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, const void * src) {
    drain();
    if (backend) {
        ggml_backend_synchronize(backend);
    }

    const size_t row = row_size(ikv, is_v);
    ggml_tensor * pool = is_v ? layers[ikv].v : layers[ikv].k;

    for (uint32_t c = c0; c < c1; ) {
        const uint32_t page = c/page_size;
        const uint32_t n    = std::min(c1, (page + 1)*page_size) - c;
        const char * in = (const char *) src + (size_t) (c - c0)*row;

        if (page_slot[page] >= 0) {
            ggml_backend_tensor_set(pool, in, (size_t) (page_slot[page]*page_size + c % page_size)*row, n*row);
            dirty[page] = 1;
        } else {
            if (!saved[page]) {
                zero_host(page);
            }
            memcpy(host(ikv, is_v, page) + (size_t) (c % page_size)*row, in, n*row);
        }
        c += n;
    }
}
