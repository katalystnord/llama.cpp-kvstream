// CPU half of split attention, alone (2026-10-29): the rate of ggml's CPU flash attention as the stream's split worker
// calls it (one decode query, 24 query heads over 4 KV heads of 256, q8_0 K and V, log-sum-exp output), per layout and
// memory of the K/V rows, number of calls and threads. The split ran at about 9.5 GB/s in the server; this separates the
// candidate causes:
//   layout:  interleaved = the host store's rows, the 4 KV heads side by side in each row (nb1 = whole row);
//            per-head    = each KV head's rows as one block (the layout of test-backend-ops' perf cases);
//   memory:  malloc, or the GPU backend's pinned host buffer (what the host store is);
//   calls:   16 = one call per attention layer over its own rows, as the worker does; 1 = one call over all of them.
// Build: kvstream/bench/build-fa-cpu-layout.sh. Usage: fa-cpu-layout [rows per layer, default 17664] [threads...]
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t D = 256, N_HEAD_KV = 4, N_HEAD = 24, N_LAYER = 16;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// one attention call over n_rows rows starting at k/v, like llama_kv_stream::split_layer
static void attend(ggml_threadpool * tp, int nth, std::vector<uint8_t> & work, const float * q, char * k, char * v,
        int64_t n_rows, bool interleaved, float * out, float * lse) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const size_t row_head = ggml_row_size(GGML_TYPE_Q8_0, D);
    auto heads = [&](char * data) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, n_rows, N_HEAD_KV);
        t->data = data;
        if (interleaved) {
            t->nb[1] = N_HEAD_KV*row_head;
            t->nb[2] = row_head;
        }
        return t;
    };
    ggml_tensor * tq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, N_HEAD);
    tq->data = (void *) q;
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, tq, heads(k), heads(v), nullptr, 1.0f/16, 0.0f, 0.0f);
    fa->data = out;
    ggml_flash_attn_ext_set_lse(fa, lse);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);
    ggml_cplan plan = ggml_graph_plan(gf, nth, tp);
    if (plan.work_size > work.size()) {
        work.resize(plan.work_size);
    }
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const int64_t rows = argc > 1 ? atoll(argv[1]) : 17664;
    std::vector<int> threads;
    for (int i = 2; i < argc; ++i) {
        threads.push_back(atoi(argv[i]));
    }
    if (threads.empty()) {
        threads = { 8, 12, 16, 24 };
    }
    ggml_backend_load_all();

    const size_t row_head  = ggml_row_size(GGML_TYPE_Q8_0, D);
    const size_t layer_kv  = (size_t) rows*N_HEAD_KV*row_head; // bytes of K (or V) per layer
    const size_t total     = 2*layer_kv*N_LAYER;

    // one layer of quantized random rows, copied to every layer (the values do not matter for speed, NaNs would)
    std::vector<float> f((size_t) rows*N_HEAD_KV*D);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & x : f) {
        x = nd(rng);
    }
    std::vector<char> layer(layer_kv);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), layer.data(), 0, rows*N_HEAD_KV, D, nullptr);

    // memory: malloc, and the GPU backend's pinned host buffer if there is one
    struct mem { std::string name; char * base; };
    std::vector<mem> mems;
    std::vector<char> plain(total);
    mems.push_back({ "malloc", plain.data() });
    ggml_backend_buffer_t pinned = nullptr;
    if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        if (ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(gpu)) {
            pinned = ggml_backend_buft_alloc_buffer(buft, total);
            if (pinned) {
                mems.push_back({ std::string("pinned (") + ggml_backend_buft_name(buft) + ")", (char *) ggml_backend_buffer_get_base(pinned) });
            }
        }
    }
    for (auto & m : mems) {
        for (int64_t l = 0; l < 2*N_LAYER; ++l) {
            memcpy(m.base + l*layer_kv, layer.data(), layer_kv);
        }
    }

    std::vector<float> q(D*N_HEAD);
    for (auto & x : q) {
        x = nd(rng);
    }
    std::vector<float> out(D*N_HEAD), lse(N_HEAD);
    std::vector<uint8_t> work;

    // correctness: the same K/V in both layouts must give the same result (the interleaved layout takes the path whose
    // work items cover every KV head, the per-head layout the one-KV-head path)
    {
        std::vector<char> kh(layer_kv), vh(layer_kv);
        const char * vsrc = layer.data(); // V = K is fine for the check
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t h = 0; h < N_HEAD_KV; ++h) {
                memcpy(kh.data() + (h*rows + r)*row_head, layer.data() + (r*N_HEAD_KV + h)*row_head, row_head);
                memcpy(vh.data() + (h*rows + r)*row_head, vsrc + (r*N_HEAD_KV + h)*row_head, row_head);
            }
        }
        std::vector<float> o1(D*N_HEAD), o2(D*N_HEAD), l1(N_HEAD), l2(N_HEAD);
        ggml_threadpool_params tpp = ggml_threadpool_params_default(8);
        ggml_threadpool * tp = ggml_threadpool_new(&tpp);
        attend(tp, 8, work, q.data(), layer.data(), layer.data(), rows, true, o1.data(), l1.data());
        attend(tp, 8, work, q.data(), kh.data(), vh.data(), rows, false, o2.data(), l2.data());
        ggml_threadpool_free(tp);
        double dout = 0, dlse = 0;
        for (size_t i = 0; i < o1.size(); ++i) dout = std::max(dout, (double) fabsf(o1[i] - o2[i]));
        for (size_t i = 0; i < l1.size(); ++i) dlse = std::max(dlse, (double) fabsf(l1[i] - l2[i]));
        printf("check: interleaved vs per-head layout, max |diff| out %.3g, lse %.3g: %s\n", dout, dlse,
                dout < 1e-4 && dlse < 1e-4 ? "ok" : "FAIL");
    }

    printf("%lld rows per layer, %lld layers, %.1f MB of K/V per pass\n", (long long) rows, (long long) N_LAYER, total/1e6);
    printf("%-34s %-11s %5s %7s %10s %8s\n", "memory", "layout", "calls", "threads", "ms/pass", "GB/s");
    for (auto & m : mems) {
        for (bool interleaved : { true, false }) {
            for (int calls : { 16, 1 }) {
                for (int nth : threads) {
                    ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
                    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
                    const int reps = 8;
                    double best = 1e9;
                    for (int r = 0; r < reps; ++r) {
                        const double t0 = now();
                        if (calls == 16) {
                            for (int64_t l = 0; l < N_LAYER; ++l) {
                                attend(tp, nth, work, q.data(), m.base + 2*l*layer_kv, m.base + (2*l + 1)*layer_kv,
                                        rows, interleaved, out.data(), lse.data());
                            }
                        } else {
                            // all K first, then all V: one call over N_LAYER*rows rows
                            attend(tp, nth, work, q.data(), m.base, m.base + N_LAYER*layer_kv, rows*N_LAYER,
                                    interleaved, out.data(), lse.data());
                        }
                        best = std::min(best, now() - t0);
                    }
                    printf("%-34s %-11s %5d %7d %10.2f %8.2f\n", m.name.c_str(), interleaved ? "interleaved" : "per-head",
                            calls, nth, best*1e3, total/best/1e9);
                    fflush(stdout);
                    ggml_threadpool_free(tp);
                }
            }
        }
    }
    if (pinned) {
        ggml_backend_buffer_free(pinned);
    }
    return 0;
}
