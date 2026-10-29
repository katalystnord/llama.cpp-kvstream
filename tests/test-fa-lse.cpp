// test-fa-lse.cpp: validate the optional log-sum-exp output of the CPU flash attention kernels
// (op_params[GGML_FA_PARAM_LSE]), which lets attention over disjoint parts of K/V be merged exactly.
//
// test-backend-ops uses the CPU backend as its reference, so it cannot check the CPU kernels against themselves. This
// test checks the property the split relies on, for every CPU path (one row with q8_0 and f16 K/V, which take the
// per-head and the split-KV paths, several rows, which take the tiled and the chunked paths, 1 and 12 threads):
//   1. attention over K/V[0, n1) and K/V[n1, n), merged by their LSE, equals attention over K/V[0, n): output and LSE;
//   2. the LSE equals log(sum(exp(scale * q.k + mask))) computed in double precision from the same (f16) inputs;
//   3. a row whose part is fully masked gets -inf and weight 0 in the merge;
//   4. on a GPU backend, if there is one and it takes the case: the GPU's output and LSE match the CPU's, and its parts
//      merge to its full result, and its first part merged with the CPU's second part (the split) equals the CPU's
//      full result (on an RDNA4 card the q8_0 decode cases run the paged decode kernel). Cases whose GPU
//      kernel does not write the LSE (the vec and tile kernels) are reported and skipped.
// With an f16 V cache the kernel accumulates the output in f16 (VKQ16), so the full run and the merged parts round
// differently by about 1e-3; there the merge must add no error beyond the kernel's own: both are compared with a
// double-precision output, and the merged error may be at most twice the full run's.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Deterministic LCG mapped to [-1, 1]
static uint64_t g_rng = 0x2468ace0ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

struct fa_case {
    const char * name;
    ggml_type    type_kv;
    int64_t      D;
    int64_t      n_head_kv;
    int64_t      gqa;
    int64_t      n_rows;
    int64_t      n_kv;
    int64_t      n1;          // split point
    bool         masked_rows; // row r sees only K/V[0, n_kv - n_masked(r)), and row 0 nothing beyond n1 / 2
    int          n_threads;
};

static ggml_backend_t g_gpu = nullptr;

// Runs attention of q over K/V rows [c0, c1) with the matching mask columns; returns output [n_rows][n_head][D] and
// LSE [n_rows][n_head].
static void run_fa(const fa_case & c, const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
        const std::vector<float> & mask, int64_t c0, int64_t c1, float scale, std::vector<float> & out, std::vector<float> & lse,
        ggml_backend_t gpu = nullptr) {
    const int64_t n_head = c.n_head_kv*c.gqa;
    const int64_t n      = c1 - c0;

    ggml_init_params ip = { 512ull*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.n_rows, n_head, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, c.type_kv, c.D, n, c.n_head_kv, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, c.type_kv, c.D, n, c.n_head_kv, 1);
    ggml_tensor * tm = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, c.n_rows, 1, 1);

    memcpy(tq->data, q.data(), q.size()*sizeof(float));
    // K/V are [n_kv_total][head][D] in k/v; copy rows [c0, c1) of each head and convert or quantize them
    std::vector<float> kp(c.D*n*c.n_head_kv), vp(c.D*n*c.n_head_kv);
    for (int64_t h = 0; h < c.n_head_kv; ++h) {
        for (int64_t i = 0; i < n; ++i) {
            memcpy(&kp[(h*n + i)*c.D], &k[((c0 + i)*c.n_head_kv + h)*c.D], c.D*sizeof(float));
            memcpy(&vp[(h*n + i)*c.D], &v[((c0 + i)*c.n_head_kv + h)*c.D], c.D*sizeof(float));
        }
    }
    if (c.type_kv == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(kp.data(), (ggml_fp16_t *) tk->data, (int64_t) kp.size());
        ggml_fp32_to_fp16_row(vp.data(), (ggml_fp16_t *) tv->data, (int64_t) vp.size());
    } else {
        ggml_quantize_chunk(c.type_kv, kp.data(), tk->data, 0, n*c.n_head_kv, c.D, nullptr);
        ggml_quantize_chunk(c.type_kv, vp.data(), tv->data, 0, n*c.n_head_kv, c.D, nullptr);
    }
    std::vector<float> mp(n*c.n_rows);
    for (int64_t r = 0; r < c.n_rows; ++r) {
        memcpy(&mp[r*n], &mask[r*c.n_kv + c0], n*sizeof(float));
    }
    ggml_fp32_to_fp16_row(mp.data(), (ggml_fp16_t *) tm->data, (int64_t) mp.size());

    lse.assign(c.n_rows*n_head, NAN);
    if (gpu) {
        // the same inputs in the GPU's memory, and the LSE in a GPU tensor (the CUDA/HIP kernels take a device pointer)
        ggml_init_params gp = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * src[4] = { tq, tk, tv, tm };
        ggml_tensor * dev[4];
        for (int i = 0; i < 4; ++i) {
            dev[i] = ggml_dup_tensor(gctx, src[i]);
        }
        ggml_tensor * tl = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, c.n_rows*n_head);
        ggml_tensor * fa = ggml_flash_attn_ext(gctx, dev[0], dev[1], dev[2], dev[3], scale, 0.0f, 0.0f);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
        for (int i = 0; i < 4; ++i) {
            ggml_backend_tensor_set(dev[i], src[i]->data, 0, ggml_nbytes(src[i]));
        }
        // NaN until written: the vec and tile kernels take the parameter but do not write it
        ggml_backend_tensor_set(tl, lse.data(), 0, lse.size()*sizeof(float));
        void * p = tl->data;
        memcpy(fa->op_params + GGML_FA_PARAM_LSE, &p, sizeof(p));
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, fa);
        if (!ggml_backend_supports_op(gpu, fa) || ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) {
            out.clear();
        } else {
            out.resize(c.D*n_head*c.n_rows);
            ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
            ggml_backend_tensor_get(tl, lse.data(), 0, lse.size()*sizeof(float));
        }
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
        ggml_free(ctx);
        return;
    }

    ggml_tensor * fa = ggml_flash_attn_ext(ctx, tq, tk, tv, tm, scale, 0.0f, 0.0f);
    float * p = lse.data();
    memcpy(fa->op_params + GGML_FA_PARAM_LSE, &p, sizeof(p));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);
    ggml_graph_compute_with_ctx(ctx, gf, c.n_threads);

    // dst is [D, n_head, n_rows, 1]
    out.assign((const float *) fa->data, (const float *) fa->data + c.D*n_head*c.n_rows);
    ggml_free(ctx);
}

static bool close(double a, double b, double atol, double rtol) {
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;
    }
    return std::fabs(a - b) <= atol + rtol*std::fabs(b);
}

static int check(const fa_case & c) {
    const int64_t n_head = c.n_head_kv*c.gqa;
    const float   scale  = 1.0f/sqrtf((float) c.D);

    std::vector<float> q(c.D*c.n_rows*n_head), k(c.n_kv*c.n_head_kv*c.D), v(c.n_kv*c.n_head_kv*c.D), mask(c.n_rows*c.n_kv, 0.0f);
    for (auto & x : q) { x = 2.0f*frand(); }
    for (auto & x : k) { x = 2.0f*frand(); }
    for (auto & x : v) { x = frand(); }
    if (c.masked_rows) {
        for (int64_t r = 0; r < c.n_rows; ++r) {
            // row 0 sees only the first half of part 1 (so part 2 is fully masked for it), the others lose a tail
            const int64_t visible = r == 0 ? c.n1/2 : c.n_kv - 37*r;
            for (int64_t i = visible; i < c.n_kv; ++i) {
                mask[r*c.n_kv + i] = -INFINITY;
            }
        }
    }

    std::vector<float> o_all, l_all, o1, l1, o2, l2;
    run_fa(c, q, k, v, mask, 0, c.n_kv, scale, o_all, l_all);
    run_fa(c, q, k, v, mask, 0, c.n1, scale, o1, l1);
    run_fa(c, q, k, v, mask, c.n1, c.n_kv, scale, o2, l2);

    int bad = 0;
    double max_do = 0.0, max_dl = 0.0, max_dref = 0.0, max_e_full = 0.0, max_e_merged = 0.0;
    for (int64_t r = 0; r < c.n_rows; ++r) {
        for (int64_t h = 0; h < n_head; ++h) {
            const int64_t il = r*n_head + h;
            // 1. merge the parts by LSE
            const double a = l1[il], b = l2[il];
            const double m = std::fmax(a, b);
            const double wa = std::isinf(a) ? 0.0 : std::exp(a - m), wb = std::isinf(b) ? 0.0 : std::exp(b - m);
            const double l_merged = m + std::log(wa + wb);
            if (std::isnan(l_all[il]) || !close(l_merged, l_all[il], 1e-4, 1e-5)) {
                if (bad++ < 5) {
                    fprintf(stderr, "  %s: row %lld head %lld: lse parts %g, %g -> %g, full %g\n", c.name, (long long) r,
                            (long long) h, a, b, l_merged, l_all[il]);
                }
            }
            if (!std::isinf(l_all[il])) {
                max_dl = std::fmax(max_dl, std::fabs(l_merged - l_all[il]));
            }
            const bool f16_acc = c.type_kv == GGML_TYPE_F16;
            for (int64_t d = 0; d < c.D; ++d) {
                const int64_t io = (r*n_head + h)*c.D + d;
                const double merged = (wa*o1[io] + wb*o2[io])/(wa + wb);
                max_do = std::fmax(max_do, std::fabs(merged - o_all[io]));
                if (!f16_acc && !close(merged, o_all[io], 1e-4, 1e-3)) {
                    if (bad++ < 5) {
                        fprintf(stderr, "  %s: row %lld head %lld dim %lld: merged %g, full %g\n", c.name, (long long) r,
                                (long long) h, (long long) d, merged, o_all[io]);
                    }
                }
            }
            if (f16_acc) {
                // double-precision output from the f16 K/V the kernel reads
                const int64_t hk = h / c.gqa;
                std::vector<double> sc(c.n_kv);
                double mx = -INFINITY;
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    double dot = 0.0;
                    for (int64_t d = 0; d < c.D; ++d) {
                        dot += (double) q[(h*c.n_rows + r)*c.D + d]*ggml_fp16_to_fp32(ggml_fp32_to_fp16(k[(i*c.n_head_kv + hk)*c.D + d]));
                    }
                    sc[i] = scale*dot + mask[r*c.n_kv + i];
                    mx = std::fmax(mx, sc[i]);
                }
                std::vector<double> ref(c.D, 0.0);
                double sum = 0.0;
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    const double w = std::isinf(sc[i]) ? 0.0 : std::exp(sc[i] - mx);
                    sum += w;
                    for (int64_t d = 0; d < c.D; ++d) {
                        ref[d] += w*ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[(i*c.n_head_kv + hk)*c.D + d]));
                    }
                }
                double e_full = 0.0, e_merged = 0.0;
                for (int64_t d = 0; d < c.D; ++d) {
                    const int64_t io = (r*n_head + h)*c.D + d;
                    e_full   = std::fmax(e_full,   std::fabs(o_all[io] - ref[d]/sum));
                    e_merged = std::fmax(e_merged, std::fabs((wa*o1[io] + wb*o2[io])/(wa + wb) - ref[d]/sum));
                }
                max_e_full = std::fmax(max_e_full, e_full);
                max_e_merged = std::fmax(max_e_merged, e_merged);
            }
            // 2. the LSE against double precision, from the inputs as the kernel sees K (f16-rounded; q8_0 is checked
            //    against f16-rounded K too, with a looser bound for its quantization)
            if (c.type_kv == GGML_TYPE_F16 || c.type_kv == GGML_TYPE_Q8_0) {
                const int64_t hk = h / c.gqa;
                double mx = -INFINITY;
                std::vector<double> s(c.n_kv);
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    double dot = 0.0;
                    for (int64_t d = 0; d < c.D; ++d) {
                        const float kf = ggml_fp16_to_fp32(ggml_fp32_to_fp16(k[(i*c.n_head_kv + hk)*c.D + d]));
                        dot += (double) q[(h*c.n_rows + r)*c.D + d]*kf;
                    }
                    s[i] = scale*dot + mask[r*c.n_kv + i];
                    mx = std::fmax(mx, s[i]);
                }
                double sum = 0.0;
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    sum += std::isinf(s[i]) ? 0.0 : std::exp(s[i] - mx);
                }
                const double ref = mx + std::log(sum);
                const double tol = c.type_kv == GGML_TYPE_F16 ? 2e-2 : 1e-1;
                max_dref = std::fmax(max_dref, std::fabs(ref - l_all[il]));
                if (!close(l_all[il], ref, tol, 0.0)) {
                    if (bad++ < 5) {
                        fprintf(stderr, "  %s: row %lld head %lld: lse %g, reference %g\n", c.name, (long long) r,
                                (long long) h, l_all[il], ref);
                    }
                }
            }
        }
    }
    // 3. the fully masked part of row 0 must be -inf
    if (c.masked_rows) {
        for (int64_t h = 0; h < n_head; ++h) {
            if (!(std::isinf(l2[h]) && l2[h] < 0)) {
                if (bad++ < 5) {
                    fprintf(stderr, "  %s: head %lld: fully masked part has lse %g, expected -inf\n", c.name, (long long) h, l2[h]);
                }
            }
        }
    }
    // 4. the GPU against the CPU, and its own parts merged
    if (g_gpu) {
        std::vector<float> og, lg, og1, lg1, og2, lg2;
        run_fa(c, q, k, v, mask, 0, c.n_kv, scale, og, lg, g_gpu);
        if (og.empty()) {
            printf("%-28s gpu: does not take this case, skipped\n", c.name);
        } else if (std::isnan(lg[0])) {
            printf("%-28s gpu: its kernel for this case does not write the LSE (vec/tile), skipped\n", c.name);
        } else {
            run_fa(c, q, k, v, mask, 0, c.n1, scale, og1, lg1, g_gpu);
            run_fa(c, q, k, v, mask, c.n1, c.n_kv, scale, og2, lg2, g_gpu);
            double d_out = 0.0, d_lse = 0.0, d_merge = 0.0, d_cross = 0.0;
            int gbad = 0;
            for (int64_t il = 0; il < c.n_rows*n_head; ++il) {
                const bool lse_ok = (std::isinf(l_all[il]) && lg[il] == l_all[il]) ||
                                    (!std::isinf(l_all[il]) && std::fabs(lg[il] - l_all[il]) <= 1e-2);
                if (!lse_ok && gbad++ < 5) {
                    fprintf(stderr, "  %s gpu: row/head %lld: lse %g, cpu %g\n", c.name, (long long) il, lg[il], l_all[il]);
                }
                if (!std::isinf(l_all[il])) {
                    d_lse = std::fmax(d_lse, std::fabs(lg[il] - l_all[il]));
                }
                const double a = og1.empty() ? NAN : lg1[il], b = og2.empty() ? NAN : lg2[il];
                const double m = std::fmax(a, b);
                const double wa = std::isinf(a) ? 0.0 : std::exp(a - m), wb = std::isinf(b) ? 0.0 : std::exp(b - m);
                // the split itself: the GPU's first part merged with the CPU's second part, against the CPU's full result
                const double xb = l2[il], xm = std::fmax(a, xb);
                const double xa_w = std::isinf(a) ? 0.0 : std::exp(a - xm), xb_w = std::isinf(xb) ? 0.0 : std::exp(xb - xm);
                for (int64_t d = 0; d < c.D; ++d) {
                    const int64_t io = il*c.D + d;
                    d_out = std::fmax(d_out, std::fabs(og[io] - o_all[io]));
                    if (!og1.empty()) {
                        d_cross = std::fmax(d_cross, std::fabs((xa_w*og1[io] + xb_w*o2[io])/(xa_w + xb_w) - o_all[io]));
                    }
                    if (!og1.empty() && !og2.empty()) {
                        d_merge = std::fmax(d_merge, std::fabs((wa*og1[io] + wb*og2[io])/(wa + wb) - og[io]));
                    }
                }
            }
            if (d_out > 1e-2) {
                fprintf(stderr, "  %s gpu: output differs from the cpu by %g\n", c.name, d_out);
                gbad++;
            }
            if (!(d_cross <= 1e-2)) {
                fprintf(stderr, "  %s gpu: gpu part 1 + cpu part 2 against the cpu's full result: %g\n", c.name, d_cross);
                gbad++;
            }
            if (og1.empty() || og2.empty() || !(d_merge <= 5e-3)) {
                fprintf(stderr, "  %s gpu: parts %s, merged against full %g\n", c.name,
                        og1.empty() || og2.empty() ? "not taken" : "taken", d_merge);
                gbad++;
            }
            printf("%-28s gpu: %s  (|out - cpu| %.2e, |lse - cpu| %.2e, |merged - full| %.2e, gpu+cpu split %.2e)\n",
                    c.name, gbad ? "FAIL" : "ok", d_out, d_lse, d_merge, d_cross);
            bad += gbad;
        }
    }
    if (max_e_merged > 2.0*max_e_full + 1e-5) {
        fprintf(stderr, "  %s: merged output error %g against double precision, full run %g\n", c.name, max_e_merged, max_e_full);
        bad++;
    }
    printf("%-28s %s  (max |merged - full|: out %.2e, lse %.2e; |lse - double ref| %.2e", c.name, bad ? "FAIL" : "ok",
            max_do, max_dl, max_dref);
    if (c.type_kv == GGML_TYPE_F16) {
        printf("; out vs double: full %.2e, merged %.2e", max_e_full, max_e_merged);
    }
    printf(")\n");
    return bad;
}

int main(void) {
    ggml_backend_load_all();
    g_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    printf("GPU backend: %s\n", g_gpu ? ggml_backend_name(g_gpu) : "none (CPU checks only)");
    const fa_case cases[] = {
        // name                          type            D   kvh gqa rows  n_kv    n1  masked thr
        { "q8_0 decode, 12 threads",     GGML_TYPE_Q8_0, 256, 4, 6,  1,    3000,  1234, false, 12 },
        { "q8_0 decode, 1 thread",       GGML_TYPE_Q8_0, 256, 4, 6,  1,    3000,  1234, false,  1 },
        { "f16 decode split-KV",         GGML_TYPE_F16,  256, 4, 6,  1,    3000,  1000, false, 12 },
        { "f16 decode masked",           GGML_TYPE_F16,  256, 4, 6,  1,    3000,  1000, true,  12 },
        { "f16 7 rows masked",           GGML_TYPE_F16,  256, 4, 6,  7,    2000,   700, true,  12 },
        { "f16 7 rows, 1 thread",        GGML_TYPE_F16,  256, 4, 6,  7,    2000,   700, true,   1 },
        { "q8_0 5 rows masked",          GGML_TYPE_Q8_0, 256, 4, 6,  5,    2000,   900, true,  12 },
        { "f16 64 rows",                 GGML_TYPE_F16,  128, 2, 4, 64,    1024,   512, false, 12 },
        { "q8_0 32 rows, D=128",         GGML_TYPE_Q8_0, 128, 8, 4, 32,    1500,   333, true,  12 },
    };
    int bad = 0;
    for (const auto & c : cases) {
        bad += check(c) ? 1 : 0;
    }
    printf("%s\n", bad ? "FAILED" : "all passed");
    if (g_gpu) {
        ggml_backend_free(g_gpu);
    }
    return bad ? 1 : 0;
}
