/* leafv5_scan.c — fused delta-memory scan kernel, matching the CURRENT
 * LEAFv5 architecture (SOTA-upgraded): read with a SEPARATE query q.
 *
 * Per head, per token (with L2-normalized q/k/v):
 *   o_prev = S @ q                       (read PRE-update, query-based)
 *   tmp    = S @ k                       (erase projection)
 *   S     <- a*S - bf*(S@k) k^T + bw * v k^T     (a = input decay, 1 if none)
 *   S     <- StateNorm(S)                if state_norm
 *   o_new = S @ q                        (read POST-update)
 *   out   = gr * o_new + alpha * o_prev
 *
 * OPTIMIZATIONS (2026-08-10 "speedup engine" pass):
 *   * OpenMP parallel over BH — every (batch*head) stream is independent, so
 *     the scan scales across cores (num_threads = OMP_NUM_THREADS by default;
 *     leafv5_scan_q_nt takes an explicit count for benchmarks).
 *   * Algebraic fusion when StateNorm is OFF:
 *       o_new = a*o_prev + (k.q) * (bw*v - bf*tmp)          (exact identity)
 *     so the post-update matvec disappears (3 matvecs -> 2 + 1 dot).
 *   * `restrict` everywhere + `#pragma omp simd` so gcc emits AVX2/AVX-512
 *     FMA code (-march=native) for the matvecs, the outer-product update and
 *     the StateNorm reduction/scale.
 *   * Per-thread ALIGNED STACK scratch (__builtin_alloca_with_align): no
 *     malloc per call — important for the T=1 token-by-token decode path.
 *   * Fused q==k kernel kept (1 matvec, paper-exact variant).
 *
 * Layouts (row-major, contiguous):
 *   q,k,v : [BH*T*dh]   bw,bf,gr: [BH*T]   dec: [BH*T] (may be NULL)  alpha: [BH]
 *   S     : [BH*dh*dh] in/out    out: [BH*T*dh]
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* aligned stack scratch, exact size, no malloc */
#define L5_SCRATCH(TYPE, NAME, N)                                              \
    TYPE *NAME = (TYPE *)__builtin_alloca_with_align(sizeof(TYPE) * (size_t)(N), 512)

static inline float state_norm_scale(const float *restrict S, int64_t dh) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (int64_t i = 0; i < dh * dh; ++i) s += S[i] * S[i];
    return sqrtf((float)dh) / (sqrtf(s) + 1e-6f);
}

/* matvec o[i] = S[i,:] @ x   (S row-major dh x dh, x length dh) */
static inline void matvec(const float *restrict S, const float *restrict x,
                          float *restrict o, int64_t dh) {
    for (int64_t i = 0; i < dh; ++i) {
        float acc = 0.0f;
        const float *restrict row = S + i * dh;
#pragma omp simd reduction(+ : acc)
        for (int64_t j = 0; j < dh; ++j) acc += row[j] * x[j];
        o[i] = acc;
    }
}

static void scan_q_body(const float *restrict q, const float *restrict k,
                        const float *restrict v, const float *restrict bw,
                        const float *restrict bf, const float *restrict gr,
                        const float *restrict dec, const float *restrict alpha,
                        float *restrict S, float *restrict out,
                        int64_t BH, int64_t T, int64_t dh, int state_norm,
                        const float *restrict sw, const float *restrict sb,
                        int nthreads) {
#ifdef _OPENMP
    int nt = (nthreads > 0) ? nthreads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(nt) if (BH > 1)
#endif
    for (int64_t b = 0; b < BH; ++b) {
        const float *restrict qq = q + b * T * dh;
        const float *restrict kk = k + b * T * dh;
        const float *restrict vv = v + b * T * dh;
        const float *restrict bbw = bw + b * T;
        const float *restrict bbf = bf + b * T;
        const float *restrict ggr = gr + b * T;
        const float *restrict dd = dec ? dec + b * T : NULL;
        const float a0 = alpha[b];
        float *restrict SS = S + b * dh * dh;
        float *restrict oo = out + b * T * dh;
        L5_SCRATCH(float, o_prev, dh);
        L5_SCRATCH(float, o_new, dh);
        L5_SCRATCH(float, tmp, dh);
        for (int64_t t = 0; t < T; ++t) {
            const float *restrict qt = qq + t * dh;
            const float *restrict kt = kk + t * dh;
            const float *restrict vt = vv + t * dh;
            const float bw_t0 = bbw[t], bf_t = bbf[t], gr_t = ggr[t];
            const float a_t = dd ? dd[t] : 1.0f;
            matvec(SS, qt, o_prev, dh);              /* o_prev = S @ q */
            matvec(SS, kt, tmp, dh);                 /* tmp    = S @ k */
            /* novelty-gated write (Tier-1): factor = clamp(1 + w*(s - b)) with
             * s = ||v - tmp||/sqrt(dh); NULL sw => off (identity) */
            float bw_t = bw_t0;
            if (sw != NULL) {
                float s2 = 0.0f;
                for (int64_t j = 0; j < dh; ++j) {
                    const float d = vt[j] - tmp[j];
                    s2 += d * d;
                }
                const float s = sqrtf(s2) / sqrtf((float)dh);
                float fac = 1.0f + sw[b] * (s - sb[b]);
                if (fac < 0.0f) fac = 0.0f; else if (fac > 2.0f) fac = 2.0f;
                bw_t *= fac;
            }
            if (state_norm) {
                /* S <- a*S + kt^T * (bw*v - bf*tmp)  (row update) */
                for (int64_t i = 0; i < dh; ++i) {
                    const float coef = bw_t * vt[i] - bf_t * tmp[i];
                    float *restrict row = SS + i * dh;
#pragma omp simd
                    for (int64_t j = 0; j < dh; ++j)
                        row[j] = a_t * row[j] + kt[j] * coef;
                }
                const float sc = state_norm_scale(SS, dh);
#pragma omp simd
                for (int64_t i = 0; i < dh * dh; ++i) SS[i] *= sc;
                matvec(SS, qt, o_new, dh);           /* o_new = S @ q (post-norm) */
            } else {
                /* fused identity (exact): o_new = a*o_prev + (k.q)*(bw*v - bf*tmp) */
                float kdotq = 0.0f;
#pragma omp simd reduction(+ : kdotq)
                for (int64_t j = 0; j < dh; ++j) kdotq += kt[j] * qt[j];
                for (int64_t i = 0; i < dh; ++i) {
                    const float coef = bw_t * vt[i] - bf_t * tmp[i];
                    o_new[i] = a_t * o_prev[i] + kdotq * coef;
                    float *restrict row = SS + i * dh;
#pragma omp simd
                    for (int64_t j = 0; j < dh; ++j)
                        row[j] = a_t * row[j] + kt[j] * coef;
                }
            }
            for (int64_t i = 0; i < dh; ++i)
                oo[t * dh + i] = gr_t * o_new[i] + a0 * o_prev[i];
        }
    }
}

/* Public API: auto threads (OMP_NUM_THREADS) — signature unchanged from the
 * original kernel, so existing ctypes wrappers keep working. */
void leafv5_scan_q(const float *q, const float *k, const float *v,
                   const float *bw, const float *bf, const float *gr,
                   const float *dec, const float *alpha,
                   float *S, float *out,
                   int64_t BH, int64_t T, int64_t dh, int state_norm) {
    scan_q_body(q, k, v, bw, bf, gr, dec, alpha, S, out,
                BH, T, dh, state_norm, NULL, NULL, 0);
}

/* Explicit thread count (0 = auto).  For benchmarks / scaling studies. */
void leafv5_scan_q_nt(const float *q, const float *k, const float *v,
                      const float *bw, const float *bf, const float *gr,
                      const float *dec, const float *alpha,
                      float *S, float *out,
                      int64_t BH, int64_t T, int64_t dh, int state_norm,
                      int nthreads) {
    scan_q_body(q, k, v, bw, bf, gr, dec, alpha, S, out,
                BH, T, dh, state_norm, NULL, NULL, nthreads);
}

/* Novelty-gated variant: sw/sb are per-head [BH] factors (NULL -> off). */
void leafv5_scan_q_s(const float *q, const float *k, const float *v,
                     const float *bw, const float *bf, const float *gr,
                     const float *dec, const float *alpha,
                     float *S, float *out,
                     int64_t BH, int64_t T, int64_t dh, int state_norm,
                     const float *sw, const float *sb) {
    scan_q_body(q, k, v, bw, bf, gr, dec, alpha, S, out,
                BH, T, dh, state_norm, sw, sb, 0);
}

void leafv5_scan_q_s_nt(const float *q, const float *k, const float *v,
                        const float *bw, const float *bf, const float *gr,
                        const float *dec, const float *alpha,
                        float *S, float *out,
                        int64_t BH, int64_t T, int64_t dh, int state_norm,
                        const float *sw, const float *sb, int nthreads) {
    scan_q_body(q, k, v, bw, bf, gr, dec, alpha, S, out,
                BH, T, dh, state_norm, sw, sb, nthreads);
}

/* ---- fused q==k kernel (paper-exact variant; no decay/norm) ---- */
static void scan_fused_body(const float *restrict k, const float *restrict v,
                            const float *restrict bw, const float *restrict bf,
                            const float *restrict gr, const float *restrict alpha,
                            float *restrict S, float *restrict out,
                            int64_t BH, int64_t T, int64_t dh, int nthreads) {
#ifdef _OPENMP
    int nt = (nthreads > 0) ? nthreads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(nt) if (BH > 1)
#endif
    for (int64_t b = 0; b < BH; ++b) {
        const float *restrict kk = k + b * T * dh;
        const float *restrict vv = v + b * T * dh;
        const float *restrict bbw = bw + b * T;
        const float *restrict bbf = bf + b * T;
        const float *restrict ggr = gr + b * T;
        const float a = alpha[b];
        float *restrict SS = S + b * dh * dh;
        float *restrict oo = out + b * T * dh;
        L5_SCRATCH(float, o_prev, dh);
        for (int64_t t = 0; t < T; ++t) {
            const float *restrict kt = kk + t * dh;
            const float *restrict vt = vv + t * dh;
            const float bw_t = bbw[t], bf_t = bbf[t], gr_t = ggr[t];
            matvec(SS, kt, o_prev, dh);              /* o_prev = S @ k */
            for (int64_t i = 0; i < dh; ++i) {
                const float coef = bw_t * vt[i] - bf_t * o_prev[i];
                float *restrict row = SS + i * dh;
#pragma omp simd
                for (int64_t j = 0; j < dh; ++j) row[j] += kt[j] * coef;
                oo[t * dh + i] = gr_t * (o_prev[i] + coef) + a * o_prev[i];
            }
        }
    }
}

void leafv5_scan_fused(const float *k, const float *v, const float *bw,
                       const float *bf, const float *gr, const float *alpha,
                       float *S, float *out,
                       int64_t BH, int64_t T, int64_t dh) {
    scan_fused_body(k, v, bw, bf, gr, alpha, S, out, BH, T, dh, 0);
}

void leafv5_scan_fused_nt(const float *k, const float *v, const float *bw,
                          const float *bf, const float *gr, const float *alpha,
                          float *S, float *out,
                          int64_t BH, int64_t T, int64_t dh, int nthreads) {
    scan_fused_body(k, v, bw, bf, gr, alpha, S, out, BH, T, dh, nthreads);
}

/* Version + compile-time capability string (for the README / bench). */
const char *leafv5_scan_version(void) {
    return "leafv5_scan v2 (2026-08-10): OpenMP + SIMD + fused no-norm path";
}
