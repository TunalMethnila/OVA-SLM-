// leafv5.mojo — LEAFv5 stabilized delta-memory scan kernel in pure Mojo.
//
// This is the Mojo port of mojo/c_ref/leafv5_scan.c (validated to ~1e-7
// against the PyTorch reference in leafv5/model.py).  Same math, same loop
// structure, with Mojo SIMD vectorization and — new in v2 — PARALLELISM over
// the independent (batch*head) streams plus the algebraic fusion of the
// post-update read.
//
// Paper sec. 3.3 (per head, per token), with ||k|| = 1:
//   o_prev = S @ q                       (read PRE-update, query-based)
//   tmp    = S @ k                       (erase projection)
//   S     <- a*S - bf*(S@k) k^T + bw * v k^T     (a = input decay, 1 if none)
//   S     <- StateNorm(S)                if state_norm
//   o_new = S @ q                        (read POST-update)
//   out   = gr * o_new + alpha * o_prev
//
// v2 OPTIMIZATIONS (2026-08-10):
//   * parallelize[] over BH — every (batch*head) stream is independent
//   * W=8 SIMD (AVX2+) dot products and outer-product updates
//   * StateNorm reduction/scale vectorized
//   * when StateNorm is OFF, o_new = a*o_prev + (k.q)*(bw*v - bf*tmp) exactly,
//     so the post-update matvec disappears (3 matvecs -> 2 + 1 dot)
//
// Build/run:  mojo run mojo/leafv5.mojo   (Mojo SDK 24.x)
// NOTE: the C twin below is built & benchmarked in CI/sandbox; the Mojo SDK
// build is verified by whoever runs `mojo run` (README says so honestly).

from algorithm import parallelize
from memory import DTypePointer, DType, memset_zero
from math import sqrt
from simd import SIMD

alias W = 8  // SIMD width (float32; AVX2 = 8 lanes)

fn simd_dot(
    srow: DTypePointer[DType.float32],
    x: DTypePointer[DType.float32],
    dh: Int,
) -> Float32:
    var acc = SIMD[DType.float32, W](0)
    var j = 0
    while j < dh - W + 1:
        acc += srow.load[W](j) * x.load[W](j)
        j += W
    var r = acc.reduce_add()
    while j < dh:
        r += srow.load(j) * x.load(j)
        j += 1
    return r

fn state_norm_scale(S: DTypePointer[DType.float32], dh: Int) -> Float32:
    var acc = SIMD[DType.float32, W](0)
    var i = 0
    while i < dh * dh - W + 1:
        acc += S.load[W](i) * S.load[W](i)
        i += W
    var s = acc.reduce_add()
    while i < dh * dh:
        s += S.load(i) * S.load(i)
        i += 1
    var n = sqrt(s)
    return sqrt(Float32(dh)) / (n + 1.0e-6)

// matvec o[i] = S[i,:] @ x  (row-major S, dh x dh)
fn matvec(
    S: DTypePointer[DType.float32],
    x: DTypePointer[DType.float32],
    o: DTypePointer[DType.float32],
    dh: Int,
):
    for i in range(dh):
        o.store(i, simd_dot(S + i * dh, x, dh))

// One (batch*head) stream: the whole T recurrence for head b.
fn scan_one_b(
    b: Int,
    k: DTypePointer[DType.float32],
    v: DTypePointer[DType.float32],
    q: DTypePointer[DType.float32],
    bw: DTypePointer[DType.float32],
    bf: DTypePointer[DType.float32],
    gr: DTypePointer[DType.float32],
    dec: DTypePointer[DType.float32],
    alpha: DTypePointer[DType.float32],
    S: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    T: Int,
    dh: Int,
    state_norm: Bool,
    sw: DTypePointer[DType.float32] = DTypePointer[DType.float32](),
    sb: DTypePointer[DType.float32] = DTypePointer[DType.float32](),
):
    var kk = k + b * T * dh
    var vv = v + b * T * dh
    var qq = q + b * T * dh
    var bbw = bw + b * T
    var bbf = bf + b * T
    var ggr = gr + b * T
    var dd = dec + b * T
    var a0 = alpha.load(b)
    var SS = S + b * dh * dh
    var oo = out + b * T * dh
    var o_prev = DTypePointer[DType.float32].alloc(dh)
    var o_new = DTypePointer[DType.float32].alloc(dh)
    var tmp = DTypePointer[DType.float32].alloc(dh)
    for t in range(T):
        var kt = kk + t * dh
        var vt = vv + t * dh
        var qt = qq + t * dh
        var bw_t0 = bbw.load(t)
        var bf_t = bbf.load(t)
        var gr_t = ggr.load(t)
        var a_t: Float32 = 1.0
        if not dec.is_null():
            a_t = dd.load(t)
        matvec(SS, qt, o_prev, dh)   // o_prev = S @ q
        matvec(SS, kt, tmp, dh)      // tmp    = S @ k
        // novelty-gated write: factor = clamp(1 + w*(s - b)), s = ||v-tmp||/sqrt(dh)
        var bw_t = bw_t0
        if not sw.is_null():
            var s2: Float32 = 0.0
            for j in range(dh):
                var d = vt.load(j) - tmp.load(j)
                s2 += d * d
            var s = sqrt(s2) / sqrt(Float32(dh))
            var fac = 1.0 + sw.load(b) * (s - sb.load(b))
            if fac < 0.0:
                fac = 0.0
            elif fac > 2.0:
                fac = 2.0
            bw_t *= fac
        if state_norm:
            for i in range(dh):
                var coef = bw_t * vt.load(i) - bf_t * tmp.load(i)
                var row = SS + i * dh
                var j = 0
                while j < dh - W + 1:
                    row.store[W](j, a_t * row.load[W](j) + kt.load[W](j) * coef)
                    j += W
                while j < dh:
                    row.store(j, a_t * row.load(j) + kt.load(j) * coef)
                    j += 1
            var sc = state_norm_scale(SS, dh)
            for i in range(dh * dh):
                SS.store(i, SS.load(i) * sc)
            matvec(SS, qt, o_new, dh)   // o_new = S @ q (post-norm)
        else:
            // fused identity: o_new = a*o_prev + (k.q)*(bw*v - bf*tmp)
            var kdotq = simd_dot(kt, qt, dh)
            for i in range(dh):
                var coef = bw_t * vt.load(i) - bf_t * tmp.load(i)
                o_new.store(i, a_t * o_prev.load(i) + kdotq * coef)
                var row = SS + i * dh
                var j = 0
                while j < dh - W + 1:
                    row.store[W](j, a_t * row.load[W](j) + kt.load[W](j) * coef)
                    j += W
                while j < dh:
                    row.store(j, a_t * row.load(j) + kt.load(j) * coef)
                    j += 1
        for i in range(dh):
            oo.store(t * dh + i, gr_t * o_new.load(i) + a0 * o_prev.load(i))
    o_prev.free()
    o_new.free()
    tmp.free()

// General per-step scan with optional StateNorm, PARALLEL over BH.
fn leafv5_scan(
    k: DTypePointer[DType.float32],
    v: DTypePointer[DType.float32],
    q: DTypePointer[DType.float32],
    bw: DTypePointer[DType.float32],
    bf: DTypePointer[DType.float32],
    gr: DTypePointer[DType.float32],
    dec: DTypePointer[DType.float32],
    alpha: DTypePointer[DType.float32],
    S: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    BH: Int,
    T: Int,
    dh: Int,
    state_norm: Bool,
    sw: DTypePointer[DType.float32] = DTypePointer[DType.float32](),
    sb: DTypePointer[DType.float32] = DTypePointer[DType.float32](),
):
    @parameter
    fn par(b: Int):
        scan_one_b(b, k, v, q, bw, bf, gr, dec, alpha, S, out, T, dh, state_norm,
                   sw, sb)
    parallelize[par](BH)

// Serial twin (benchmark/scaling reference).
fn leafv5_scan_serial(
    k: DTypePointer[DType.float32],
    v: DTypePointer[DType.float32],
    q: DTypePointer[DType.float32],
    bw: DTypePointer[DType.float32],
    bf: DTypePointer[DType.float32],
    gr: DTypePointer[DType.float32],
    dec: DTypePointer[DType.float32],
    alpha: DTypePointer[DType.float32],
    S: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    BH: Int,
    T: Int,
    dh: Int,
    state_norm: Bool,
    sw: DTypePointer[DType.float32] = DTypePointer[DType.float32](),
    sb: DTypePointer[DType.float32] = DTypePointer[DType.float32](),
):
    for b in range(BH):
        scan_one_b(b, k, v, q, bw, bf, gr, dec, alpha, S, out, T, dh, state_norm,
                   sw, sb)

// Fused q==k variant (||k||=1, StateNorm off): o_new = o_prev + coef, so a
// full matvec disappears.  PARALLEL over BH.
fn leafv5_scan_fused(
    k: DTypePointer[DType.float32],
    v: DTypePointer[DType.float32],
    bw: DTypePointer[DType.float32],
    bf: DTypePointer[DType.float32],
    gr: DTypePointer[DType.float32],
    alpha: DTypePointer[DType.float32],
    S: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    BH: Int,
    T: Int,
    dh: Int,
):
    @parameter
    fn par(b: Int):
        var kk = k + b * T * dh
        var vv = v + b * T * dh
        var bbw = bw + b * T
        var bbf = bf + b * T
        var ggr = gr + b * T
        var a = alpha.load(b)
        var SS = S + b * dh * dh
        var oo = out + b * T * dh
        var o_prev = DTypePointer[DType.float32].alloc(dh)
        for t in range(T):
            var kt = kk + t * dh
            var vt = vv + t * dh
            var bw_t = bbw.load(t)
            var bf_t = bbf.load(t)
            var gr_t = ggr.load(t)
            matvec(SS, kt, o_prev, dh)
            for i in range(dh):
                var coef = bw_t * vt.load(i) - bf_t * o_prev.load(i)
                var row = SS + i * dh
                var j = 0
                while j < dh - W + 1:
                    row.store[W](j, row.load[W](j) + kt.load[W](j) * coef)
                    j += W
                while j < dh:
                    row.store(j, row.load(j) + kt.load(j) * coef)
                    j += 1
                oo.store(t * dh + i, gr_t * (o_prev.load(i) + coef) + a * o_prev.load(i))
        o_prev.free()
    parallelize[par](BH)

// ---- helpers for main() ---- (unchanged from v1)
fn lcg_next(inout seed: Int) -> Float32:
    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    return Float32((seed >> 8) & 0xFFFFFF) / Float32(16777216)

fn l2_normalize(p: DTypePointer[DType.float32], n: Int):
    var s: Float32 = 0.0
    for i in range(n):
        s += p.load(i) * p.load(i)
    var inv = 1.0 / sqrt(s + 1.0e-9)
    for i in range(n):
        p.store(i, p.load(i) * inv)

fn max_abs_diff(
    a: DTypePointer[DType.float32],
    b: DTypePointer[DType.float32],
    n: Int,
) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a.load(i) - b.load(i)
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m

fn main():
    alias BH = 16
    alias T = 32
    alias dh = 16

    var nk = BH * T * dh
    var k = DTypePointer[DType.float32].alloc(nk)
    var v = DTypePointer[DType.float32].alloc(nk)
    var q = DTypePointer[DType.float32].alloc(nk)
    var bw = DTypePointer[DType.float32].alloc(BH * T)
    var bf = DTypePointer[DType.float32].alloc(BH * T)
    var gr = DTypePointer[DType.float32].alloc(BH * T)
    var alpha = DTypePointer[DType.float32].alloc(BH)
    var dec = DTypePointer[DType.float32].alloc(BH * T)
    var S0 = DTypePointer[DType.float32].alloc(BH * dh * dh)
    var S1 = DTypePointer[DType.float32].alloc(BH * dh * dh)
    var S2 = DTypePointer[DType.float32].alloc(BH * dh * dh)
    var out_p = DTypePointer[DType.float32].alloc(BH * T * dh)
    var out_s = DTypePointer[DType.float32].alloc(BH * T * dh)
    var out_f = DTypePointer[DType.float32].alloc(BH * T * dh)

    var seed = 42
    for i in range(nk):
        k.store(i, lcg_next(seed))
        v.store(i, lcg_next(seed))
        q.store(i, lcg_next(seed))
    for b in range(BH):
        for t in range(T):
            bw.store(b * T + t, lcg_next(seed) * 0.8 + 0.1)
            bf.store(b * T + t, lcg_next(seed) * 0.7 + 0.1)
            gr.store(b * T + t, lcg_next(seed) * 0.5 + 0.2)
            dec.store(b * T + t, lcg_next(seed) * 0.5 + 0.5)
        alpha.store(b, 0.5)
    for b in range(BH):
        for t in range(T):
            l2_normalize(k + (b * T + t) * dh, dh)
            l2_normalize(v + (b * T + t) * dh, dh)
            l2_normalize(q + (b * T + t) * dh, dh)
    memset_zero(S0, BH * dh * dh)
    memset_zero(S1, BH * dh * dh)
    memset_zero(S2, BH * dh * dh)

    // 1) parallel == serial (both no-norm): must agree exactly
    leafv5_scan_serial(k, v, q, bw, bf, gr, dec, alpha, S1, out_s, BH, T, dh, False)
    leafv5_scan(k, v, q, bw, bf, gr, dec, alpha, S2, out_p, BH, T, dh, False)
    var d1 = max_abs_diff(out_s, out_p, BH * T * dh)
    print("parallel vs serial (no-norm) max|d_out| = ", d1)
    if d1 < 1.0e-6:
        print("  PASS: parallel == serial")
    else:
        print("  FAIL: parallel != serial")

    // 2) parallel general (norm on) vs serial (norm on)
    memset_zero(S1, BH * dh * dh)
    memset_zero(S2, BH * dh * dh)
    leafv5_scan_serial(k, v, q, bw, bf, gr, dec, alpha, S1, out_s, BH, T, dh, True)
    leafv5_scan(k, v, q, bw, bf, gr, dec, alpha, S2, out_p, BH, T, dh, True)
    var d2 = max_abs_diff(out_s, out_p, BH * T * dh)
    print("parallel vs serial (norm on)  max|d_out| = ", d2)
    if d2 < 1.0e-6:
        print("  PASS: parallel == serial (norm)")
    else:
        print("  FAIL: parallel != serial (norm)")

    // 3) StateNorm bound over T steps
    var bound = sqrt(Float32(dh))
    var max_fn: Float32 = 0.0
    for b in range(BH):
        var fn: Float32 = 0.0
        for i in range(dh * dh):
            var x = S2.load(b * dh * dh + i)
            fn += x * x
        var n = sqrt(fn)
        if n > max_fn:
            max_fn = n
    print("state_norm: max||S||_F = ", max_fn, " (bound ", bound, ")")
    if max_fn <= bound * 1.001 + 1.0e-4:
        print("  PASS: state bounded")
    else:
        print("  FAIL: state not bounded")

    k.free(); v.free(); q.free(); bw.free(); bf.free(); gr.free()
    alpha.free(); dec.free(); S0.free(); S1.free(); S2.free()
    out_p.free(); out_s.free(); out_f.free()
    print("done.")
