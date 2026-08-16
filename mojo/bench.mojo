// bench.mojo — benchmark the Mojo LEAFv5 scan kernels on T4-like shapes.
//   mojo run mojo/bench.mojo
// The same shapes are benchmarked for the C twin and torch in
// mojo/c_ref/bench.py (numbers in the README).
//
// v2 (2026-08-10): also times the PARALLEL kernels (parallelize over BH) vs
// the serial twins, so you can see the multi-core scaling on the host.

from algorithm import parallelize
from memory import DTypePointer, DType, memset_zero
from math import sqrt
from time import now

from leafv5 import (
    leafv5_scan, leafv5_scan_serial, leafv5_scan_fused,
    lcg_next, l2_normalize, max_abs_diff,
)

fn main():
    alias BH = 192
    alias T = 512
    alias dh = 48

    var nk = BH * T * dh
    var k = DTypePointer[DType.float32].alloc(nk)
    var v = DTypePointer[DType.float32].alloc(nk)
    var q = DTypePointer[DType.float32].alloc(nk)
    var bw = DTypePointer[DType.float32].alloc(BH * T)
    var bf = DTypePointer[DType.float32].alloc(BH * T)
    var gr = DTypePointer[DType.float32].alloc(BH * T)
    var dec = DTypePointer[DType.float32].alloc(BH * T)
    var alpha = DTypePointer[DType.float32].alloc(BH)
    var S = DTypePointer[DType.float32].alloc(BH * dh * dh)
    var S2 = DTypePointer[DType.float32].alloc(BH * dh * dh)
    var S3 = DTypePointer[DType.float32].alloc(BH * dh * dh)
    var out_p = DTypePointer[DType.float32].alloc(BH * T * dh)
    var out_s = DTypePointer[DType.float32].alloc(BH * T * dh)
    var out_f = DTypePointer[DType.float32].alloc(BH * T * dh)

    var seed = 7
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
    memset_zero(S, BH * dh * dh)
    memset_zero(S2, BH * dh * dh)
    memset_zero(S3, BH * dh * dh)

    // correctness first: parallel == serial (both norm states)
    leafv5_scan_serial(k, v, q, bw, bf, gr, dec, alpha, S2, out_s, BH, T, dh, False)
    leafv5_scan(k, v, q, bw, bf, gr, dec, alpha, S, out_p, BH, T, dh, False)
    var d = max_abs_diff(out_s, out_p, BH * T * dh)
    print("parallel vs serial max|d_out| = ", d)
    if d < 1.0e-6:
        print("  PASS")
    else:
        print("  FAIL")

    var positions = Float64(BH * T)
    var flops = 2.0 * Float64(BH) * Float64(T) * Float64(dh) * Float64(dh) * 3.0 * 2.0

    fn time_ms(f: fn() -> None, reps: Int) -> Float64:
        var t0 = now()
        for _ in range(reps):
            f()
        return Float64(now() - t0) / 1.0e6 / Float64(reps)

    // general, norm on (training path): serial vs parallel
    var ms = time_ms(fn() -> None: leafv5_scan_serial(k, v, q, bw, bf, gr, dec, alpha, S, out_s, BH, T, dh, True), 3)
    print("Mojo general norm  : serial ", ms, " ms  ", positions / ms * 1.0e3, " pos/s  ", flops / (ms * 1.0e-3) / 1.0e9, " GFLOP/s")
    ms = time_ms(fn() -> None: leafv5_scan(k, v, q, bw, bf, gr, dec, alpha, S, out_p, BH, T, dh, True), 3)
    print("Mojo general norm  : PARAL  ", ms, " ms  ", positions / ms * 1.0e3, " pos/s  ", flops / (ms * 1.0e-3) / 1.0e9, " GFLOP/s")

    // general, norm off (fused algebraic path): serial vs parallel
    ms = time_ms(fn() -> None: leafv5_scan_serial(k, v, q, bw, bf, gr, dec, alpha, S, out_s, BH, T, dh, False), 3)
    print("Mojo general fused : serial ", ms, " ms  ", positions / ms * 1.0e3, " pos/s  ", flops / (ms * 1.0e-3) / 1.0e9, " GFLOP/s")
    ms = time_ms(fn() -> None: leafv5_scan(k, v, q, bw, bf, gr, dec, alpha, S, out_p, BH, T, dh, False), 3)
    print("Mojo general fused : PARAL  ", ms, " ms  ", positions / ms * 1.0e3, " pos/s  ", flops / (ms * 1.0e-3) / 1.0e9, " GFLOP/s")

    // fused q==k (1 matvec; parallel)
    ms = time_ms(fn() -> None: leafv5_scan_fused(k, v, bw, bf, gr, alpha, S, out_f, BH, T, dh), 3)
    print("Mojo fused q==k     : PARAL  ", ms, " ms  ", positions / ms * 1.0e3, " pos/s  ", flops / (ms * 1.0e-3) / 1.0e9, " GFLOP/s")

    k.free(); v.free(); q.free(); bw.free(); bf.free(); gr.free()
    dec.free(); alpha.free(); S.free(); S2.free(); S3.free()
    out_p.free(); out_s.free(); out_f.free()
    print("done.")
