# Benchmark results — reference run

These are **indicative** numbers from one machine; re-run `jit_eager.jl` for
figures on your hardware. They measure *speed*, not correctness — the JIT path's
bit-for-bit equivalence to eager is asserted in `test/test_jit_integration.jl`.

## `jit_eager.jl` — eager vs Reactant-JIT (single LIFCell step)

Timing methodology: min over 7 batches of 200 reps each, GC disabled during each
batch (so the figures are warm-cache best-case throughput, not including the
one-time compile/trace cost).

| n_units | eager µs/step | JIT µs/step | speedup |
|--------:|--------------:|------------:|--------:|
| 64      | 24.2          | 2.5         | 9.52×   |
| 256     | 39.4          | 11.9        | 3.32×   |
| 1024    | 65.6          | 23.9        | 2.74×   |
| 4096    | 260.5         | 61.1        | 4.26×   |

**Takeaway:** the Reactant/XLA path is consistently faster than the eager-spliced
runner across sizes (≈2.7–9.5× here). The advantage is largest at small layer
widths — where Julia-side per-call dispatch/allocation dominates the eager path
and XLA's fixed overhead is negligible — compresses in the mid-range, and widens
again at large N as XLA's optimized kernels pay off. The non-monotonic *ratio* is
expected; both paths scale monotonically in absolute time.

### Environment

```
Julia Version 1.12.6
OS:      Linux (x86_64-linux-gnu)
CPU:     2 × Intel(R) Core(TM) i7-3630QM @ 2.40 GHz
WORD_SIZE: 64
Reactant 0.2.262   (CPU backend — no GPU on this machine)
```

### Caveats

- **CPU-only run.** This machine has no GPU; XLA is on its CPU backend. On a GPU
  the JIT advantage at large `n_units` would likely be substantially larger.
- **Excludes compile time.** `compile_with_reactant!` pays a one-time trace +
  XLA-compile cost (seconds) not reflected above; the speedup is per-step
  steady-state. JIT is worthwhile when a Process is run many times (e.g.
  multi-step inference / training loops), not for a single step.
- **Old 2-core CPU.** Absolute numbers will differ markedly on newer hardware;
  treat the *ratios* as the more transferable signal, and even those only
  loosely.
