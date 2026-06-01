# NGCLearn benchmarks

Performance benchmarks for NGCLearn, kept separate from the test suite (which
asserts *correctness* — that the JIT path matches eager bit-for-bit — not speed).

## `jit_eager.jl` — eager vs Reactant-JIT

Times a single `LIFCell` step through the two compiled-process paths across a
range of layer widths:

- **eager** — `compile_process!`, a plain-Julia spliced ctx runner
- **JIT** — `compile_with_reactant!`, the same runner traced through
  `Reactant.@compile` (XLA)

Both produce identical output (the conformance property is tested in
`test/test_jit_integration.jl`); this script measures the speed difference.

Run it:

```bash
julia --project=benchmark benchmark/jit_eager.jl
```

(The `benchmark/` environment dev-links NGCLearn + NGCSimLib and adds Reactant;
first run pays the Reactant precompile cost.)

A captured reference run — including the machine/version it was measured on — is
in [`results.md`](results.md). Numbers are machine- and Reactant-version-
dependent, so treat the committed table as indicative, not a contract; re-run
locally for figures on your hardware.
