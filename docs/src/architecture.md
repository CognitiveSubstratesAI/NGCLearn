```@meta
CurrentModule = NGCLearn
```

# Architecture & Design

NGCLearn is a faithful 1:1 port of upstream `ngc-learn`, adapted to idiomatic
Julia. This page explains the load-bearing design decisions a user or
contributor needs to know. The full, dated decision log lives in
`docs/decisions.md` in the repository; this is the narrative summary.

## The component protocol

Every component is a `mutable struct <: `[`JaxComponent`](@ref) (itself
`<: NGCSimLib.AbstractComponent`). The abstract type — not a macro-injected
mixin — is the dispatch root for the shared verbs [`advance_state!`](@ref),
[`reset_state!`](@ref), and (for plastic synapses) [`evolve!`](@ref). Each
concrete type declares the four standard component fields (`name`,
`context_path`, `args`, `kwargs`) explicitly, then its scalar hyperparameters,
then its `Compartment` fields. Reflection over the struct keeps only the
`Compartment`-typed fields, so the substrate machinery works unchanged.

Two naming conventions worth noting:

- **Scalar hyperparameters stay scalar.** Fields like `tau_m`, `v_rest`,
  `target_freq` are plain typed fields accessed as `c.tau_m` — not
  `Compartment`s. (This even fixes a latent upstream bug where `LIFCell`
  accessed `self.tau_m.get()` on a non-compartment.)
- **PRNG state is a `UInt64` seed**, carried in a `key` compartment and driven
  through `Xoshiro` — the idiomatic Julia analog of upstream's splittable JAX
  key array. [`make_prng_key`](@ref) mints one (OS-random, or explicit for
  reproducibility).

## Wiring: setup-before-wire

Components communicate by wiring one component's output compartment into
another's input with `>>`:

```julia
NGCSimLib.Context("net") do _ctx
    enc = PoissonCell(; name="enc", n_units=4)
    syn = TraceSTDPSynapse(; name="syn", shape=(4, 8), A_plus=1e-2, A_minus=1e-4)

    NGCSimLib.post_init!.((enc, syn))   # ① set up ALL components first
    enc.outputs >> syn.inputs           # ② THEN wire
end
```

**The order is load-bearing.** `wire!` (invoked by `>>`) snapshots the *source*
compartment's global-state key at call time. If you wire before
`post_init!`, the source has no key yet, the wire copies `nothing`, and the
connection is silently severed — the downstream component reads zeros and
nothing propagates. Always `post_init!` every component, *then* wire. The
shipped [Models](models.md) follow this; so should yours.

## Eager vs JIT

This is the single most important thing to understand about the current state
of the port.

- **The eager path is the ground truth.** A component constructed and stepped
  with plain `advance_state!` calls (optionally wired inside a `Context`) is
  fully functional and is what every test verifies. All numbers in the docs and
  tests are observed from this path.
- **The Reactant JIT path is a later phase.** Upstream marks methods
  `@compilable` so a `Process` can be traced and XLA-compiled. NGCLearn keeps
  the `@compilable` annotations, but tracing a full `Process` does not yet work
  for components with **scalar hyperparameters**: the NGCSimLib parser only
  rewrites `c.field` accesses that resolve to a `Compartment`, leaving
  `c.tau_m`-style scalars dangling in the rewritten pure function. The fix
  belongs in NGCSimLib (inline scalar hyperparameters as trace-time constants),
  after which the eager path becomes the JIT path's conformance oracle.

Because of this, the shipped models drive components with explicit, hand-ordered
eager loops rather than `MethodProcess`/`compile_process!`. The upstream process
*orderings* are preserved exactly, so swapping in a real compiled `Process`
later is mechanical.

## Faithful, but corrected

The port is **upstream-but-corrected**: where a clear upstream bug is found
during porting, it is fixed and the divergence is documented in the relevant
file's preamble and the decision log (e.g. the `LIFCell.tau_m` access above).
Scope reductions are likewise explicit — for instance `GaussianErrorCell` ships
the scalar-variance predictive-coding path; the full-covariance reduction is
deferred until a model needs it.

## Verification discipline

Acceptance is reproduction of an `ngc-museum` exhibit, not just unit tests.
[`PCN`](@ref) reproduces `pc_discrim` (it learns a task end-to-end);
[`DC_SNN`](@ref) reproduces `diehl_cook_snn` (winner-take-all under lateral
inhibition). Per-component port specs (line-by-line vs the Python source) live
under `docs/specs/`.
