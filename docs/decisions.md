# NGCLearn design decisions log

Cross-cutting patterns for the Layer-1 component zoo. Per-file rationale lives
in code comments at the point it's made — this file captures decisions that
**span the package** or that future-you needs without grep'ing every source
file.

NGCLearn sits on top of NGCSimLib (Layer 0). **~80% of NGCSimLib's decisions
apply here verbatim** — singletons via `OncePerProcess`, multiple dispatch over
metaclasses, no bare `get`/`set` shadowing `Base`, "fix-upstream-bugs-and-
document", Blue formatting, skip-Aqua-during-scaffold, Julia-1.12 floor. Read
`../../NGCSimLib/docs/decisions.md` first; the entries below are the
NGCLearn-specific deltas.

Format: one section per decision, dated. Append as we go. If we change our
minds, edit + re-date the section in place.

---

## 1. Component class hierarchy is an abstract type, not `@ngc_component`

Date: 2026-05-31

Upstream `LIFCell <: JaxComponent <: Component`. We preserve that two-level
hierarchy with an abstract type:

    abstract type JaxComponent <: NGCSimLib.AbstractComponent end
    mutable struct LIFCell <: JaxComponent ... end

We deliberately **do not** use NGCSimLib's `@ngc_component` macro for cells,
even though it would save the four standard-field declarations. The macro
hardcodes the supertype to `AbstractComponent` (NGCSimLib Component.jl:120),
which would erase the `JaxComponent` layer and the polymorphic dispatch root for
`advance_state!`/`reset_state!`. Each cell therefore declares the four standard
fields (`name`, `context_path`, `args`, `kwargs`) explicitly, then its
hyperparameters, then its compartments. `compartments(c)` still works — it
reflects over fields and keeps the `Compartment`-typed ones, ignoring the
scalar hyperparameter fields.

Implication: a new cell family copies the field-block + keyword-constructor
shape from `LIFCell`. The standard fields are boilerplate, but explicit and
greppable.

---

## 2. PRNG state is a `UInt64` seed in a Compartment, not a JAX key array

Date: 2026-05-31

Upstream `JaxComponent` holds `self.key = Compartment(random.PRNGKey(...))` — a
splittable JAX key array threaded through `random.split`. Julia's RNGs
(`Xoshiro`) seed from an integer, so the faithful analog is a `UInt64` seed
carried in the `key` compartment. `make_prng_key(seed)` mints one (OS-random
when `seed === nothing`, the analog of upstream's `time.time_ns()` seed;
explicit integer for reproducibility).

Where upstream does `key, skey = random.split(key)`, we do
`rng = Xoshiro(get_value(c.key)); ...; set!(c.key, rand(rng, UInt64))` — build a
generator from the stored seed, draw, then store a fresh advanced seed. Same
"deterministic given the seed, advances each draw" contract.

---

## 3. Scalar hyperparameters stay scalar — including fixing `LIFCell.tau_m`

Date: 2026-05-31

Cell hyperparameters (`tau_m`, `v_rest`, `thr`, ...) are plain typed struct
fields, accessed as `c.tau_m`. They are **not** compartments.

This surfaces a genuine upstream bug: `LIFCell.advance_state` calls
`self.tau_m.get()` (LIFCell.py:171), but `tau_m` is assigned a bare float in
`__init__` (LIFCell.py:124), and **every sibling cell** (IFCell, adExCell,
fitzhughNagumoCell, WTASCell) accesses it as a bare scalar. The `.get()` is the
bug; the port uses `c.tau_m` directly. (NGCSimLib decisions #5:
fix-and-document, don't port faithfully.)

---

## 4. The EAGER path is the ground truth; the Parser/Reactant JIT path is a
##    separate, later phase — and needs an NGCSimLib parser feature

Date: 2026-05-31  ·  **(load-bearing — read before wrapping any cell in `@compile`)**

NGCSimLib's `@compilable` does two things: (1) defines the method normally so
eager Julia dispatch works, and (2) registers the body Expr for the Parser to
rewrite into a pure `_pure_<Type>_<method>(ctx; kwargs...)` for Reactant tracing.

For NGCLearn cells, **(1) works fully and is what we test and trust.** Pre-
`setup!` compartments act as plain mutable value holders (NGCSimLib
Compartment.jl: `get_value`→`initial_value`, `set!`→writes it), so a cell can be
constructed and stepped standalone — no Context, no GlobalState — and verified
against hand-computed dynamics. That is the ground-truth path and the contract
the JIT path must later match bit-for-bit.

**(2) does NOT yet work for cells with scalar hyperparameters**, and this is a
real architectural finding, not an oversight:

  - The Parser's `ContextTransformer` only rewrites a `c.field` chain when it
    resolves to a `Compartment` (NGCSimLib ContextTransformer.jl:62 —
    `val isa Compartment ? val : nothing`). Scalar fields like `c.tau_m` resolve
    to `nothing` and are **left as literal `c.tau_m`**.
  - But the rewritten pure function has signature `(ctx; kwargs...)` — the
    receiver `c` is gone. A surviving `c.tau_m` references an undefined binding.
  - NGCSimLib's own example components only ever touched compartments + kwargs
    (e.g. `dt`, `leak` passed as kwargs), so this shape was never exercised.

The fix belongs in NGCSimLib, not here: teach the Parser to **inline scalar
hyperparameter field accesses as trace-time constants** (read `getproperty(c, f)`
off the instance at compile time and splice the literal value), since they are
immutable for the life of a compiled process. That mirrors how Reactant already
treats string ctx keys as compile-time constants (NGCSimLib decisions #9).

**Phasing (mirrors NGCSimLib's own substrate-first/Reactant-later sequence):**
Phase A (this work) ships eager, hand-verified cells. The JIT path lands after
the NGCSimLib parser enhancement, with the eager path as its conformance oracle.
Until then, do not claim a cell is Reactant-traceable through a Process.

---

## 5. Scalar-`sigma` predictive-coding path first; full covariance deferred

Date: 2026-05-31

`GaussianErrorCell` is ported on the scalar-`sigma` path (the common PC case):
`Sigma` is a `(1,1)` block, `dmu = e / Sigma`, and `L = -sum(e^2)*(0.5/Sigma)`
collapses to a scalar via `_squeeze`. Upstream also supports a 4-D convolutional
`shape` and a true covariance matrix; the `shape` field plumbing is preserved,
but full-covariance `L` is deferred until a downstream model (e.g. an ngc-museum
exhibit) actually exercises it. Documented so the gap is visible, not silent
(per [[feedback_no_sweeping_claims_without_grep]] in project memory).

---

## 6. Verb naming: `advance_state!` / `reset_state!`

Date: 2026-05-31

Upstream method names are `advance_state` and `reset`. We use `advance_state!`
and `reset_state!`: the `!` marks in-place mutation (Julia convention), and
`reset_state!` (over bare `reset`) avoids implying any relationship to a `Base`
verb and reads unambiguously. Process steps reference these by symbol
(`(cell, :advance_state!)`), so the spelling is the contract — keep it stable.

---

## Decision update policy

Same as NGCSimLib: update when a decision affects more than one file or would
surprise a reader who didn't write the original code. Per-file decisions stay in
code comments. Correct in place when a decision changes; don't leave stale advice
next to current advice.
