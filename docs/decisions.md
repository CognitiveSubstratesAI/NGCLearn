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

## 4. The EAGER path is the ground truth; the Parser/Reactant JIT rewrite now
##    works (world-age is the only remaining JIT-invocation gap)

Date: 2026-05-31, updated 2026-06-01  ·  **(load-bearing — read before wrapping any cell in `@compile`)**

NGCSimLib's `@compilable` does two things: (1) defines the method normally so
eager Julia dispatch works, and (2) registers the body Expr for the Parser to
rewrite into a pure `_pure_<Type>_<method>(ctx; kwargs...)` for Reactant tracing.

For NGCLearn cells, **(1) works fully and is what we test and trust.** Pre-
`setup!` compartments act as plain mutable value holders (NGCSimLib
Compartment.jl: `get_value`→`initial_value`, `set!`→writes it), so a cell can be
constructed and stepped standalone — no Context, no GlobalState — and verified
against hand-computed dynamics. That is the ground-truth path and the contract
the JIT path must later match bit-for-bit.

**(2) now WORKS for NGCLearn cells** (updated 2026-06-01). The Parser inlines
scalar hyperparameter field accesses as trace-time constants — `c.tau_m` becomes
the literal `10.0`, `c.thr` becomes `-52.0`, etc. — while compartment accesses
become `ctx[key]`. Verified: `parse_method(LIFCell, :advance_state!)` produces a
clean `_pure_LIFCell_advance_state!(ctx; dt, t)` with **zero** dangling receiver
references, and `compile_process!` succeeds on it.

Two fixes got it there:
  - NGCSimLib's `ContextTransformer` gained the scalar-inline branch (resolve the
    field value off the instance at parse time, splice it as a literal) — this
    pre-dated 2026-06-01 and was already in place when re-checked.
  - The broadcast-recursion fix (NGCSimLib `2c424fe`, 2026-06-01): an explicit
    broadcast `f.(args)` shares AST head `:.` with field access, so the
    transformer skipped its subtree and left `c.field` nested inside a broadcast
    (e.g. `max.(_v, c.v_min)`) un-rewritten. Now it recurses. This was the last
    dangling-reference gap.

**Remaining limitation is world-age, NOT the parser.** The rewritten pure
function is `eval`'d at `compile_process!` time; calling it in the SAME world-age
hits Julia's "method too new" error. This is a PRE-EXISTING trait of the whole
compiled-process path — NGCSimLib's own tests stop at "compiles successfully" for
exactly this reason — and is orthogonal to the rewrite correctness. Real use
(compile in one top-level scope, run later) or `Base.invokelatest` sidesteps it;
a proper fix (RuntimeGeneratedFunctions, or building the runner without runtime
`eval`) is a separate NGCSimLib task.

**Bottom line:** the eager path remains the verified ground truth for all
component/model tests. The JIT rewrite is now correct and unblocked; wrapping a
Process in `Reactant.@compile` is the next milestone, gated only on the world-age
invocation mechanism, not on the parser.

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

## 7. Composed networks live in `src/models/` — and must setup-before-wire

Date: 2026-05-31

Multi-component networks (first: `PCN`, the pc_discrim exhibit) live in
`src/models/<name>.jl` as a plain `mutable struct` holding the component graph
plus a driver (`process!`). They are NOT `JaxComponent`s — they compose
components, they aren't one. The model builds its graph inside a
`NGCSimLib.Context`, drives components eagerly (hand-ordered `advance_state!`),
and preserves upstream's advance/reset/project orderings exactly (see `pcn.jl`
`_advance!`/`_project!`). We do not use `MethodProcess`/`compile_process!`: the
Reactant-JIT path can't yet trace scalar-hyperparameter components (§4), and the
eager loop is the ground-truth oracle.

**CRITICAL — setup-before-wire.** Every compartment must be `post_init!`-ed
(which assigns its global-state root key) BEFORE any `>>` wiring. `wire!`
snapshots the SOURCE compartment's `target` key at call time (mirrors upstream
`__rrshift__`, compartment.py:150-165). If you wire before setup, the source's
target is still `nothing`, the wire copies `nothing`, and the dest's later
`setup!` points it at its own slot — silently severing the connection. Symptom:
the whole forward path reads zeros and nothing learns (cost a full
write-commit-revert cycle on 2026-05-31 before reading upstream). Upstream gets
correct ordering for free because its metaclass runs `_setup` at construction;
in Julia we order it by hand: construct all → `post_init!` all → THEN `>>`.
`test/test_pcn_integration.jl` has a "wires are live" testset guarding this.

Weight tying (`Q = W`, `E = Wᵀ`) and latent seeding use explicit
`get_value`/`set!` in `process!`, re-applied each sample — faithful to upstream's
`PCN.process` (pcn_model.py:320-346). Acceptance is a deterministic learnability
test (error collapses >10×, trained net classifies the set), with every asserted
number OBSERVED from a real run first — never an aspirational threshold.

## Decision update policy

Same as NGCSimLib: update when a decision affects more than one file or would
surprise a reader who didn't write the original code. Per-file decisions stay in
code comments. Correct in place when a decision changes; don't leave stale advice
next to current advice.
