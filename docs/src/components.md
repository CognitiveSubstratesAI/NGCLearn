```@meta
CurrentModule = NGCLearn
```

# Components

Every component is a `mutable struct <: `[`JaxComponent`](@ref) that owns named
[`Compartment`](https://cognitivesubstratesai.github.io/NGCSimLib/)s and
implements the shared verbs [`advance_state!`](@ref) (one simulation step) and
[`reset_state!`](@ref). Plastic synapses additionally implement
[`evolve!`](@ref). Construct components with keyword constructors; drive them
inside an `NGCSimLib.Context` so wiring (`>>`) propagates — see
[Architecture & Design](architecture.md).

All components are 1:1 ports of upstream `ngc-learn`, verified on the eager
path against hand-computed dynamics. The per-component port specs (line-by-line
vs the Python source) live under `docs/specs/` in the repository.

## Shared protocol

```@docs
JaxComponent
advance_state!
reset_state!
make_prng_key
```

## Neurons

The three ported neuronal cells — [`LIFCell`](@ref) (spiking),
[`RateCell`](@ref) (graded rate) and [`GaussianErrorCell`](@ref) (graded
predictive-coding error) — and their full per-cell dynamics are documented on
the dedicated [Neuronal Cells](modeling/neurons.md) page.

## Synapses

The three ported synaptic cables — [`DenseSynapse`](@ref) (static linear),
[`HebbianSynapse`](@ref) (two-factor Hebbian + optimizer) and
[`TraceSTDPSynapse`](@ref) (spike-timing-dependent plasticity) — along with the
plasticity verbs [`compute_update!`](@ref) and [`evolve!`](@ref), are documented
with their full learning-rule theory on the dedicated [Synapses](modeling/synapses.md)
page.

## Input encoders & traces

The input encoder [`PoissonCell`](@ref) (real values → Poisson spike train) and
the trace node [`VarTrace`](@ref) (low-pass spike-trace accumulator, the pre/post
trace source for [`TraceSTDPSynapse`](@ref)) are documented with their full
theory on the dedicated [Input Encoders & Traces](modeling/input_encoders.md)
page.
