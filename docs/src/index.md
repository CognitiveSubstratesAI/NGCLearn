```@meta
CurrentModule = NGCLearn
```

# NGCLearn.jl

**Julia port of [NACLab](https://www.cs.rit.edu/~ago/nac_lab.html)'s
[`ngc-learn`](https://github.com/NACLab/ngc-learn).** The biophysical *component
zoo* — spiking and graded neurons, plastic and static synapses, input encoders,
and traces — for building neurobiologically-plausible and predictive-coding
models in Julia.

NGCLearn is **Layer 1** of the NGC stack, built on
[NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib) (Layer 0, the
Component / Compartment / Context / Process substrate). Models are assembled by
wiring components together and driving them; see [Models](models.md) for two
complete, learning examples.

## The NGC stack

| Layer | Package | Role |
|------:|---------|------|
| 0 | [NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib) | substrate — Component / Compartment / Context / Process |
| **1** | **NGCLearn** (this) | biophysical component zoo (neurons, synapses, encoders, traces) |
| 2 | FabricPC.jl *(planned)* | predictive-coding graph training framework |

Each layer is a faithful, independently-versioned port verified against the
upstream [`ngc-museum`](https://github.com/NACLab/ngc-museum) exhibits.

## What's inside

**Neurons**

| Component | Kind | Summary |
|-----------|------|---------|
| [`LIFCell`](@ref) | spiking | leaky integrate-and-fire (refractory, adaptive threshold, WTA modes) |
| [`RateCell`](@ref) | graded | rate-coded leaky neuron with scale-shift priors |
| [`GaussianErrorCell`](@ref) | graded | fixed-point predictive-coding error unit |

**Synapses**

| Component | Learning | Summary |
|-----------|----------|---------|
| [`DenseSynapse`](@ref) | none (static) | dense linear cable (upstream `StaticSynapse`) |
| [`HebbianSynapse`](@ref) | 2-factor Hebbian + SGD/Adam | gradient-style plastic cable |
| [`TraceSTDPSynapse`](@ref) | trace-based STDP | spike-timing-dependent plasticity |

**Encoders & traces**

| Component | Summary |
|-----------|---------|
| [`PoissonCell`](@ref) | converts real inputs to a Poisson spike train |
| [`VarTrace`](@ref) | low-pass filter / spike-trace accumulator |

**Models** (assembled networks — see [Models](models.md))

| Model | Paradigm | Exhibit |
|-------|----------|---------|
| [`PCN`](@ref) | predictive coding | `pc_discrim` (Whittington & Bogacz 2017) |
| [`DC_SNN`](@ref) | spiking + STDP | `diehl_cook_snn` (Diehl & Cook 2015) |

**Backend**: [`ode_utils`](@ref get_integrator_code) (Euler / midpoint
integration) and optimizers ([`adam_init`](@ref), [`sgd_init`](@ref)).

## Eager vs JIT

Every component is verified on the **eager dispatch path**, which is the ground
truth — construct a component, call `advance_state!`, read the result, all in
plain Julia. Tracing a full `Process` through Reactant (the JIT path) needs an
NGCSimLib parser enhancement (inlining scalar hyperparameters as trace-time
constants); that is a later phase, with the eager path as its conformance
oracle. See [Architecture & Design](architecture.md) for the full story.

## Read next

```@contents
Pages = [
    "getting_started/installation.md",
    "getting_started/quickstart.md",
    "components.md",
    "models.md",
    "architecture.md",
    "api/index.md"
]
Depth = 2
```

## Citation

If you use this work, please also cite the upstream `ngc-learn`:

```bibtex
@article{ororbia2022neural,
  title   = {The neural coding framework for learning generative models},
  author  = {Ororbia, Alexander and Kifer, Daniel},
  journal = {Nature Communications},
  volume  = {13},
  number  = {1},
  pages   = {2064},
  year    = {2022}
}
```
