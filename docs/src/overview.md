```@meta
CurrentModule = NGCLearn
```

# Overview

Advances in research on artificial neural networks (ANNs) have led to many
breakthroughs in machine learning and beyond, resulting in powerful models that
can categorize and forecast, as well as agents that can play games and solve
complex problems. Behind these achievements is the backpropagation of errors
(or backprop) algorithm. Although elegant and powerful, a major long-standing
criticism of backprop has been its biological implausibility: it is not likely
that the brain adjusts the synapses connecting the billions of neurons that
compose it in the way that backprop would prescribe.

Although ANNs are (loosely) inspired by our current understanding of the human
brain, their connection to the actual mechanisms that drive systems of natural
neurons is loose at best. While the question of how the brain exactly conducts
*credit assignment* — the process of determining the contribution of each and
every neuron to the system's overall error on some task (the "blame game") — is
still an open one, it would prove invaluable to have a flexible computational
and software framework that can facilitate the design and development of
brain-inspired neural systems that can also learn complex tasks. These tasks
range from generative modeling to interacting with and manipulating
dynamically-evolving environments. Such a framework would benefit researchers in
fields including, but not limited to, machine learning, (computational)
neuroscience, and cognitive science.

`ngc-learn`[^1] aims to fill this need by concretely instantiating neuronal
dynamics and forms of synaptic plasticity as flexibly rearranged components and
operations, in order to build arbitrary, modular, and complex biomimetic systems
for research in brain-inspired computing and neurocognitive modeling. More
importantly, it is designed to facilitate the design, development, and analysis
of novel models of neural computation and information processing, neuronal
circuitry, biologically-plausible credit assignment, and neuromimetic agents.
Specifically, it implements a general schema for simulating biomimetic systems
characterized by differential equations, including ones based on biophysical
*spiking neuronal cells*.

The overarching goal is to provide researchers and engineers with:

* a modular design that allows for the flexible creation, simulation, and
  analysis of neural systems and circuits under the framework of predictive
  processing;
* an approachable tool, written and maintained by researchers directly studying
  and advancing predictive processing and biomimetic systems, meant to lower the
  barriers to entry to this field of research;
* a "model museum" that captures the essence of fundamental and interesting
  predictive processing and other biomimetic models, allowing for the study of
  and experimentation with classical and modern ideas.

## History

The `ngc-learn` software framework was originally developed in 2019 by the
[Neural Adaptive Computing (NAC) laboratory](https://www.cs.rit.edu/~ago/nac_lab.html)
at the Rochester Institute of Technology to serve as an internal tool for
predictive coding research (with earlier incarnations in the Scala programming
language dating back to early 2017). It remains actively maintained by and used
for predictive processing and biomimetics research in the NAC lab (see
`ngc-learn`'s mention in this
[engineering blog post](https://engineeringcommunity.nature.com/posts/the-neural-coding-framework-for-learning-generative-models)).

**NGCLearn.jl** is a faithful 1:1 Julia port of NACLab's Python `ngc-learn`. It
preserves the upstream component schema, neuronal dynamics, and learning rules,
adapting only the surface API to idiomatic Julia (mutating methods such as
`advance_state!` / `reset_state!` / `evolve!`, keyword constructors,
`get_value` / `set!` compartment access, and a `UInt64` PRNG seed in place of a
JAX key). It is **Layer 1** of the NGC stack, built on
[NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib) (Layer 0 — the
Component / Compartment / Context / Process substrate), with the
predictive-coding graph framework FabricPC.jl planned as Layer 2.

## Citation

Please cite `ngc-learn`'s source/core paper if you use this framework in your
publications:

```bibtex
@article{Ororbia2022,
  author  = {Ororbia, Alexander and Kifer, Daniel},
  title   = {The neural coding framework for learning generative models},
  journal = {Nature Communications},
  year    = {2022},
  month   = {Apr},
  day     = {19},
  volume  = {13},
  number  = {1},
  pages   = {2064},
  issn    = {2041-1723},
  doi     = {10.1038/s41467-022-29632-7},
  url     = {https://doi.org/10.1038/s41467-022-29632-7}
}
```

## What NGCLearn provides today

The Julia port currently implements the following subset of the upstream
component zoo. Components not yet ported (e.g. convolutional synapses,
additional spiking cell models) are omitted here and noted in their respective
pages.

**Neurons**

* [`LIFCell`](@ref) — leaky integrate-and-fire spiking neuron.
* [`RateCell`](@ref) — graded, rate-coded leaky neuron.
* [`GaussianErrorCell`](@ref) — graded fixed-point predictive-coding error unit.

**Synapses**

* [`DenseSynapse`](@ref) — static dense linear cable (upstream `StaticSynapse`).
* [`HebbianSynapse`](@ref) — two-factor Hebbian plastic cable.
* [`TraceSTDPSynapse`](@ref) — trace-based spike-timing-dependent plasticity.

**Encoders & traces**

* [`PoissonCell`](@ref) — converts real-valued inputs to a Poisson spike train.
* [`VarTrace`](@ref) — low-pass filter / spike-trace accumulator.

**Models** (assembled, learning networks)

* [`PCN`](@ref) — predictive coding network (`pc_discrim` exhibit).
* [`DC_SNN`](@ref) — Diehl & Cook spiking network with STDP (`diehl_cook_snn`
  exhibit).

[^1]: The name `ngc-learn` stems from an important theory in neuroscience that
    served as one of the library's first motivations — *predictive processing*,
    which posits that the brain is largely a continual prediction engine,
    constantly hypothesizing the state of its environment and updating its own
    internal mental model of it as data is gathered. The very first paradigm of
    neural computation that `ngc-learn` implemented and offered general support
    for was a predictive coding framework known as neural generative coding
    (NGC) (Ororbia and Kifer, 2022).
