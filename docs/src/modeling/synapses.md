```@meta
CurrentModule = NGCLearn
```

# Synapses

The synapse is a key building block for connecting/wiring together the various
component cells that one would use for characterizing a biomimetic neural
system. These objects perform, per simulated time step, a specific type of
transformation -- here, a linear transform -- using their underlying synaptic
parameters. A synaptic cable is represented by a matrix of weights that conducts
a projection of an input signal (presented to a pre-synaptic/input compartment),
producing an output signal (in a post-synaptic compartment).

ngc-learn organizes synapses into two broad families: 1) non-plastic *static*
synapses, which only perform a fixed transformation of input signals; and 2)
*plastic* synapses, which additionally carry out long-term evolution of their
weights. Plastic synapse components are associated with a local plasticity rule
-- e.g., a Hebbian-type update or a trace-based spike-timing rule -- triggered
online at simulation time steps.

Every synapse is a `mutable struct <: `[`JaxComponent`](@ref) that owns named
[`Compartment`](https://cognitivesubstratesai.github.io/NGCSimLib/)s and steps
its forward dynamics with [`advance_state!`](@ref) and re-initialises its
compartments with [`reset_state!`](@ref). Plastic synapses additionally
implement a plasticity verb ([`compute_update!`](@ref) and/or [`evolve!`](@ref)).
Construct synapses with their keyword constructors and drive them inside an
`NGCSimLib.Context` so that wiring (`>>`) propagates -- see
[Architecture & Design](../architecture.md). Compartment values are read with
`NGCSimLib.get_value(c.weights)` and written with `NGCSimLib.set!(c.weights, x)`.
The PRNG seed is a `UInt64` `key` (created via [`make_prng_key`](@ref)) rather
than a JAX key.

Each synapse below is a 1:1 port of upstream `ngc-learn`, verified on the eager
path against hand-computed dynamics. Several methods support construction-time
sparsification via the `p_conn` probability-of-existence argument, producing a
sparsely-connected weight matrix (a multiplicative `mask` compartment also gates
the weights at runtime).

> **Not yet ported.** Upstream ngc-learn additionally provides convolutional /
> deconvolutional synapses, dynamic (chemical) synapses (exponential,
> double-exponential, alpha), short-term-plasticity synapses, BCM, exponential
> and event-driven STDP, and reward-modulated (three-factor, MSTDP-ET) synapses.
> These are **not yet ported** to NGCLearn.jl and are omitted here.

## Non-Plastic Synapse Types

### Static (Dense) Synapse

This synapse performs a linear transform of its input signals. It does not
evolve and is meant to be used for fixed-value (dense) synaptic connections. Its
forward dynamics are

```math
\mathbf{outputs} = \big(\mathbf{inputs} \cdot (\mathbf{W} \odot \mathbf{mask})\big)\, R_\text{scale} + \mathbf{b}
```

where ``\mathbf{W}`` is the weight matrix, ``\mathbf{mask}`` a multiplicative
connectivity gate (zero entries implement sparsity / pruning), ``R_\text{scale}``
the `resist_scale` factor, and ``\mathbf{b}`` an optional bias. Like all dense
cables, it supports construction-time sparsification through the `p_conn`
argument.

```@docs
DenseSynapse
```

## Multi-Factor Learning Synapse Types

Hebbian rules operate in a local manner -- they use information immediately
available to synapses in both space and time -- and can come in a variety of
flavors. One way to categorize variants of Hebbian learning is to clarify what
neural statistics they operate on (real-valued information vs. discrete spikes)
and how many factors (distinct terms) are involved in the update.

### (Two-Factor) Hebbian Synapse

This synapse performs the same linear transform as [`DenseSynapse`](@ref) and
evolves according to a strictly two-factor/term update rule: the synaptic
efficacy matrix is changed according to a product between pre-synaptic
compartment values (`pre`) and post-synaptic compartment values (`post`), which
can contain any type of vector/matrix statistics. This synapse further features
tools for advanced forms of Hebbian descent/ascent, such as applying the update
through an adaptive learning rate optimizer (adaptive moment estimation, i.e.,
Adam).

**Update rule.** Given a (mini-batch of) pre-synaptic activity ``\mathbf{pre}``
and post-synaptic activity ``\mathbf{post}``, the two-factor Hebbian adjustment
is the outer product

```math
\Delta \mathbf{W} = (\gamma_\text{pre}\, \mathbf{pre})^\top\, (\gamma_\text{post}\, \mathbf{post}),
\qquad
\Delta \mathbf{b} = \textstyle\sum_\text{batch} (\gamma_\text{post}\, \mathbf{post})
```

where ``\gamma_\text{pre}`` (`pre_wght`) and ``\gamma_\text{post}`` (`post_wght`)
re-weight the two factors. When a soft bound is enabled (`w_bound > 0`), the
weight update is scaled toward the bound,

```math
\Delta \mathbf{W} \leftarrow \Delta \mathbf{W} \odot (w_\text{bound} - |\mathbf{W}|),
```

and an optional prior contributes a regularization term ``\Delta\mathbf{W}_\text{reg}``:

```math
\Delta\mathbf{W}_\text{reg} =
\begin{cases}
-\lambda\, \mathbf{W} & \text{(L2 / ridge)} \\
-\lambda\, \operatorname{sign}(\mathbf{W}) & \text{(L1 / lasso)} \\
s\,\big(-r\,\operatorname{sign}(\mathbf{W}) - \tfrac{1-r}{2}\,\mathbf{W}\big) & \text{(L1L2 / elastic-net, } \lambda = (s, r))
\end{cases}
```

The final update is ``(\Delta\mathbf{W} + \Delta\mathbf{W}_\text{reg},\ \Delta\mathbf{b})``,
scaled by `sign_value` (``+1`` for ascent, ``-1`` for descent). The verb
[`compute_update!`](@ref) writes these into the `dWeights` / `dBiases`
compartments without stepping the parameters; [`evolve!`](@ref) does the same
and then steps `weights` / `biases` through the embedded optimizer (`"sgd"` or
`"adam"`, with learning rate `eta`), re-applies hard bounding (clipping to
``[-w_\text{bound}, w_\text{bound}]``, or ``[0, w_\text{bound}]`` when
`is_nonnegative`), and re-applies the connectivity `mask`.

This rule is local: it operates only on pre- and post-synaptic activity.

```@docs
HebbianSynapse
compute_update!
```

## Spike-Timing-Dependent Plasticity (STDP) Synapse Types

Synapses that evolve according to a spike-timing-dependent plasticity (STDP)
process operate, at a high level, much like multi-factor Hebbian rules (STDP is
a generalization of Hebbian adjustment to spike trains) and share many of their
properties. A distinguishing factor is that STDP synapses must involve action
potential pulses (spikes) in their calculations, and they typically compute
synaptic change according to the relative timing of spikes. In principle the
synapse in this grouping adapts its efficacies according to a rule built from
(at least) four terms: a pre-synaptic spike (an "event"), a pre-synaptic delta
timing (here a trace), a post-synaptic spike (or event), and a post-synaptic
delta timing (also a trace). STDP rules in ngc-learn typically enforce
soft/hard synaptic strength bounding -- there is a maximum magnitude allowed for
any single synaptic efficacy, and by default an STDP synapse enforces its
strengths to be non-negative.

Note: these rules are technically considered "two-factor" rules since they only
operate on pre- and post-synaptic activity (despite each factor being
represented by two or more terms).

### Trace-based STDP

This is a four-term STDP rule that adjusts the underlying synaptic strength
matrix via a weighted combination of long-term depression (LTD) and long-term
potentiation (LTP). For the LTP portion of the update, a pre-synaptic trace and
a post-synaptic event/spike-trigger are used; for the LTD portion, a
pre-synaptic event/spike-trigger and a post-synaptic trace are utilized. This
rule can be configured to use a soft, power-scaling form of STDP via the
hyper-parameter `mu`.

**Update rule.** In the additive regime (`mu = 0`), with pre-synaptic trace
``\mathbf{z}_\text{pre}``, pre-synaptic spikes ``\mathbf{s}_\text{pre}``,
post-synaptic trace ``\mathbf{z}_\text{post}`` and post-synaptic spikes
``\mathbf{s}_\text{post}``, the weight change is

```math
\Delta \mathbf{W}
= A_+\, (\mathbf{z}_\text{pre} - x_\text{tar})^\top\, \mathbf{s}_\text{post}
\;-\; A_-\, \mathbf{s}_\text{pre}^\top\, \mathbf{z}_\text{post},
```

where ``A_+`` (`A_plus`) and ``A_-`` (`A_minus`) are the LTP and LTD strengths
and ``x_\text{tar}`` (`pretrace_target`) is a pre-synaptic trace target. The LTD
term is omitted when ``A_- = 0``.

When power-law scaling is enabled (`mu > 0`), the LTP term is scaled by
``(w_\text{bound} - \mathbf{W})^\mu`` and the LTD term by ``\mathbf{W}^\mu``,
recovering a soft-bounded, power-scaling form of STDP:

```math
\Delta \mathbf{W}
= A_+\, (w_\text{bound} - \mathbf{W})^{\mu} \odot \big((\mathbf{z}_\text{pre} - x_\text{tar})^\top\, \mathbf{s}_\text{post}\big)
\;-\; A_-\, \mathbf{W}^{\mu} \odot \big(\mathbf{s}_\text{pre}^\top\, \mathbf{z}_\text{post}\big).
```

The plasticity verb [`evolve!`](@ref) applies this update with a global rate
`eta` and an optional weight decay (`tau_w > 0`),

```math
\mathbf{W} \leftarrow \operatorname{clip}\Big(\mathbf{W} + \eta\, \Delta\mathbf{W} - \tfrac{\mathbf{W}}{\tau_w},\ w_\epsilon,\ w_\text{bound} - w_\epsilon\Big),
```

then re-applies the connectivity `mask`. The pre/post trace compartments are
typically fed by [`VarTrace`](@ref) low-pass filters driven by the spiking cell
outputs.

```@docs
TraceSTDPSynapse
```

## Plasticity verb

The plastic synapses on this page share the [`evolve!`](@ref) verb (and, for
[`HebbianSynapse`](@ref), the companion [`compute_update!`](@ref) verb that
computes the update without stepping the parameters). The exact signature
differs by component: `HebbianSynapse` evolves with a time step
(`evolve!(c, dt)`), while `TraceSTDPSynapse` evolves without one
(`evolve!(c)`).

```@docs
evolve!
```

## References

```bibtex
@article{morrison2008phenomenological,
  title={Phenomenological models of synaptic plasticity based on spike timing},
  author={Morrison, Abigail and Diesmann, Markus and Gerstner, Wulfram},
  journal={Biological cybernetics},
  volume={98},
  number={6},
  pages={459--478},
  year={2008}
}

@article{bi2001synaptic,
  title={Synaptic modification by correlated activity: Hebb's postulate revisited},
  author={Bi, Guoqiang and Poo, Muming},
  journal={Annual review of neuroscience},
  volume={24},
  number={1},
  pages={139--166},
  year={2001}
}
```
