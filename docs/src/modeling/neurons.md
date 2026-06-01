```@meta
CurrentModule = NGCLearn
```

# Neuronal Cells

The neuron (or neuronal cell) represents one of the fundamental building blocks
of a biomimetic neural system. These objects perform, per simulated time step,
a calculation of output activity values given an internal arrangement of
compartments (the sources where signals from other neuronal cell(s) are
deposited). Typically a neuron integrates an (ordinary) differential equation
whose form depends on the type of neuronal cell and the dynamics under
consideration.

Every cell is a `mutable struct <: `[`JaxComponent`](@ref) holding named
[`Compartment`](https://cognitivesubstratesai.github.io/NGCSimLib/)s. Drive it
one step with [`advance_state!`](@ref) and re-initialise its compartments with
[`reset_state!`](@ref). Construct cells with their keyword constructors and read
or write compartments with `NGCSimLib.get_value(c.v)` / `NGCSimLib.set!(c.v, x)`.
Stochastic cells carry a `UInt64` PRNG seed in their `key` compartment (seeded
via [`make_prng_key`](@ref)) rather than a JAX key.

NGCLearn.jl currently ports three of upstream `ngc-learn`'s neuronal cells —
[`RateCell`](@ref) and [`GaussianErrorCell`](@ref) from the graded family and
[`LIFCell`](@ref) from the spiking family. Each is a 1:1 port verified on the
eager path against the upstream dynamics.

## Graded, Real-valued Neurons

This family of neuronal cells adheres to dynamics — or performs calculations —
utilizing graded (real-valued/continuous) values; in other words, they do not
produce any discrete signals or action-potential values.

### The Rate Cell

This cell evolves one set of dynamics over state `z` (a sort of real-valued
continuous membrane potential). The "electrical" inputs that drive it include
`j` (non-modulated signals) and `j_td` (modulated signals), which can be mapped
to bottom-up and top-down pressures (such as those produced by error neurons) if
one is building a strictly hierarchical neural model. Note that the "spikes" `zF`
emitted are real-valued for the rate-cell and are produced by applying a
nonlinear activation function `fx` (default `identity`, configured by the user
through `act_fx`).

The state `z` is governed by a leaky differential equation whose leak term is the
derivative of a centred scale-shift prior over `z`:

```math
\tau_m \frac{dz}{dt} = -\gamma\, \text{prior}'(z) + (j + j_\text{td}),
\qquad z_F = f_x(z)\cdot \text{scale}
```

where `\gamma` is the prior leak rate (`prior = (name, leak_rate)`) and the prior
derivative `\text{prior}'(z)` selects the leak form. The four ported priors are:

```math
\text{prior}'(z) =
\begin{cases}
z & \text{gaussian (default)} \\
\operatorname{sign}(z) & \text{laplacian} \\
\dfrac{2z}{1 + z^2} & \text{cauchy} \\
2\,z\,e^{-z^2} & \text{exp}
\end{cases}
```

The bottom-up current is first modulated by the activation derivative,
`j \leftarrow j \cdot f_x'(z)`, then scaled by `resist_scale`, before being fed
to the integrator (`euler` or `rk2`). An optional thresholding sub-dynamic
(`threshold = (kind, lmbda)`, with `kind` one of `"none"`, `"soft_threshold"`,
`"cauchy_threshold"`) can be applied to the integrated state.

When `tau_m <= 0` the cell switches to a **stateless** passthrough that skips
integration entirely (`z = j + j_\text{td}`), matching upstream's
`_run_cell_stateless` short-circuit.

```@docs
RateCell
```

### The Error Cell

This cell is (currently) a stateless neuron — it is not driven by an underlying
differential equation, thus emulating a "fixed-point" error or mismatch
calculation. Variations of the fixed-point error cell depend on the local
distribution assumed over mismatch activities; a Gaussian distribution yields a
Gaussian error cell, which also shapes its internal compartments (typically a
`target`, `mu`, `dtarget`, and `dmu`).

#### Gaussian Error Cell

This cell is fixed to be a Gaussian cell. Note that it has several important
compartments: among the input compartments, `target` holds the desired target
activity level while `mu` holds an externally produced mean prediction value;
among the output compartments, `dtarget` is the first derivative with respect to
the target (sometimes used to emulate a top-down pressure/expectation in
predictive coding) and `dmu` is the first derivative with respect to the mean.
A variance/covariance compartment `Sigma` provides the precision weighting and a
log-likelihood `L` is emitted as a local free-energy term.

Given prediction `\mu`, target `t`, and (co)variance `\Sigma`, the cell computes
the precision-weighted mismatch and the local Gaussian log-likelihood:

```math
e = t - \mu, \qquad
\delta\mu = \frac{e}{\Sigma}, \qquad
\delta t = -\,\delta\mu, \qquad
L = -\frac{1}{2\,\Sigma}\sum e^2
```

The error signals `\delta\mu` and `\delta t` are gated by a `modulator` and a
one-shot `mask` (the mask is "eaten" — reset to all-ones — after each step).

The scalar-`sigma` path is ported and tested. Upstream also supports a 4-D
convolutional `shape` and a full covariance matrix; the `shape` field plumbing is
preserved, but the full-covariance form of `L` is **not yet ported** (the scalar
collapse follows upstream's behaviour).

```@docs
GaussianErrorCell
```

!!! note "Other graded error cells"
    Upstream additionally provides Laplacian and Bernoulli error cells. These
    are **not yet ported**.

## Spiking Neurons

These neuronal cells exhibit dynamics that involve emission of discrete action
potentials (or spikes). Typically such neurons are modeled with multiple
compartments, including at least one for the electrical current `j`, the
membrane potential `v`, the voltage threshold `thr`, and the action potential
`s`. The interactions or dynamics underlying each component may themselves be
complex and nonlinear, depending on the neuronal cell simulated (i.e., some
neurons run multiple differential equations under the hood).

### The LIF (Leaky Integrate-and-Fire) Cell

This cell (the "leaky integrator") models dynamics over the voltage `v` and an
optional homeostatic threshold shift `thr_theta`. The baseline `thr` serves as
the membrane-potential threshold while `thr_theta` is treated as a form of
short-term plasticity, so that the full threshold at time `t` is
`thr + thr_theta(t)`.

The membrane voltage evolves under the leaky integrate-and-fire ODE:

```math
\tau_m \frac{dv}{dt} = (v_\text{rest} - v)\, g_L + (j \cdot R)\, m_\text{rfr}
```

where `R` is `resist_m`, `g_L` is the leak conductance (`conduct_leak`; `g_L = 1`
gives a leaky integrator, `g_L = 0` a pure integrate-and-fire), and
`m_\text{rfr} = \mathbb{1}[\text{rfr} \ge \text{refract\_T}]` is a refractory
mask that gates out the input current while the cell is still within its
refractory window. Integration uses the `euler` or `rk2` (midpoint) scheme.

A spike is emitted and the voltage is reset when the voltage crosses the full
threshold:

```math
s = \mathbb{1}\!\left[v > (\text{thr} + \text{thr\_theta})\right], \qquad
v \leftarrow (1-s)\,v + s\,v_\text{reset}
```

After a spike the refractory timer is reset
(`rfr \leftarrow (rfr + dt)(1 - s)`) and the time-of-last-spike compartment
`tols` is updated to the current time `t`.

When the homeostatic mechanism is active (`tau_theta > 0`), the adaptive
threshold shift decays exponentially and is bumped by `theta_plus` on each raw
spike:

```math
\text{thr\_theta} \leftarrow \text{thr\_theta}\, e^{-dt/\tau_\theta}
   + s_\text{raw}\, \theta_+
```

Two optional lateral-competition modes are supported. With `one_spike = true`, a
single spike is stochastically retained (per row, via the cell's `key`) whenever
more than one unit fires. With `max_one_spike = true`, only the spike at the
maximum-voltage unit in each row is kept. An optional voltage floor `v_min`
clamps `v` from below.

```@docs
LIFCell
```

!!! note "Other spiking cells"
    Upstream additionally provides several spiking cells — among them the
    simplified LIF (sLIF), plain IF, winner-take-all (WTAS), quadratic LIF,
    adaptive-exponential (AdEx), FitzHugh–Nagumo, resonate-and-fire, Izhikevich,
    and Hodgkin–Huxley cells. These are **not yet ported**.

## References

The dynamics, framing, and default biophysical constants above follow NACLab's
upstream `ngc-learn` documentation and source. See the upstream project and the
predictive-coding / spiking-neuron literature it cites for the original
derivations.
