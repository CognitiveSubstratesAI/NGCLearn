```@meta
CurrentModule = NGCLearn
```

# Input Encoders & Trace Operators

This page covers two families of utility nodes: **input encoders**, which
transform sensory input / data into a desired form (typically spike trains), and
**other operators** such as variable traces, which post-process or filter the
signals flowing between components.

An input encoder is generally a non-parameterized component that, ideally
on-the-fly, transforms an input pattern into a spike train — for example, one can
use a [`PoissonCell`](@ref) to convert a fixed real-valued input into an
approximate spike train with a maximum frequency (in Hertz). Some of these
encoders can be interpreted as coarse-grained approximations of dynamics that
would normally be modeled by complex differential equations.

Trace operators, by contrast, sit downstream of other components: they integrate
a differential equation over an external compartment value (e.g. the spike `s` of
a spiking cell) to produce a real-valued cumulative representation across time.
A [`VarTrace`](@ref) is the canonical example — a low-pass filter of a (usually
spiking) signal sequence, oft-used to drive local synaptic updates.

Every node here is a `mutable struct <: `[`JaxComponent`](@ref) holding named
[`Compartment`](https://cognitivesubstratesai.github.io/NGCSimLib/)s. Drive one
step with [`advance_state!`](@ref) and re-initialise its compartments with
[`reset_state!`](@ref). Construct nodes with their keyword constructors and read
or write compartments with `NGCSimLib.get_value(c.v)` / `NGCSimLib.set!(c.v, x)`.
Stochastic nodes carry a `UInt64` PRNG seed in their `key` compartment (seeded
via [`make_prng_key`](@ref)) rather than a JAX key.

## Input Encoding Operators

Input encoders take a data pattern and transform it to another desired form,
i.e., a real-valued vector to a sample of a spike train at time `t`. Some
encoders emulate aspects of biological/biophysical cells, e.g., the
Poisson-distributed nature of the spikes emitted by certain neuronal cells.

NGCLearn.jl currently ports one encoder, the [`PoissonCell`](@ref).

### The Poisson Cell

This cell takes a real-valued pattern and transforms it on-the-fly into a spike
train, where each spike vector is a sample of a Poisson spike-train with a maximum
frequency (given in Hertz). This assumes that each dimension of the real-valued
pattern is normalized to `[0, 1]` (otherwise, results may not be as expected).

Per step, the per-dimension spike probability is the input magnitude scaled by the
time step (in seconds, `dt` being in milliseconds) and the maximum frequency
`target_freq`; a spike is emitted wherever an i.i.d. uniform draw falls below it:

```math
p_\text{spike} = \text{inputs}\cdot\frac{dt}{1000}\cdot \text{target\_freq},
\qquad
\varepsilon \sim \mathcal{U}(0, 1),
\qquad
\text{outputs} = \mathbb{1}\!\left[\varepsilon < p_\text{spike}\right]
```

The time-of-last-spike compartment `tols` is then latched to the current time `t`
wherever a spike fired, and held otherwise:

```math
\text{tols} \leftarrow (1 - \text{outputs})\cdot \text{tols}
   + \text{outputs}\cdot t
```

The `key` compartment (a `UInt64` PRNG seed) is advanced after every draw so that
successive steps consume fresh randomness.

```@docs
PoissonCell
```

!!! note "Other input encoders"
    Upstream additionally provides the Bernoulli, Latency, and Phasor cells.
    These are **not yet ported**.

## Trace Operators

Other operators range from variable traces to kernels and hand-crafted
transformations. An important and oft-used one, in the case of spiking neural
systems, is the variable trace (or filter): one might need to track a cumulative
value based on spikes over time in order to trigger local updates to synaptic
cable values.

### The Variable Trace

This operator processes and tracks a particular value (dependent on which external
component's compartment is wired into its `inputs`). In general, a trace
integrates a differential equation based on an external compartment value — e.g.,
the spike `s` of a spiking neuronal cell — producing a real-valued cumulative
representation of it across time. Instead of directly tracking spike times, a
trace gives a soft, single-valued approximation; equivalently, it acts as a
low-pass filter of another signal sequence (and is consumed by trace-based STDP
synapses such as [`TraceSTDPSynapse`](@ref)).

Each step first decays the running `trace` by a factor that depends on
`decay_type` and the trace time constant `tau_tr`:

```math
\text{decay} =
\begin{cases}
e^{-dt/\tau_\text{tr}} & \text{"exp" (default)} \\[2pt]
1 - dt/\tau_\text{tr} & \text{"lin"} \\[2pt]
0 & \text{otherwise ("step")}
\end{cases}
\qquad
x_\text{tr} = \gamma_\text{tr}\cdot \text{trace}\cdot \text{decay}
```

where `\gamma_\text{tr}` is the trace scale (`gamma_tr`). The decayed trace is
then updated by one of three rules, selected by `n_nearest_spikes` and `a_delta`:

```math
x_\text{tr} \leftarrow
\begin{cases}
x_\text{tr} + \text{inputs}\cdot\left(a_\delta - \dfrac{\text{trace}}{k}\right)
  & k > 0 \quad\text{(}k\text{-nearest-neighbor)} \\[10pt]
x_\text{tr} + \text{inputs}\cdot a_\delta
  & a_\delta > 0 \quad\text{(additive)} \\[6pt]
x_\text{tr}\cdot(1 - \text{inputs}) + \text{inputs}\cdot P_\text{scale}
  & \text{otherwise (gated snap)}
\end{cases}
```

Here `a_\delta` (`a_delta`) is the spike increment, `k` (`n_nearest_spikes`) the
nearest-neighbor count, and `P_\text{scale}` (`P_scale`) the snap target used by
the gated form. The resulting `x_\text{tr}` is written to both the `trace` state
compartment and the `outputs` compartment.

```@docs
VarTrace
```

!!! note "Other operators and kernels"
    Upstream additionally provides spike-response-model kernels such as the
    exponential kernel (`ExpKernel`). These are **not yet ported**.

## References

The dynamics, framing, and default constants above follow NACLab's upstream
`ngc-learn` documentation and source. See the upstream project and the
spiking-neuron / spike-timing-dependent-plasticity literature it cites for the
original derivations.
