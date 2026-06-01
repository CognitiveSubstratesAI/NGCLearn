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

### LIFCell (spiking)

Leaky integrate-and-fire dynamics:

```math
\tau_m \frac{dv}{dt} = (v_\text{rest} - v)\, g_L + j\, R, \qquad
\text{spike if } v > \theta,\ \text{then } v \leftarrow v_\text{reset}
```

with a refractory period, an optional homeostatic adaptive threshold
(`thr_theta`), and one-spike / max-one-spike winner-take-all modes.

```@docs
LIFCell
```

### RateCell (graded)

Rate-coded leaky neuron with a scale-shift prior on the leak term:

```math
\tau_m \frac{dz}{dt} = -\gamma\, \text{prior}'(z) + (j + j_\text{td}),
\qquad z_F = f_x(z)\cdot \text{scale}
```

Priors: gaussian, laplacian, cauchy, exp. Stateless passthrough when
`tau_m ≤ 0`.

```@docs
RateCell
```

### GaussianErrorCell (graded, predictive coding)

A fixed-point error unit computing a precision-weighted mismatch and its local
Gaussian log-likelihood:

```math
e = \text{target} - \mu, \quad
\delta\mu = e / \Sigma, \quad
L = -\tfrac{1}{2\Sigma}\textstyle\sum e^2
```

```@docs
GaussianErrorCell
```

## Synapses

### DenseSynapse (static)

A dense linear cable with no in-built learning (upstream `StaticSynapse`):

```math
\text{outputs} = (\text{inputs} \cdot (W \odot \text{mask}))\, R + b
```

```@docs
DenseSynapse
```

### HebbianSynapse (2-factor Hebbian + optimizer)

Same forward pass as `DenseSynapse`, plus a two-factor Hebbian update
`dW = \text{pre}^\top \text{post}` (with optional priors/bounds) stepped through
an embedded SGD or Adam optimizer.

```@docs
HebbianSynapse
compute_update!
```

### TraceSTDPSynapse (spike-timing-dependent plasticity)

Trace-based STDP (Morrison et al. 2007; Bi & Poo 2001):

```math
dW = A_+\, (\text{preTrace} - x_\text{tar})^\top \text{postSpike}
   - A_-\, \text{preSpike}^\top \text{postTrace}
```

with optional power-law scaling (`mu > 0`), global rate `eta`, weight decay
(`tau_w > 0`), and clipping to `[w_eps, w_bound]`.

```@docs
TraceSTDPSynapse
```

### Plasticity verb

```@docs
evolve!
```

## Input encoders

### PoissonCell

Converts real-valued inputs into a homogeneous Poisson spike train, with
per-dimension spike probability `pspike = inputs · (dt/1000) · target_freq`.

```@docs
PoissonCell
```

## Traces

### VarTrace

A low-pass filter (spike-trace accumulator) with exp / lin / step decay and
additive, gated-snap, or k-nearest-neighbor update modes — the pre/post trace
source for [`TraceSTDPSynapse`](@ref).

```@docs
VarTrace
```
