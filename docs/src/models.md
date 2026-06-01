```@meta
CurrentModule = NGCLearn
```

# Models

Models are composed networks built from the component zoo. They live in
`src/models/`, are plain `mutable struct`s holding the component graph plus a
driver function, and are **not** themselves [`JaxComponent`](@ref)s — they
compose components, they aren't one.

Each model builds its graph inside an `NGCSimLib.Context`, `post_init!`s every
component, **then** wires it with `>>` (the setup-before-wire rule — see
[Architecture & Design](architecture.md)), and drives it with a hand-ordered
eager loop. Two models ship today, spanning two very different learning
paradigms.

## PCN — predictive coding (`pc_discrim`)

A faithful port of the discriminative predictive-coding network of
**Whittington & Bogacz (2017)**. Two coupled sub-networks (a generative path
`z → e → W` with static feedback, and an inference/projection path `q → Q`)
trained with the PEM cycle:

1. **Projection** — tie inference weights `Q = W` and feedback `E = Wᵀ`, then
   feed-forward project to seed the latents.
2. **Expectation** — `T` steps of error-driven latent settling.
3. **Maximization** — one Hebbian-Adam weight update.

```julia
using NGCLearn

m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, eta=0.002, key=7)

x = reshape(Float64[1, 1, 0, 0], 1, 4)   # one observation
y = reshape(Float64[1, 0], 1, 2)         # one label

y_inf, y_mu, EFE = process!(m, x, y)     # one PEM training step
pred = project(m, x)                      # test-time inference (no learning)
```

On a small deterministic classification task this drives the mean output error
from ~0.7 to <0.001 over 60 epochs and classifies the training set 4/4.
See `examples/02_pc_discrim_train.jl` for the full loop.

```@docs
PCN
process!
project
```

## DC-SNN — spiking STDP (`diehl_cook_snn`)

A faithful port of the unsupervised spiking classifier of **Diehl & Cook
(2015)**. A Poisson-encoded input drives an excitatory LIF layer through
STDP-adapted weights, with fixed lateral inhibition implementing winner-take-all
competition:

```
z0 (Poisson) ──W1 (STDP)──▶ z1e (excitatory LIF)
                               │  ▲
                       (W1ei, eye, +)  (W1ie, hollow, −)
                               ▼  │
                            z1i (inhibitory LIF)

z1e.j = Summation(W1.outputs, W1ie.outputs)   # feedforward + lateral inhibition
```

`process!` runs a `T`-step stimulus window: reset → clamp the observation into
the Poisson encoder → advance all components each step (running the STDP
`evolve!` when adapting) → L1-normalize `W1` ([`norm!`](@ref)).

```julia
using NGCLearn

m = DC_SNN(; in_dim=64, hid_dim=10, T=200, key=3)
obs = fill(0.9, 1, 64)                 # one stimulus pattern (batch size 1)

counts = process!(m, obs; adapt=true)  # 1×hid_dim excitatory spike counts
```

Under drive, lateral inhibition produces the expected winner-take-all behavior:
one excitatory unit dominates the spike counts while the rest are suppressed.

```@docs
DC_SNN
norm!
```

## Writing your own model

The two shipped models are the template. The essentials:

- Build all components inside `NGCSimLib.Context(name) do _ctx … end`, with a
  distinct `name` per model instance (same-named Contexts share global-state
  slots).
- Call `post_init!` on **every** component before any `>>` wiring.
- Drive with an explicit, ordered sequence of `advance_state!` / `evolve!`
  calls (mirror the upstream process ordering); the eager path is the ground
  truth.

The rationale for each of these is in [Architecture & Design](architecture.md).
