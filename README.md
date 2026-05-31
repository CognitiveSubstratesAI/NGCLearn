# NGCLearn.jl

[![CI](https://github.com/CognitiveSubstratesAI/NGCLearn/actions/workflows/CI.yml/badge.svg)](https://github.com/CognitiveSubstratesAI/NGCLearn/actions/workflows/CI.yml)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://cognitivesubstratesai.github.io/NGCLearn/stable/)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://cognitivesubstratesai.github.io/NGCLearn/dev/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Julia port of [NACLab's ngc-learn](https://github.com/NACLab/ngc-learn) — the
biophysical component zoo (spiking + graded neurons, synapses, encoders) for
building neurobiologically-plausible / predictive-coding models.

This is **Layer 1** of the NGC stack:

| Layer | Package | Role |
|------:|---------|------|
| 0 | [NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib) | Simulation substrate — Component / Compartment / Context / Process |
| **1** | **NGCLearn** (this repo) | **Biophysical component zoo — neurons & synapses** |
| 2 | FabricPC.jl *(planned)* | Predictive-coding graph training framework |

NGCLearn depends on NGCSimLib via a `[sources]` path dev-link (see
`Project.toml`); clone both side by side under the same parent directory.

## Status — Phase A scaffold (v0.1.0)

First two components ported, eager-path verified against hand-computed dynamics:

- **`LIFCell`** — leaky integrate-and-fire spiking neuron (Euler + midpoint
  integration, refractory period, optional homeostatic adaptive threshold,
  one-spike / max-one-spike modes, voltage floor).
- **`GaussianErrorCell`** — fixed-point Gaussian error/mismatch cell for
  predictive coding (precision-scaled error + local log-likelihood).
- **`ode_utils`** — the ODE integration backend (`get_integrator_code`,
  `step_euler`, `step_rk2`).

These two cells are the minimum needed to start reproducing the `pc_discrim`
ngc-museum exhibit (the Phase-A acceptance gate).

> **Eager vs JIT:** components are verified on the eager dispatch path, which is
> the ground truth. Tracing a full Process through Reactant needs an NGCSimLib
> parser enhancement (inline scalar hyperparameters as trace-time constants) —
> see `docs/decisions.md` #4. That is a later phase, with the eager path as its
> conformance oracle.

## Layout

```
src/
  NGCLearn.jl                       # module + includes + exports
  utils/diffeq/ode_utils.jl         # integration backend
  components/
    jax_component.jl                # abstract JaxComponent base + shared verbs
    neurons/spiking/lif_cell.jl     # LIFCell
    neurons/graded/gaussian_error_cell.jl  # GaussianErrorCell
test/                               # per-component eager-path tests
docs/
  decisions.md                      # cross-cutting design decisions
  specs/                            # per-component line-by-line port specs
```

## Quick start

```julia
using NGCLearn
using NGCSimLib: get_value, set!

cell = LIFCell(; name="layer1", n_units=4, tau_m=10.0)
set!(cell.j, [120.0 80.0 200.0 40.0])   # drive with input current
advance_state!(cell, 1.0, 1.0)          # (dt, t)
get_value(cell.s)                        # emitted spikes
```

## Development

Julia **1.12+** (inherits NGCSimLib's `OncePerProcess` floor).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Before committing any `.jl` change, run the Blue formatter (CI enforces it):

```bash
julia -e 'using JuliaFormatter; format(".")'
```

## License

BSD 3-Clause. Portions are a Julia port of `ngc-learn` (NAC Lab, RIT), also
BSD 3-Clause. See `LICENSE`.
