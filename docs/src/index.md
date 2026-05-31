```@meta
CurrentModule = NGCLearn
```

# NGCLearn.jl

**Julia port of [NACLab](https://www.cs.rit.edu/~ago/nac_lab.html)'s
[`ngc-learn`](https://github.com/NACLab/ngc-learn).**

The biophysical component zoo — spiking + graded neurons and synapses — for
building neurobiologically-plausible / predictive-coding models. Built on
[NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib) (Layer 0).

```@contents
Pages = [
    "getting_started/installation.md",
    "getting_started/quickstart.md",
    "api/index.md"
]
Depth = 2
```

## Layer in the stack

NGCLearn is **Layer 1** of the NGC Julia stack:

| Layer | Package | Role |
|-------|---------|------|
| 0 | NGCSimLib | substrate — Component / Compartment / Context / Process |
| 1 | **NGCLearn** (this) | biophysical component zoo (neurons, synapses) |
| 2 | FabricPC.jl | predictive-coding graph framework |

## What's inside

- **Neurons** — `LIFCell` (spiking), `RateCell` (graded), `GaussianErrorCell`
  (predictive-coding error unit)
- **Synapses** — `DenseSynapse` (static linear cable), `HebbianSynapse`
  (two-factor Hebbian + SGD/Adam)
- **Models** — `PCN`, the `pc_discrim` predictive-coding network (Whittington &
  Bogacz 2017) with the full PEM training loop
- **Backend** — `ode_utils` (Euler / midpoint integration) and `optim` (SGD, Adam)

## Eager vs JIT

Components are verified on the **eager dispatch path**, which is the ground
truth. Tracing a full `Process` through Reactant needs an NGCSimLib parser
enhancement (inlining scalar hyperparameters as trace-time constants) — that is
a later phase, with the eager path as its conformance oracle.

## Quick example

See [Quickstart](getting_started/quickstart.md) for a runnable walk-through.
```
