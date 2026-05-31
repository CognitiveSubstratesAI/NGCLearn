# Quickstart

## A single cell

```julia
using NGCLearn
using NGCSimLib: get_value, set!

cell = LIFCell(; name="layer1", n_units=4, tau_m=10.0)
set!(cell.j, [120.0 80.0 200.0 40.0])   # drive with input current
advance_state!(cell, 1.0, 1.0)          # (dt, t)
get_value(cell.s)                        # emitted spikes
```

## A predictive-coding network

The `PCN` model wires the component zoo into the `pc_discrim` exhibit and trains
it with the full PEM loop (project → settle → Hebbian-Adam update):

```julia
using NGCLearn

m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, eta=0.002, key=7)

x = reshape(Float64[1, 1, 0, 0], 1, 4)   # one observation
y = reshape(Float64[1, 0], 1, 2)         # one label

y_inf, y_mu, EFE = process!(m, x, y)     # one training step
pred = project(m, x)                      # test-time inference (no learning)
```

See `examples/02_pc_discrim_train.jl` for a runnable end-to-end training loop,
and `docs/specs/` for the per-component port specs.
