# Port spec: `RateCell`

Upstream: `ngclearn/components/neurons/graded/rateCell.py`
Port: `src/components/neurons/graded/rate_cell.jl`

## Dynamics (rate-coded, leaky scale-shift prior)

    tau_m * dz/dt = -gamma * prior'(z) + (j + j_td)
    then  zF = fx(z) * output_scale

where `prior'(z)` is the derivative of a centred scale-shift distribution that
enters the leak term:

| prior | `z_leak = prior'(z)` |
|---|---|
| `gaussian` (default) | `z` |
| `laplacian` | `sign(z)` |
| `cauchy` | `(2z) / (1 + z^2)` |
| `exp` | `exp(-z^2) * z * 2` |

Stateless mode: when `tau_m <= 0`, integration is skipped and `z = j + j_td`
(mirrors upstream `_run_cell_stateless`). The bottom-up current `j` is first
modulated by `dfx(z)` and scaled by `resist_scale` before integration.

## Compartments

| kind | name | init |
|---|---|---|
| input | `j` | zeros `(batch, n_units)` (bottom-up current) |
| input | `j_td` | zeros (top-down pressure) |
| state | `z` | zeros (rate activity) |
| output | `zF` | zeros (post-activation output) |

`fx`/`dfx` are bare `Function` struct fields (from `create_function`), not
compartments; `tau_m`, `prior_leak_rate` (`gamma`), `output_scale`,
`resist_scale`, `threshold_type`/`thr_lmbda` are scalar hyperparameters.

## `advance_state!(c, dt)` — maps RateCell.py:216-252

Stateful path: modulate `j` by `dfx(z)`, scale by `resist_scale`, integrate one
step via `step_rk2` (intg_flag 1) or `step_euler` (else) with the prior-selected
ODE rhs, apply optional soft/cauchy thresholding, then `zF = fx(z)*output_scale`.
Stateless path (`tau_m<=0`): `z = j + j_td`, `zF = fx(z)*output_scale`.

## `reset_state!(c)` — maps RateCell.py:254-263

Zero `j`, `j_td`, `z`, `zF`.

## Divergences (see `docs/decisions.md`)

- **#1** explicit fields, no `@ngc_component`. **#3** scalar hyperparameters stay
  scalar. **#4** eager path is the ground truth.
- Prior name → integer code via `_prior_type_int` (unknown ⇒ gaussian, matching
  upstream's `.get(name, 0)` fallback).

## Verification

`test/test_rate_cell.jl`, eager path: construction/shape defaults; Euler gaussian
zero-leak; gaussian leak; laplacian prior; stateless mode; `output_scale` scales
`zF` not `z`; tanh `dfx` modulation; soft-threshold; `reset_state!`; rk2 path;
`create_function` rejects unknown activation.
