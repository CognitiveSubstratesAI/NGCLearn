# Port spec: `LIFCell`

Upstream: `ngclearn/components/neurons/spiking/LIFCell.py`
Port: `src/components/neurons/spiking/lif_cell.jl`

## Dynamics

    tau_m * dv/dt = (v_rest - v) * g_L + j * resist_m
    spike  s = 1[v > thr + thr_theta]
    on spike:  v <- v_reset,  rfr <- 0,  tols <- t
    thr_theta <- thr_theta * exp(-dt/tau_theta) + s * theta_plus   (if tau_theta > 0)

`g_L = conduct_leak`: 1 ⇒ LIF, 0 ⇒ pure integrate-and-fire.

## Compartments

| kind | name | init |
|---|---|---|
| input | `j` | zeros `(1, n_units)` |
| state | `v` | `v_rest` |
| state | `rfr` | `refract_T` |
| state | `thr_theta` | 0 |
| state | `key` | PRNG seed (`UInt64`) |
| output | `s`, `s_raw` | 0 |
| output | `tols` | 0 |

## `advance_state!(c, dt, t)` — maps LIFCell.py:152-205

1. `j = get(c.j) * resist_m`; `_v_thr = get(thr_theta) + thr`.
2. integrate `v` one step via `step_rk2` (code 1) or `step_euler` (else), with
   `_dfv_lif` and `params = (j, rfr, tau_m, refract_T, v_rest, g_L)`.
3. `s = 1[_v > _v_thr]`; `_rfr = (rfr+dt)*(1-s)`; `_v = _v*(1-s) + s*v_reset`.
4. optional `one_spike` (stochastic single-spike) / `max_one_spike` (max-volt
   spike) masking.
5. optional adaptive-threshold Euler step (`tau_theta > 0`).
6. `tols = (1-s)*tols + s*t`; optional `v_min` clamp.
7. write back `v, s, s_raw, rfr`.

## `reset_state!(c)` — maps LIFCell.py:208-216

Reset all compartments to init; `j` reset only if not externally `targeted`.

## Divergences (see `docs/decisions.md`)

- **#3** — `tau_m` is a scalar field `c.tau_m`. Upstream `self.tau_m.get()`
  (LIFCell.py:171) is a bug: `tau_m` is a bare float and every sibling cell uses
  the scalar. Fixed in the port.
- Commented-out surrogate-function setup in upstream `__init__` is omitted (dead
  code there too — `surrogate_type` is accepted but never wired into emission).
- PRNG: `UInt64` seed via `Xoshiro`, not a JAX key array (decisions #2).

## Verification

`test/test_lif_cell.jl`, eager path, hand-computed: construction/shapes;
sub-threshold integration (`v: -65 → -55`, no spike); supra-threshold spike
(`v → v_reset`, `rfr → 0`, `tols → t`); adaptive-threshold bump; rk2 selection;
`reset_state!`; `v_min` floor.
