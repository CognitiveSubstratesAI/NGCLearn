# Port spec: `ode_utils` (integration backend)

Upstream: `ngclearn/utils/diffeq/ode_utils.py`
Port: `src/utils/diffeq/ode_utils.jl`

## Scope

The integration backend ships in lockstep with the components that need it.
Phase A ports the subset `LIFCell` uses: `get_integrator_code`, `step_euler`
(RK-1), `step_rk2` (midpoint), and the shared `_step_forward`.

## Function map

| Upstream (py) | Port (jl) | Notes |
|---|---|---|
| `get_integrator_code(s)` (L18-50) | `get_integrator_code(s)` | string → int code; unknown ⇒ 0 (Euler) |
| `_step_forward(t,x,dx_dt,dt,x_scale)` (L54-57) | `_step_forward` | `_x = x*x_scale + dx_dt*dt`; broadcast |
| `step_euler` / `_euler` (L59-114) | `step_euler(t,x,dfx,dt,params; x_scale=1.0)` | one RK-1 step |
| `step_rk2` / `_rk2` (L218-289) | `step_rk2(...)` | midpoint: 2 `dfx` evals |

## Integrator codes (`get_integrator_code`)

| code | strings |
|:----:|---|
| 0 | `euler`, `rk1`, anything unrecognized (default) |
| 1 | `midpoint`, `rk2` |
| 2 | `rk2_heun`, `heun` |
| 3 | `rk2_ralston`, `ralston` |
| 4 | `rk4` |

Codes 2–4 are reserved (returned by `get_integrator_code`) but their step
functions are **not yet ported** — they land with the first component that
selects them.

## Divergences

- Upstream wraps `_euler`/`_rk2` in `@partial(jit, static_argnums=...)`. We keep
  plain broadcasting functions; the JIT (Reactant tracing) is applied at the
  Process layer, not inside the integrator. See `docs/decisions.md` #4.
- `x_scale` is positional-with-default upstream; exposed as a keyword here. The
  `dfx(t, x, params)` co-routine signature is preserved exactly.

## Verification

`test/test_ode_utils.jl`: code mapping; Euler exactness on `dx/dt = const`;
midpoint exact on `dx/dt = t` (where Euler provably undershoots); array
broadcasting.
