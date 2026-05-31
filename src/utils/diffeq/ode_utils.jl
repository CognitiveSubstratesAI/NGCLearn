# ode_utils.jl — ordinary-differential-equation integration backend.
#
# 1:1 port of ngclearn/utils/diffeq/ode_utils.py (the subset used by the Phase-A
# component zoo). Upstream supports five back-ends; we port the two exercised by
# the first components (LIFCell): Euler (RK-1) and the midpoint method (RK-2).
# The remaining codes (Heun, Ralston, RK-4) are reserved by `get_integrator_code`
# and will be filled in as the components that need them are ported.
#
# Spec: docs/specs/01_ode_utils.md.
#
# Design notes:
#   - Upstream JIT-compiles each step with `@partial(jit, static_argnums=...)`.
#     We keep these as plain broadcasting Julia functions; Reactant tracing
#     (the Julia analog of jax.jit) is applied at the Process layer, not here.
#   - `x_scale` is a positional-with-default in upstream; we expose it as a
#     keyword for clarity. The `dfx` co-routine signature `(t, x, params)` is
#     preserved exactly so component `_dfv`-style functions port unchanged.

"""
    get_integrator_code(integration_type::AbstractString) -> Int

Map an integrator-type string to ngc-learn's internal integer code.

| code | methods                                   |
|:----:|:------------------------------------------|
| 0    | `"euler"` / `"rk1"` (default / fallback)  |
| 1    | `"midpoint"` / `"rk2"`                     |
| 2    | `"rk2_heun"` / `"heun"`                     |
| 3    | `"rk2_ralston"` / `"ralston"`               |
| 4    | `"rk4"`                                     |

Mirrors `get_integrator_code` (ode_utils.py:18-50). Any unrecognized string
falls back to Euler (code 0), matching upstream.
"""
function get_integrator_code(integration_type::AbstractString)
    if integration_type == "midpoint" || integration_type == "rk2"
        return 1
    elseif integration_type == "rk2_heun" || integration_type == "heun"
        return 2
    elseif integration_type == "rk2_ralston" || integration_type == "ralston"
        return 3
    elseif integration_type == "rk4"
        return 4
    else
        return 0  # Euler / RK-1 / default
    end
end

# Internal single-step advance, shared by every explicit method.
# Mirrors `_step_forward` (ode_utils.py:54-57):
#   _t = t + dt
#   _x = x * x_scale + dx_dt * dt
@inline function _step_forward(t, x, dx_dt, dt, x_scale)
    _t = t + dt
    _x = x .* x_scale .+ dx_dt .* dt
    return _t, _x
end

"""
    step_euler(t, x, dfx, dt, params; x_scale=1.0) -> (t_next, x_next)

One Euler (first-order Runge-Kutta, RK-1) integration step of `dx/dt = dfx(t, x, params)`.

`dfx` is the component-provided ODE co-routine with signature `(t, x, params)`.
Mirrors `step_euler` / `_euler` (ode_utils.py:59-114).
"""
function step_euler(t, x, dfx, dt, params; x_scale=1.0)
    dx_dt = dfx(t, x, params)
    return _step_forward(t, x, dx_dt, dt, x_scale)
end

"""
    step_rk2(t, x, dfx, dt, params; x_scale=1.0) -> (t_next, x_next)

One midpoint-method (second-order Runge-Kutta, RK-2) integration step. More
accurate than Euler at the cost of a second `dfx` evaluation.

Mirrors `step_rk2` / `_rk2` (ode_utils.py:218-289):
    f1       = dfx(t, x, params)
    (t1, x1) = step(t, x, f1, dt/2)
    f2       = dfx(t1, x1, params)
    (_t, _x) = step(t, x, f2, dt)
"""
function step_rk2(t, x, dfx, dt, params; x_scale=1.0)
    f1 = dfx(t, x, params)
    t1, x1 = _step_forward(t, x, f1, dt * 0.5, x_scale)
    f2 = dfx(t1, x1, params)
    return _step_forward(t, x, f2, dt, x_scale)
end
