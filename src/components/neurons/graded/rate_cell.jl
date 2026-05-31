# rate_cell.jl — rate-coded / continuous-state graded neuron.
#
# 1:1 port of ngclearn/components/neurons/graded/rateCell.py.
#
# Dynamics (per the upstream docstring, RateCell.py:107-116):
#   tau_m * dz/dt = -leak_gamma * prior(z) + (j + j_td)
#
# where `prior(z)` is one of {gaussian, laplacian, cauchy, exp} — a centred
# scale-shift distribution whose *derivative* enters the leak term:
#
#   gaussian  z_leak = z                              (default)
#   laplacian z_leak = sign(z)
#   cauchy    z_leak = (2z) / (1 + z^2)
#   exp       z_leak = exp(-z^2) * z * 2
#
# Then `zF = fx(z) * output_scale` is the post-activation output.
#
# Stateless mode: when `tau_m <= 0`, integration is skipped — `z = j + j_td` —
# matching upstream's `_run_cell_stateless` short-circuit.
#
# Spec: docs/specs/04_rate_cell.md.
# Decisions: NGCSimLib §3 (no Base.get shadow); NGCLearn §1 (no @ngc_component),
# §3 (scalar hyperparams), §4 (EAGER is ground truth).

# ── Module-level ODE co-routines ─────────────────────────────────────────────
# `(t, z, params)` signature kept identical to ode_utils.step_*; `params` packs
# `(j, j_td, tau_m, leak_gamma)`.

# Gaussian prior: z_leak = z. Mirrors `_dfz_internal_gaussian` (RateCell.py:30).
function _dfz_gaussian(t, z, params)
    j, j_td, tau_m, leak_gamma = params
    return (.-z .* leak_gamma .+ (j .+ j_td)) .* (1.0 / tau_m)
end

# Laplacian prior: z_leak = sign(z). Mirrors `_dfz_internal_laplace`
# (RateCell.py:15).
function _dfz_laplace(t, z, params)
    j, j_td, tau_m, leak_gamma = params
    return (.-sign.(z) .* leak_gamma .+ (j .+ j_td)) .* (1.0 / tau_m)
end

# Cauchy prior: z_leak = (2z) / (1 + z^2). Mirrors `_dfz_internal_cauchy`
# (RateCell.py:20).
function _dfz_cauchy(t, z, params)
    j, j_td, tau_m, leak_gamma = params
    z_leak = (z .* 2.0) ./ (1.0 .+ z .* z)
    return (.-z_leak .* leak_gamma .+ (j .+ j_td)) .* (1.0 / tau_m)
end

# Exponential prior: z_leak = exp(-z^2) * z * 2. Mirrors
# `_dfz_internal_exp` (RateCell.py:25).
function _dfz_exp(t, z, params)
    j, j_td, tau_m, leak_gamma = params
    z_leak = exp.(.-z .* z) .* z .* 2.0
    return (.-z_leak .* leak_gamma .+ (j .+ j_td)) .* (1.0 / tau_m)
end

# Prior-type → ODE-rhs lookup. Mirrors `_dfz_fns` (RateCell.py:75-81).
const _DFZ_FNS = (
    _dfz_gaussian,  # 0
    _dfz_laplace,   # 1
    _dfz_cauchy,    # 2
    _dfz_exp       # 3
)

# Apply the modulator `dfx_val` to the current `j`. Mirrors `_modulate`
# (RateCell.py:36).
@inline _modulate(j, dfx_val) = j .* dfx_val

# Prior-name → priorType integer. Mirrors `priorTypeDict` (RateCell.py:176).
function _prior_type_int(name::AbstractString)
    if name == "gaussian"
        return 0
    elseif name == "laplacian"
        return 1
    elseif name == "cauchy"
        return 2
    elseif name == "exp"
        return 3
    else
        return 0  # default to gaussian (upstream's `.get(name, 0)` fallback)
    end
end

# ── Component type ───────────────────────────────────────────────────────────

"""
    RateCell <: JaxComponent

A non-spiking, rate-coded graded cell with leaky scale-shift prior dynamics:

| `tau_m * dz/dt = -gamma * prior'(z) + (j + j_td)`
|  then  `zF = fx(z) * output_scale`

| Input compartments:  `j` (bottom-up current), `j_td` (top-down pressure)
| State compartments:  `z` (rate activity)
| Output compartments: `zF` (post-activation output)

Stateless mode is triggered when `tau_m <= 0`: integration is skipped and
`z = j + j_td` (matches upstream's `_run_cell_stateless`).

Construct with the keyword constructor [`RateCell(; ...)`](@ref). Required:
`name`, `n_units`, `tau_m`.
"""
mutable struct RateCell <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters (scalars — decisions.md §3).
    n_units::Int
    batch_size::Int
    tau_m::Float64
    is_stateful::Bool
    output_scale::Float64
    prior_type::Int           # 0=gaussian, 1=laplacian, 2=cauchy, 3=exp
    prior_leak_rate::Float64  # `gamma` in the docstring
    threshold_type::String    # "none" | "soft_threshold" | "cauchy_threshold"
    thr_lmbda::Float64
    resist_scale::Float64
    integration_type::String
    intg_flag::Int
    # `(fx, dfx)` from `create_function`. Plain function values; the receiver
    # accesses them as `c.fx`/`c.dfx`.
    fx::Function
    dfx::Function

    # State / I-O compartments. All allocated at `(batch_size, n_units)` =
    # zeros at construction time. NB: only `j`, `j_td`, `z`, `zF` are
    # registered Compartments — `fx`/`dfx` are bare function fields.
    j::NGCSimLib.Compartment
    j_td::NGCSimLib.Compartment
    z::NGCSimLib.Compartment
    zF::NGCSimLib.Compartment
end

"""
    RateCell(; name, n_units, tau_m, prior=("gaussian", 0.0),
              act_fx="identity", output_scale=1.0, threshold=("none", 0.0),
              integration_type="euler", batch_size=1, resist_scale=1.0,
              is_stateful=true, context_path="", args=Any[],
              kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `RateCell.__init__` (RateCell.py:161-213).

Hyperparameter defaults track upstream verbatim. `prior` is a `(name, leak_rate)`
tuple — supported names: `"gaussian"`, `"laplacian"`, `"cauchy"`, `"exp"`.
`threshold` is a `(kind, lmbda)` tuple — supported kinds: `"none"`,
`"soft_threshold"`, `"cauchy_threshold"`.

Per upstream, `tau_m <= 0.0` switches the cell to stateless mode
(`z = j + j_td`, no integration).
"""
function RateCell(;
    name::AbstractString,
    n_units::Integer,
    tau_m::Real,
    prior::Tuple{<:AbstractString, <:Real}=("gaussian", 0.0),
    act_fx::AbstractString="identity",
    output_scale::Real=1.0,
    threshold::Tuple{<:AbstractString, <:Real}=("none", 0.0),
    integration_type::AbstractString="euler",
    batch_size::Integer=1,
    resist_scale::Real=1.0,
    is_stateful::Bool=true,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    # Upstream RateCell.py:172-174: `tau_m <= 0` overrides to stateless.
    _is_stateful = is_stateful && !(tau_m isa Real && Float64(tau_m) <= 0.0)

    prior_name, leak_rate = prior
    thr_kind, thr_lmbda = threshold
    fx, dfx = create_function(String(act_fx))

    rest = zeros(Float64, batch_size, n_units)

    return RateCell(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        Int(batch_size),
        Float64(tau_m),
        _is_stateful,
        Float64(output_scale),
        _prior_type_int(String(prior_name)),
        Float64(leak_rate),
        String(thr_kind),
        Float64(thr_lmbda),
        Float64(resist_scale),
        String(integration_type),
        get_integrator_code(integration_type),
        fx,
        dfx,
        NGCSimLib.Compartment(
            copy(rest); display_name="Input Stimulus Current", units="mA"
        ),
        NGCSimLib.Compartment(
            copy(rest); display_name="Modulatory Stimulus Current", units="mA"
        ),
        NGCSimLib.Compartment(copy(rest); display_name="Rate Activity", units="mA"),
        NGCSimLib.Compartment(copy(rest); display_name="Transformed Rate Activity")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (RateCell.py:216-252). Marked `@compilable` so the
# method is registered for the Parser; eager dispatch is the ground-truth path
# until the NGCSimLib parser learns to inline scalar hyperparameters
# (decisions.md §4).
NGCSimLib.@compilable function advance_state!(c::RateCell, dt)
    j = NGCSimLib.get_value(c.j)
    j_td = NGCSimLib.get_value(c.j_td)
    z = NGCSimLib.get_value(c.z)

    if c.is_stateful
        # Modulate, scale, integrate one step.
        dfx_val = c.dfx(z)
        j = _modulate(j, dfx_val)
        j = j .* c.resist_scale

        dfz_fn = _DFZ_FNS[c.prior_type + 1]    # +1: 0-indexed → 1-indexed
        params = (j, j_td, c.tau_m, c.prior_leak_rate)
        _, z_new = if (c.intg_flag == 1)
            step_rk2(0.0, z, dfz_fn, dt, params)
        else
            step_euler(0.0, z, dfz_fn, dt, params)
        end

        # Optional thresholding sub-dynamics.
        if c.threshold_type == "soft_threshold"
            z_new = threshold_soft(z_new, c.thr_lmbda)
        elseif c.threshold_type == "cauchy_threshold"
            z_new = threshold_cauchy(z_new, c.thr_lmbda)
        end
        z = z_new
        zF = c.fx(z) .* c.output_scale
    else
        # Stateless mode: passthrough sum. Mirrors `_run_cell_stateless`
        # (RateCell.py:93-104) + RateCell.py:243-246.
        j_total = j .+ j_td
        z = j_total .+ 0.0           # upstream returns `j + 0` (a copy)
        zF = c.fx(z) .* c.output_scale
    end

    NGCSimLib.set!(c.j, j)
    NGCSimLib.set!(c.j_td, j_td)
    NGCSimLib.set!(c.z, z)
    NGCSimLib.set!(c.zF, zF)
    return c
end

# Mirrors `reset` (RateCell.py:254-263).
NGCSimLib.@compilable function reset_state!(c::RateCell)
    rest = zeros(Float64, c.batch_size, c.n_units)
    NGCSimLib.set!(c.j, copy(rest))
    NGCSimLib.set!(c.j_td, copy(rest))
    NGCSimLib.set!(c.z, copy(rest))
    NGCSimLib.set!(c.zF, copy(rest))
    return c
end
