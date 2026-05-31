# lif_cell.jl — leaky integrate-and-fire spiking neuron.
#
# 1:1 port of ngclearn/components/neurons/spiking/LIFCell.py.
#
# Dynamics (per the upstream docstring):
#   tau_m * dv/dt = (v_rest - v) * g_L + j * resist_m
#   if v > threshold:  emit spike, v <- v_reset
# with an optional homeostatic adaptive threshold `thr_theta` that decays toward
# zero and is bumped by `theta_plus` on each spike.
#
# Spec: docs/specs/02_lif_cell.md.
#
# Upstream divergences (see docs/decisions.md #3):
#   - `tau_m` is a SCALAR hyperparameter, accessed as `c.tau_m`. Upstream
#     LIFCell.py:171 calls `self.tau_m.get()`, but `tau_m` is assigned a plain
#     float in `__init__` (LIFCell.py:124) and EVERY sibling cell (IFCell,
#     adExCell, fitzhughNagumoCell, WTASCell) uses the bare scalar. The `.get()`
#     is an upstream bug; the port uses the scalar directly.
#   - The commented-out surrogate-function setup in upstream `__init__` is
#     omitted (it is dead code there too — `surrogate_type` is accepted but the
#     estimator is never wired into spike emission).

# ── Module-level ODE / threshold co-routines (free functions) ─────────────────

# Voltage dynamics dv/dt. Mirrors `_dfv` (LIFCell.py:11-18).
# params = (j, rfr, tau_m, refract_T, v_rest, g_L)
function _dfv_lif(t, v, params)
    j, rfr, tau_m, refract_T, v_rest, g_L = params
    mask = (rfr .>= refract_T) .* 1.0   # refractory mask: 1 once refractory period elapsed
    dv_dt = (v_rest .- v) .* g_L .+ (j .* mask)
    dv_dt = dv_dt .* (1.0 / tau_m)
    return dv_dt
end

# One Euler step of the homeostatic threshold. Mirrors `_update_theta`
# (LIFCell.py:22-30).
function _update_theta(dt, v_theta, s, tau_theta, theta_plus=0.05)
    theta_decay = exp(-dt / tau_theta)
    return v_theta .* theta_decay .+ s .* theta_plus
end

# Per-row one-hot of the argmax along dim 2 (the analog of
# `nn.one_hot(jnp.argmax(m, axis=1), num_classes=size(m, 2))`).
function _row_one_hot(m::AbstractMatrix)
    out = zeros(eltype(m), size(m))
    @inbounds for i in axes(m, 1)
        out[i, argmax(@view m[i, :])] = one(eltype(m))
    end
    return out
end

# ── Component type ────────────────────────────────────────────────────────────

"""
    LIFCell <: JaxComponent

A spiking cell governed by leaky integrate-and-fire dynamics.

| Input compartments:  `j` (electrical current)
| State compartments:  `v` (voltage), `rfr` (refractory), `thr_theta`
|                      (adaptive threshold), `key` (PRNG seed)
| Output compartments: `s` (spikes), `s_raw` (pre-processing spikes),
|                      `tols` (time-of-last-spike)

Construct with the keyword constructor [`LIFCell(; ...)`](@ref). Required:
`name`, `n_units`, `tau_m`.
"""
mutable struct LIFCell <: JaxComponent
    # Standard component fields (see jax_component.jl, decisions.md #1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters (scalars / config — NOT compartments).
    n_units::Int
    batch_size::Int
    tau_m::Float64
    resist_m::Float64
    thr::Float64
    v_rest::Float64
    v_reset::Float64
    g_L::Float64           # conduct_leak: 1 => LIF, 0 => pure IF
    tau_theta::Float64
    theta_plus::Float64
    refract_T::Float64     # refract_time (ms)
    one_spike::Bool
    max_one_spike::Bool
    v_min::Union{Float64, Nothing}
    integration_type::String
    intg_flag::Int

    # State / I-O compartments.
    key::NGCSimLib.Compartment
    j::NGCSimLib.Compartment
    v::NGCSimLib.Compartment
    s::NGCSimLib.Compartment
    s_raw::NGCSimLib.Compartment
    rfr::NGCSimLib.Compartment
    thr_theta::NGCSimLib.Compartment
    tols::NGCSimLib.Compartment
end

"""
    LIFCell(; name, n_units, tau_m, resist_m=1.0, thr=-52.0, v_rest=-65.0,
            v_reset=-60.0, conduct_leak=1.0, tau_theta=1e7, theta_plus=0.05,
            refract_time=5.0, one_spike=false, max_one_spike=false,
            integration_type="euler", v_min=nothing, key=nothing,
            context_path="", args=Any[], kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `LIFCell.__init__` (LIFCell.py:110-150). Sets up
all eight compartments at `(batch_size, n_units)` shape with `batch_size = 1`.
"""
function LIFCell(;
    name::AbstractString,
    n_units::Integer,
    tau_m::Real,
    resist_m::Real=1.0,
    thr::Real=-52.0,
    v_rest::Real=-65.0,
    v_reset::Real=-60.0,
    conduct_leak::Real=1.0,
    tau_theta::Real=1e7,
    theta_plus::Real=0.05,
    refract_time::Real=5.0,
    one_spike::Bool=false,
    max_one_spike::Bool=false,
    integration_type::AbstractString="euler",
    v_min::Union{Real, Nothing}=nothing,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    resist_m > 0.0 || error("LIFCell: resist_m must be > 0 (got $resist_m)")
    batch_size = 1
    rest = zeros(Float64, batch_size, n_units)

    return LIFCell(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        batch_size,
        Float64(tau_m),
        Float64(resist_m),
        Float64(thr),
        Float64(v_rest),
        Float64(v_reset),
        Float64(conduct_leak),
        Float64(tau_theta),
        Float64(theta_plus),
        Float64(refract_time),
        one_spike,
        max_one_spike,
        v_min === nothing ? nothing : Float64(v_min),
        String(integration_type),
        get_integrator_code(integration_type),
        NGCSimLib.Compartment(make_prng_key(key)),
        NGCSimLib.Compartment(copy(rest); display_name="Current", units="mA"),
        NGCSimLib.Compartment(rest .+ v_rest; display_name="Voltage", units="mV"),
        NGCSimLib.Compartment(copy(rest); display_name="Spikes"),
        NGCSimLib.Compartment(copy(rest); display_name="Raw Spike Pulses"),
        NGCSimLib.Compartment(
            rest .+ refract_time; display_name="Refractory Time Period", units="ms"
        ),
        NGCSimLib.Compartment(
            copy(rest); display_name="Threshold Adaptive Shift", units="mV"
        ),
        NGCSimLib.Compartment(copy(rest); display_name="Time-of-Last-Spike", units="ms")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (LIFCell.py:152-205). Marked `@compilable` so the
# method is registered for the Parser; eager dispatch (used by the tests and the
# ground-truth path) works regardless.
NGCSimLib.@compilable function advance_state!(c::LIFCell, dt, t)
    j = NGCSimLib.get_value(c.j) .* c.resist_m
    _v_thr = NGCSimLib.get_value(c.thr_theta) .+ c.thr

    v_params = (j, NGCSimLib.get_value(c.rfr), c.tau_m, c.refract_T, c.v_rest, c.g_L)
    if c.intg_flag == 1
        _, _v = step_rk2(0.0, NGCSimLib.get_value(c.v), _dfv_lif, dt, v_params)
    else
        _, _v = step_euler(0.0, NGCSimLib.get_value(c.v), _dfv_lif, dt, v_params)
    end

    s = (_v .> _v_thr) .* 1.0
    _rfr = (NGCSimLib.get_value(c.rfr) .+ dt) .* (1.0 .- s)
    _v = _v .* (1.0 .- s) .+ s .* c.v_reset
    raw_s = s

    if c.one_spike && !c.max_one_spike
        # Stochastically keep a single spike when more than one fired.
        rng = Xoshiro(NGCSimLib.get_value(c.key))
        m_switch = (sum(s) > 0.0) ? 1.0 : 0.0
        rS = s .* rand(rng, Float64, size(s))
        rS = _row_one_hot(rS)
        s = s .* (1.0 - m_switch) .+ rS .* m_switch
        NGCSimLib.set!(c.key, rand(rng, UInt64))
    end

    if c.max_one_spike
        # Keep only the spike at the maximum-voltage unit.
        rS = _row_one_hot(NGCSimLib.get_value(c.v))
        s = s .* rS
    end

    if c.tau_theta > 0.0
        thr_theta = _update_theta(
            dt, NGCSimLib.get_value(c.thr_theta), raw_s, c.tau_theta, c.theta_plus
        )
        NGCSimLib.set!(c.thr_theta, thr_theta)
    end

    NGCSimLib.set!(c.tols, (1.0 .- s) .* NGCSimLib.get_value(c.tols) .+ (s .* t))

    if c.v_min !== nothing
        _v = max.(_v, c.v_min)
    end

    NGCSimLib.set!(c.v, _v)
    NGCSimLib.set!(c.s, s)
    NGCSimLib.set!(c.s_raw, raw_s)
    NGCSimLib.set!(c.rfr, _rfr)
    return c
end

# Mirrors `reset` (LIFCell.py:208-216). The input compartment `j` is only reset
# when it is not externally wired (`targeted`).
NGCSimLib.@compilable function reset_state!(c::LIFCell)
    rest = zeros(Float64, c.batch_size, c.n_units)
    if !NGCSimLib.targeted(c.j)
        NGCSimLib.set!(c.j, copy(rest))
    end
    NGCSimLib.set!(c.v, rest .+ c.v_rest)
    NGCSimLib.set!(c.s, copy(rest))
    NGCSimLib.set!(c.s_raw, copy(rest))
    NGCSimLib.set!(c.rfr, rest .+ c.refract_T)
    NGCSimLib.set!(c.tols, copy(rest))
    return c
end
