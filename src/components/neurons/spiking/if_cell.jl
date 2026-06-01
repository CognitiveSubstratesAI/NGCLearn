# if_cell.jl — integrate-and-fire spiking neuron.
#
# 1:1 port of ngclearn/components/neurons/spiking/IFCell.py.
#
# A simplification of LIFCell with NO leak term: the membrane integrates input
# current directly.
#   tau_m * dv/dt = j * resist_m        (gated by the refractory mask)
#   spike s = 1[v > thr];  on spike: v <- v_reset, rfr <- 0, tols <- t
# Optional `lower_clamp_voltage` floors v at v_rest.
#
# Compared to LIFCell: no (v_rest - v) leak, no homeostatic adaptive threshold,
# no one_spike/max_one_spike, no PRNG/key, no s_raw compartment.
#
# Spec: docs/specs/10_if_cell.md.
# Decisions: NGCLearn §1 (no @ngc_component), §3 (scalar hyperparams), §4 (EAGER
# is ground truth).

# Voltage dynamics dv/dt — current-only (no leak). Mirrors `_dfv` (IFCell.py:22).
# params = (j, rfr, tau_m, refract_T)
function _dfv_if(t, v, params)
    j, rfr, tau_m, refract_T = params
    mask = (rfr .>= refract_T) .* 1.0     # refractory mask: 1 once period elapsed
    return (j .* mask) .* (1.0 / tau_m)
end

"""
    IFCell <: JaxComponent

A spiking cell governed by integrate-and-fire dynamics (LIF without the leak):
`tau_m * dv/dt = j * resist_m`, gated by a refractory mask.

| Input compartments:  `j` (electrical current)
| State compartments:  `v` (voltage), `rfr` (refractory)
| Output compartments: `s` (spikes), `tols` (time-of-last-spike)

Construct with [`IFCell(; ...)`](@ref). Required: `name`, `n_units`, `tau_m`.
"""
mutable struct IFCell <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol,Any}

    # Hyperparameters (scalars).
    n_units::Int
    batch_size::Int
    tau_m::Float64
    resist_m::Float64
    thr::Float64
    v_rest::Float64
    v_reset::Float64
    refract_T::Float64
    lower_clamp_voltage::Bool
    integration_type::String
    intg_flag::Int

    # Compartments.
    j::NGCSimLib.Compartment
    v::NGCSimLib.Compartment
    s::NGCSimLib.Compartment
    rfr::NGCSimLib.Compartment
    tols::NGCSimLib.Compartment
end

"""
    IFCell(; name, n_units, tau_m, resist_m=1.0, thr=-52.0, v_rest=-65.0,
           v_reset=-60.0, refract_time=0.0, integration_type="euler",
           lower_clamp_voltage=true, context_path="", args=Any[],
           kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `IFCell.__init__` (IFCell.py:90-135). `batch_size`
is fixed at 1; all five compartments are `(1, n_units)`.
"""
function IFCell(;
    name::AbstractString,
    n_units::Integer,
    tau_m::Real,
    resist_m::Real=1.0,
    thr::Real=-52.0,
    v_rest::Real=-65.0,
    v_reset::Real=-60.0,
    refract_time::Real=0.0,
    integration_type::AbstractString="euler",
    lower_clamp_voltage::Bool=true,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}()
)
    resist_m > 0.0 || error("IFCell: resist_m must be > 0 (got $resist_m)")
    batch_size = 1
    rest = zeros(Float64, batch_size, n_units)
    return IFCell(
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
        Float64(refract_time),
        lower_clamp_voltage,
        String(integration_type),
        get_integrator_code(integration_type),
        NGCSimLib.Compartment(copy(rest); display_name="Current", units="mA"),
        NGCSimLib.Compartment(rest .+ v_rest; display_name="Voltage", units="mV"),
        NGCSimLib.Compartment(copy(rest); display_name="Spikes"),
        NGCSimLib.Compartment(
            rest .+ refract_time; display_name="Refractory Time Period", units="ms"
        ),
        NGCSimLib.Compartment(copy(rest); display_name="Time-of-Last-Spike", units="ms")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (IFCell.py:137-167).
NGCSimLib.@compilable function advance_state!(c::IFCell, dt, t)
    j = NGCSimLib.get_value(c.j) .* c.resist_m
    v_params = (j, NGCSimLib.get_value(c.rfr), c.tau_m, c.refract_T)
    if c.intg_flag == 1
        _, _v = step_rk2(0.0, NGCSimLib.get_value(c.v), _dfv_if, dt, v_params)
    else
        _, _v = step_euler(0.0, NGCSimLib.get_value(c.v), _dfv_if, dt, v_params)
    end

    s = (_v .> c.thr) .* 1.0
    rfr = (NGCSimLib.get_value(c.rfr) .+ dt) .* (1.0 .- s)
    v = _v .* (1.0 .- s) .+ s .* c.v_reset

    NGCSimLib.set!(c.tols, (1.0 .- s) .* NGCSimLib.get_value(c.tols) .+ (s .* t))
    if c.lower_clamp_voltage
        v = max.(v, c.v_rest)
    end

    NGCSimLib.set!(c.v, v)
    NGCSimLib.set!(c.s, s)
    NGCSimLib.set!(c.rfr, rfr)
    return c
end

# Mirrors `reset` (IFCell.py:169-177). `j` reset only if not externally wired.
NGCSimLib.@compilable function reset_state!(c::IFCell)
    rest = zeros(Float64, c.batch_size, c.n_units)
    if !NGCSimLib.targeted(c.j)
        NGCSimLib.set!(c.j, copy(rest))
    end
    NGCSimLib.set!(c.v, rest .+ c.v_rest)
    NGCSimLib.set!(c.s, copy(rest))
    NGCSimLib.set!(c.rfr, rest .+ c.refract_T)
    NGCSimLib.set!(c.tols, copy(rest))
    return c
end
