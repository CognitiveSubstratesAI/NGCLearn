# var_trace.jl — variable trace (low-pass filter) over an input signal.
#
# 1:1 port of ngclearn/components/other/varTrace.py.
#
# Maintains a decaying trace of incoming (typically spike) signals:
#   decay = exp(-dt/tau_tr)   ("exp")  |  1 - dt/tau_tr  ("lin")  |  0  ("step")
#   x_tr  = gamma_tr * trace * decay
# then one of:
#   - nearest-neighbor (n_nearest_spikes>0): x_tr += inputs*(a_delta - trace/k)
#   - additive (a_delta>0):                  x_tr += inputs * a_delta
#   - gated snap (else):                     x_tr = x_tr*(1-inputs) + inputs*P_scale
#
# Spec: docs/specs/08_var_trace.md.
# Decisions: NGCLearn §1 (no @ngc_component), §4 (EAGER is ground truth).

"""
    VarTrace <: JaxComponent

A low-pass filter node that maintains a decaying `trace` of its `inputs` (a
spike-trace accumulator, consumed by trace-based STDP synapses).

| Input compartments:  `inputs`
| State compartments:  `trace`
| Output compartments: `outputs` (== `trace`)

Construct with [`VarTrace(; ...)`](@ref). Required: `name`, `n_units`, `tau_tr`,
`a_delta`.
"""
mutable struct VarTrace <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters (scalars / config).
    n_units::Int
    batch_size::Int
    tau_tr::Float64
    a_delta::Float64
    P_scale::Float64
    gamma_tr::Float64
    n_nearest_spikes::Int
    decay_type::String

    # Compartments.
    key::NGCSimLib.Compartment
    inputs::NGCSimLib.Compartment
    outputs::NGCSimLib.Compartment
    trace::NGCSimLib.Compartment
end

"""
    VarTrace(; name, n_units, tau_tr, a_delta, P_scale=1.0, gamma_tr=1.0,
             decay_type="exp", n_nearest_spikes=0, batch_size=1, key=nothing,
             context_path="", args=Any[], kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `VarTrace.__init__` (varTrace.py:79-99).

  - `tau_tr`: trace time constant (ms).
  - `a_delta`: spike increment; if `<= 0`, a gated snap-to-`P_scale` trace is
    used instead of the additive form.
  - `decay_type`: `"exp"` (default), `"lin"`, or `"step"` (any other ⇒ decay 0).
  - `n_nearest_spikes`: `k > 0` makes a k-nearest-neighbor trace.
"""
function VarTrace(;
    name::AbstractString,
    n_units::Integer,
    tau_tr::Real,
    a_delta::Real,
    P_scale::Real=1.0,
    gamma_tr::Real=1.0,
    decay_type::AbstractString="exp",
    n_nearest_spikes::Integer=0,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    rest = zeros(Float64, batch_size, n_units)
    return VarTrace(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        Int(batch_size),
        Float64(tau_tr),
        Float64(a_delta),
        Float64(P_scale),
        Float64(gamma_tr),
        Int(n_nearest_spikes),
        String(decay_type),
        NGCSimLib.Compartment(make_prng_key(key)),
        NGCSimLib.Compartment(copy(rest); display_name="Input Stimulus"),
        NGCSimLib.Compartment(copy(rest); display_name="Trace Output"),
        NGCSimLib.Compartment(copy(rest); display_name="Trace")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (varTrace.py:101-120).
NGCSimLib.@compilable function advance_state!(c::VarTrace, dt)
    decay = if occursin("exp", c.decay_type)
        exp(-dt / c.tau_tr)
    elseif occursin("lin", c.decay_type)
        1.0 - dt / c.tau_tr
    else
        0.0
    end

    inputs = NGCSimLib.get_value(c.inputs)
    trace = NGCSimLib.get_value(c.trace)
    x_tr = c.gamma_tr .* trace .* decay

    if c.n_nearest_spikes > 0
        x_tr = x_tr .+ inputs .* (c.a_delta .- (trace ./ c.n_nearest_spikes))
    elseif c.a_delta > 0.0
        x_tr = x_tr .+ inputs .* c.a_delta
    else
        x_tr = x_tr .* (1.0 .- inputs) .+ inputs .* c.P_scale
    end

    NGCSimLib.set!(c.trace, x_tr)
    NGCSimLib.set!(c.outputs, x_tr)
    return c
end

# Mirrors `reset` (varTrace.py:123-130). `inputs` reset only if not wired.
NGCSimLib.@compilable function reset_state!(c::VarTrace)
    rest = zeros(Float64, c.batch_size, c.n_units)
    if !NGCSimLib.targeted(c.inputs)
        NGCSimLib.set!(c.inputs, copy(rest))
    end
    NGCSimLib.set!(c.outputs, copy(rest))
    NGCSimLib.set!(c.trace, copy(rest))
    return c
end
