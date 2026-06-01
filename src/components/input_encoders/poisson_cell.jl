# poisson_cell.jl — Poisson spike-train input encoder.
#
# 1:1 port of ngclearn/components/input_encoders/poissonCell.py.
#
# Samples a homogeneous Poisson process on the fly: each input dimension spikes
# with probability proportional to its magnitude, capped by `target_freq`.
#
#   pspike  = inputs * (dt / 1000) * target_freq
#   outputs = 1[uniform(0,1) < pspike]
#   tols    = (1 - outputs) * tols + outputs * t
#
# Spec: docs/specs/07_poisson_cell.md.
# Decisions: NGCLearn §1 (no @ngc_component), §2 (UInt64 PRNG seed), §4 (EAGER
# is ground truth).

"""
    PoissonCell <: JaxComponent

An input encoder that converts real-valued `inputs` into a Poisson spike train,
where each dimension's spike probability is proportional to its magnitude and
constrained by a maximum frequency `target_freq` (Hz).

| Input compartments:  `inputs` (external signal)
| State compartments:  `key` (PRNG seed)
| Output compartments: `outputs` (binary spikes), `tols` (time-of-last-spike)

Construct with [`PoissonCell(; ...)`](@ref). Required: `name`, `n_units`.
"""
mutable struct PoissonCell <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters (scalars).
    n_units::Int
    batch_size::Int
    target_freq::Float64

    # Compartments.
    key::NGCSimLib.Compartment
    inputs::NGCSimLib.Compartment
    outputs::NGCSimLib.Compartment
    tols::NGCSimLib.Compartment
end

"""
    PoissonCell(; name, n_units, target_freq=63.75, batch_size=1, key=nothing,
                context_path="", args=Any[], kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `PoissonCell.__init__` (poissonCell.py:33-50).
`target_freq` is the maximum spike frequency in Hz (must be > 0).
"""
function PoissonCell(;
    name::AbstractString,
    n_units::Integer,
    target_freq::Real=63.75,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    rest = zeros(Float64, batch_size, n_units)
    return PoissonCell(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        Int(batch_size),
        Float64(target_freq),
        NGCSimLib.Compartment(make_prng_key(key)),
        NGCSimLib.Compartment(copy(rest); display_name="Input Stimulus"),
        NGCSimLib.Compartment(copy(rest); display_name="Spikes"),
        NGCSimLib.Compartment(copy(rest); display_name="Time-of-Last-Spike", units="ms")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (poissonCell.py:52-61).
NGCSimLib.@compilable function advance_state!(c::PoissonCell, t, dt)
    rng = Xoshiro(NGCSimLib.get_value(c.key))
    inputs = NGCSimLib.get_value(c.inputs)
    pspike = inputs .* (dt / 1000.0) .* c.target_freq
    eps = rand(rng, Float64, size(inputs))
    outputs = (eps .< pspike) .* 1.0
    NGCSimLib.set!(c.outputs, outputs)
    NGCSimLib.set!(
        c.tols, (1.0 .- outputs) .* NGCSimLib.get_value(c.tols) .+ (outputs .* t)
    )
    NGCSimLib.set!(c.key, rand(rng, UInt64))
    return c
end

# Mirrors `reset` (poissonCell.py:63-70). `inputs` reset only if not wired.
NGCSimLib.@compilable function reset_state!(c::PoissonCell)
    rest = zeros(Float64, c.batch_size, c.n_units)
    if !NGCSimLib.targeted(c.inputs)
        NGCSimLib.set!(c.inputs, copy(rest))
    end
    NGCSimLib.set!(c.outputs, copy(rest))
    NGCSimLib.set!(c.tols, copy(rest))
    return c
end
