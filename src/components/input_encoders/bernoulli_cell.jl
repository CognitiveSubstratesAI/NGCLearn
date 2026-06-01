# bernoulli_cell.jl — Bernoulli spike-train input encoder.
#
# 1:1 port of ngclearn/components/input_encoders/bernoulliCell.py.
#
# Samples a Bernoulli distribution per unit: each input dimension (interpreted as
# a probability in [0,1]) spikes with that probability on each step.
#   outputs = 1[uniform(0,1) < inputs]      (== Bernoulli(p=inputs))
#   tols    = (1 - outputs) * tols + outputs * t
#
# Sibling of PoissonCell (which scales the probability by dt·target_freq); the
# Bernoulli encoder uses the input value directly as the spike probability.
#
# Spec: docs/specs/11_bernoulli_cell.md.
# Decisions: NGCLearn §1 (no @ngc_component), §2 (UInt64 PRNG seed), §4 (EAGER).

"""
    BernoulliCell <: JaxComponent

An input encoder that converts real-valued `inputs` (probabilities in [0,1]) into
a Bernoulli spike train — each unit spikes with probability equal to its input.

| Input compartments:  `inputs` (per-unit spike probabilities)
| State compartments:  `key` (PRNG seed)
| Output compartments: `outputs` (binary spikes), `tols` (time-of-last-spike)

Construct with [`BernoulliCell(; ...)`](@ref). Required: `name`, `n_units`.
"""
mutable struct BernoulliCell <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters.
    n_units::Int
    batch_size::Int

    # Compartments.
    key::NGCSimLib.Compartment
    inputs::NGCSimLib.Compartment
    outputs::NGCSimLib.Compartment
    tols::NGCSimLib.Compartment
end

"""
    BernoulliCell(; name, n_units, batch_size=1, key=nothing,
                  context_path="", args=Any[], kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `BernoulliCell.__init__` (bernoulliCell.py:30-42).
`inputs` should hold per-unit spike probabilities in [0,1].
"""
function BernoulliCell(;
    name::AbstractString,
    n_units::Integer,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    rest = zeros(Float64, batch_size, n_units)
    return BernoulliCell(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        Int(batch_size),
        NGCSimLib.Compartment(make_prng_key(key)),
        NGCSimLib.Compartment(copy(rest); display_name="Input Stimulus"),
        NGCSimLib.Compartment(copy(rest); display_name="Spikes"),
        NGCSimLib.Compartment(copy(rest); display_name="Time-of-Last-Spike", units="ms")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (bernoulliCell.py:44-48). Bernoulli(p=inputs) via a
# uniform draw: spike where uniform < input probability.
NGCSimLib.@compilable function advance_state!(c::BernoulliCell, t)
    rng = Xoshiro(NGCSimLib.get_value(c.key))
    inputs = NGCSimLib.get_value(c.inputs)
    outputs = (rand(rng, Float64, size(inputs)) .< inputs) .* 1.0
    NGCSimLib.set!(c.outputs, outputs)
    NGCSimLib.set!(
        c.tols, (1.0 .- outputs) .* NGCSimLib.get_value(c.tols) .+ (outputs .* t)
    )
    NGCSimLib.set!(c.key, rand(rng, UInt64))
    return c
end

# Mirrors `reset` (bernoulliCell.py:50-57). `inputs` reset only if not wired.
NGCSimLib.@compilable function reset_state!(c::BernoulliCell)
    rest = zeros(Float64, c.batch_size, c.n_units)
    if !NGCSimLib.targeted(c.inputs)
        NGCSimLib.set!(c.inputs, copy(rest))
    end
    NGCSimLib.set!(c.outputs, copy(rest))
    NGCSimLib.set!(c.tols, copy(rest))
    return c
end
