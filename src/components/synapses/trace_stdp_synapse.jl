# trace_stdp_synapse.jl — dense synapse with trace-based STDP plasticity.
#
# 1:1 port of ngclearn/components/synapses/hebbian/traceSTDPSynapse.py.
#
# Forward pass is identical to DenseSynapse. Plasticity is trace-based
# spike-timing-dependent plasticity (Morrison et al. 2007; Bi & Poo 2001):
#
#   dW = A_plus * (preTrace - x_tar)ᵀ · postSpike  -  A_minus * preSpikeᵀ · postTrace
#
# with optional power-law scaling (mu>0): the LTP term is scaled by
# (w_bound - W)^mu and the LTD term by W^mu. `evolve!` then applies the update
# with global rate `eta`, an optional weight decay (tau_w>0), clips to
# [w_eps, w_bound], and re-applies the weight mask.
#
# Spec: docs/specs/09_trace_stdp_synapse.md.
# Decisions: NGCLearn §1 (explicit fields, no inheritance / @ngc_component),
# §2 (UInt64 PRNG seed), §4 (EAGER is ground truth).

"""
    TraceSTDPSynapse <: JaxComponent

A dense synaptic cable whose weights adapt via trace-based STDP.

| Input compartments:  `inputs`, `preSpike`, `postSpike`, `preTrace`, `postTrace`
| State compartments:  `weights`, `mask`, `key`
| Output compartments: `outputs`, `dWeights`

Forward pass: `outputs = (inputs * (weights ⊙ mask)) * resist_scale`.
Plasticity: see [`evolve!`](@ref).

Construct with [`TraceSTDPSynapse(; ...)`](@ref). Required: `name`, `shape`,
`A_plus`, `A_minus`.
"""
mutable struct TraceSTDPSynapse <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters.
    shape::Tuple{Int, Int}
    batch_size::Int
    resist_scale::Float64
    A_plus::Float64
    A_minus::Float64
    eta::Float64
    mu::Float64                 # power-scale exponent (0 ⇒ plain additive STDP)
    preTrace_target::Float64    # x_tar
    w_bound::Float64
    w_eps::Float64
    tau_w::Float64              # weight-decay coefficient (0 ⇒ off)

    # Compartments.
    key::NGCSimLib.Compartment
    inputs::NGCSimLib.Compartment
    outputs::NGCSimLib.Compartment
    weights::NGCSimLib.Compartment
    mask::NGCSimLib.Compartment
    preSpike::NGCSimLib.Compartment
    postSpike::NGCSimLib.Compartment
    preTrace::NGCSimLib.Compartment
    postTrace::NGCSimLib.Compartment
    dWeights::NGCSimLib.Compartment
end

"""
    TraceSTDPSynapse(; name, shape, A_plus, A_minus, eta=1.0, mu=0.0,
                     pretrace_target=0.0, weight_init=("uniform", 0.025, 0.8),
                     resist_scale=1.0, p_conn=1.0, w_bound=1.0, tau_w=0.0,
                     weight_mask=nothing, batch_size=1, key=nothing,
                     context_path="", args=Any[], kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `TraceSTDPSynapse.__init__`
(traceSTDPSynapse.py:62-95). `A_plus`/`A_minus` are LTP/LTD strengths; `mu>0`
enables power-law scaling; `tau_w>0` enables weight decay.
"""
function TraceSTDPSynapse(;
    name::AbstractString,
    shape::Tuple{<:Integer, <:Integer},
    A_plus::Real,
    A_minus::Real,
    eta::Real=1.0,
    mu::Real=0.0,
    pretrace_target::Real=0.0,
    weight_init=("uniform", 0.025, 0.8),
    resist_scale::Real=1.0,
    p_conn::Real=1.0,
    w_bound::Real=1.0,
    tau_w::Real=0.0,
    weight_mask::Union{Nothing, AbstractMatrix}=nothing,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    n_in, n_out = Int(shape[1]), Int(shape[2])
    seed = make_prng_key(key)
    rng = Xoshiro(seed)
    # Reuse _init_array (defined in dense_synapse.jl, same module).
    weights = _init_array((n_in, n_out), rng, weight_init)
    if 0.0 < Float64(p_conn) < 1.0
        p_mask = (rand(rng, Float64, n_in, n_out) .< Float64(p_conn)) .* 1.0
        weights = weights .* p_mask
    end
    _mask = weight_mask === nothing ? ones(Float64, 1, 1) : Matrix{Float64}(weight_mask)
    weights = weights .* _mask     # mirror upstream `weights *= weight_mask`
    advanced = rand(rng, UInt64)

    pre = zeros(Float64, batch_size, n_in)
    post = zeros(Float64, batch_size, n_out)

    return TraceSTDPSynapse(
        String(name),
        String(context_path),
        args,
        kwargs,
        (n_in, n_out),
        Int(batch_size),
        Float64(resist_scale),
        Float64(A_plus),
        Float64(A_minus),
        Float64(eta),
        Float64(mu),
        Float64(pretrace_target),
        Float64(w_bound),
        0.0,                       # w_eps (upstream hardcodes 0.)
        Float64(tau_w),
        NGCSimLib.Compartment(advanced),
        NGCSimLib.Compartment(copy(pre); display_name="Inputs"),
        NGCSimLib.Compartment(copy(post); display_name="Outputs"),
        NGCSimLib.Compartment(weights; display_name="Weights"),
        NGCSimLib.Compartment(_mask; display_name="Weight Mask"),
        NGCSimLib.Compartment(copy(pre); display_name="Pre-Synaptic Spike"),
        NGCSimLib.Compartment(copy(post); display_name="Post-Synaptic Spike"),
        NGCSimLib.Compartment(copy(pre); display_name="Pre-Synaptic Trace"),
        NGCSimLib.Compartment(copy(post); display_name="Post-Synaptic Trace"),
        NGCSimLib.Compartment(zeros(Float64, n_in, n_out); display_name="dW")
    )
end

# Trace-STDP weight-change rule. Mirrors `_compute_update`
# (traceSTDPSynapse.py:97-119).
function _trace_stdp_update(c::TraceSTDPSynapse)
    W = NGCSimLib.get_value(c.weights)
    preTr = NGCSimLib.get_value(c.preTrace)
    postTr = NGCSimLib.get_value(c.postTrace)
    preSp = NGCSimLib.get_value(c.preSpike)
    postSp = NGCSimLib.get_value(c.postSpike)

    if c.mu > 0.0
        post_shift = (c.w_bound .- W) .^ c.mu
        pre_shift = W .^ c.mu
        dWpost =
            (post_shift .* (transpose(preTr .- c.preTrace_target) * postSp)) .* c.A_plus
        dWpre = if c.A_minus > 0.0
            .-(pre_shift .* (transpose(preSp) * postTr)) .* c.A_minus
        else
            zeros(eltype(W), size(W))
        end
    else
        dWpost = transpose(preTr .- c.preTrace_target) * (postSp .* c.A_plus)
        dWpre = if c.A_minus > 0.0
            .-(transpose(preSp) * (postTr .* c.A_minus))
        else
            zeros(eltype(W), size(W))
        end
    end
    return dWpost .+ dWpre
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Forward pass (identical to DenseSynapse.advance_state). No bias term upstream
# (TraceSTDPSynapse passes bias_init=None).
NGCSimLib.@compilable function advance_state!(c::TraceSTDPSynapse)
    W = NGCSimLib.get_value(c.weights) .* NGCSimLib.get_value(c.mask)
    out = (NGCSimLib.get_value(c.inputs) * W) .* c.resist_scale
    NGCSimLib.set!(c.outputs, out)
    return c
end

# evolve!(c) — apply the trace-STDP update with rate eta + optional decay,
# clip to [w_eps, w_bound], re-apply mask. Mirrors `evolve`
# (traceSTDPSynapse.py:121-134).
NGCSimLib.@compilable function evolve!(c::TraceSTDPSynapse)
    dWeights = _trace_stdp_update(c)
    W = NGCSimLib.get_value(c.weights)
    decay = c.tau_w > 0.0 ? W ./ c.tau_w : zeros(eltype(W), size(W))

    w = W .+ (dWeights .* c.eta) .- decay
    w = clamp.(w, c.w_eps, c.w_bound - c.w_eps)
    mask = NGCSimLib.get_value(c.mask)
    w = ifelse.(mask .!= 0.0, w, 0.0)

    NGCSimLib.set!(c.weights, w)
    NGCSimLib.set!(c.dWeights, dWeights)
    return c
end

# Mirrors `reset` (traceSTDPSynapse.py:137-149). `inputs` reset only if not wired.
NGCSimLib.@compilable function reset_state!(c::TraceSTDPSynapse)
    n_in, n_out = c.shape
    pre = zeros(Float64, c.batch_size, n_in)
    post = zeros(Float64, c.batch_size, n_out)
    if !NGCSimLib.targeted(c.inputs)
        NGCSimLib.set!(c.inputs, copy(pre))
    end
    NGCSimLib.set!(c.outputs, copy(post))
    NGCSimLib.set!(c.preSpike, copy(pre))
    NGCSimLib.set!(c.postSpike, copy(post))
    NGCSimLib.set!(c.preTrace, copy(pre))
    NGCSimLib.set!(c.postTrace, copy(post))
    NGCSimLib.set!(c.dWeights, zeros(Float64, n_in, n_out))
    return c
end
