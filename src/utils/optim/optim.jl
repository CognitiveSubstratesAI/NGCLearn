# optim.jl — gradient-style optimizers used to step Hebbian (and future) synapses.
#
# 1:1 port of ngclearn/utils/optim/{sgd,adam}.py + optim_utils.py.
#
# Phase-A scope: SGD + Adam (the two the synapse zoo uses today). NAG can be
# ported when a component imports it.
#
# Functional state model — every step returns NEW state + NEW params (no
# in-place mutation), matching upstream JAX-functional convention. State is
# a NamedTuple of plain Float64 arrays:
#   SGD :  (; time_step::Float64)
#   Adam: (; g1::Vector{Matrix{Float64}}, g2::Vector{Matrix{Float64}}, time_step::Float64)
#
# `theta` and `updates` are `Vector{Matrix{Float64}}` — typically
# `[weights]` or `[weights, biases]` to allow optimizing several parameters
# under one optimizer.

# ── SGD ──────────────────────────────────────────────────────────────────────

"""
    sgd_init(theta::Vector{<:AbstractMatrix}) -> NamedTuple

Initial SGD state: just a step counter. Mirrors `sgd_init`
(optim/sgd.py:43-44).
"""
sgd_init(theta::Vector{<:AbstractMatrix}) = (; time_step=0.0)

"""
    sgd_step(opt_params, theta, updates; eta=0.001) -> (new_opt_params, new_theta)

One SGD step: `θ_new = θ - η * update`. Mirrors `sgd_step`
(optim/sgd.py:16-40).
"""
function sgd_step(opt_params, theta::Vector{<:AbstractMatrix},
    updates::Vector{<:AbstractMatrix}; eta::Real=0.001)
    new_step = opt_params.time_step + 1.0
    new_theta = [theta[i] .- updates[i] .* eta for i in eachindex(theta)]
    return (; time_step=new_step), new_theta
end

# ── Adam ─────────────────────────────────────────────────────────────────────

"""
    adam_init(theta::Vector{<:AbstractMatrix}) -> NamedTuple

Initial Adam state: zero 1st + 2nd moments per parameter, step counter at 0.
Mirrors `adam_init` (optim/adam.py:87-92).
"""
function adam_init(theta::Vector{<:AbstractMatrix})
    g1 = [zero(t) for t in theta]
    g2 = [zero(t) for t in theta]
    return (; g1=g1, g2=g2, time_step=0.0)
end

# Per-parameter Adam step (matches `step_update`, optim/adam.py:8-49).
@inline function _adam_step_param(param, update, g1, g2, eta, beta1, beta2,
    time_step, eps)
    _g1 = beta1 .* g1 .+ (1.0 - beta1) .* update
    _g2 = beta2 .* g2 .+ (1.0 - beta2) .* (update .^ 2)
    g1_unb = _g1 ./ (1.0 - beta1^time_step)
    g2_unb = _g2 ./ (1.0 - beta2^time_step)
    _param = param .- eta .* g1_unb ./ (sqrt.(g2_unb) .+ eps)
    return _param, _g1, _g2
end

"""
    adam_step(opt_params, theta, updates; eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8) -> (new_opt_params, new_theta)

One Adam step over each `(theta[i], updates[i])` pair. Mirrors `adam_step`
(optim/adam.py:51-85).
"""
function adam_step(opt_params, theta::Vector{<:AbstractMatrix},
    updates::Vector{<:AbstractMatrix};
    eta::Real=0.001, beta1::Real=0.9, beta2::Real=0.999,
    eps::Real=1e-8)
    new_step = opt_params.time_step + 1.0
    new_g1 = similar(opt_params.g1)
    new_g2 = similar(opt_params.g2)
    new_theta = similar(theta)
    for i in eachindex(theta)
        p, g1, g2 = _adam_step_param(theta[i], updates[i],
            opt_params.g1[i], opt_params.g2[i],
            eta, beta1, beta2, new_step, eps)
        new_theta[i] = p
        new_g1[i] = g1
        new_g2[i] = g2
    end
    return (; g1=new_g1, g2=new_g2, time_step=new_step), new_theta
end

# ── Dispatcher (matches upstream's `get_opt_*_fn` factory pattern) ──────────

"""
    get_opt_init_fn(opt::AbstractString) -> Function

Return the init-state constructor for the named optimizer. Mirrors
`get_opt_init_fn` (optim/optim_utils.py:5-10). Currently supports `"adam"`
and `"sgd"`.
"""
function get_opt_init_fn(opt::AbstractString)
    opt == "adam" && return adam_init
    opt == "sgd" && return sgd_init
    error("get_opt_init_fn: unsupported optimizer `$opt` (have: adam, sgd)")
end

"""
    get_opt_step_fn(opt::AbstractString; kwargs...) -> Function

Return a step function pre-bound with optimizer hyperparameters. Mirrors
`get_opt_step_fn` (optim/optim_utils.py:13-19). The returned function takes
`(opt_params, theta, updates)` and returns `(new_opt_params, new_theta)`.

Phase-A coverage: `"adam"`, `"sgd"`.
"""
function get_opt_step_fn(opt::AbstractString; kwargs...)
    if opt == "adam"
        return (op, th, up) -> adam_step(op, th, up; kwargs...)
    elseif opt == "sgd"
        return (op, th, up) -> sgd_step(op, th, up; kwargs...)
    else
        error("get_opt_step_fn: unsupported optimizer `$opt` (have: adam, sgd)")
    end
end
