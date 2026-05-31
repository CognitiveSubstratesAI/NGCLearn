# hebbian_synapse.jl — dense synapse with two-factor Hebbian plasticity.
#
# 1:1 port of ngclearn/components/synapses/hebbian/hebbianSynapse.py.
#
# Extends the forward dynamics of `DenseSynapse` with:
#   - a two-factor Hebbian update rule  dW = pre' * post   (+ regularizer)
#   - an embedded optimizer (SGD or Adam) that physically steps the weights
#   - sign / bound / prior knobs for descent vs ascent, soft bounding, etc.
#
# Upstream uses inheritance (`HebbianSynapse <: DenseSynapse`); Julia doesn't
# do single-class inheritance for mutable structs, so we duplicate the
# `DenseSynapse` fields verbatim (this matches NGCLearn decisions §1's
# "no @ngc_component, declare fields explicitly" stance).
#
# Spec: docs/specs/06_hebbian_synapse.md (forthcoming — auto-generated from
# this file's docstrings until the canonical port-audit lands).

# ── Helpers: Hebbian update + weight constraint ─────────────────────────────

"""
    _calc_update(pre, post, W; w_bound, sign_value, prior_type, prior_lmbda,
                 pre_wght, post_wght) -> (dW, db)

Two-factor Hebbian adjustment. Mirrors `_calc_update`
(hebbianSynapse.py:16-69).

Update law:
  `dW = (pre * pre_wght)' * (post * post_wght)`
  `db = sum(post, dims=1)`
then `w_bound > 0` soft-bounds `dW`, and an optional prior contributes a
regularization term `dW_reg`. Final return is `(dW + dW_reg, db) * sign_value`.

Supported `prior_type`s:
  - `"l2"` / `"ridge"` : `dW_reg = -W * prior_lmbda`
  - `"l1"` / `"lasso"` : `dW_reg = -sign(W) * prior_lmbda`
  - `"l1l2"` / `"elastic_net"` : `prior_lmbda` is `(scale, l1_ratio)`
  - any other (including `"constant"`) → no regularizer.
"""
function _calc_update(pre, post, W;
    w_bound::Real=0.0, sign_value::Real=1.0,
    prior_type::AbstractString="constant",
    prior_lmbda=0.0,
    pre_wght::Real=1.0, post_wght::Real=1.0)
    _pre = pre .* pre_wght
    _post = post .* post_wght
    dW = transpose(_pre) * _post                    # Hebbian adjustment
    db = sum(_post; dims=1)

    if w_bound > 0.0
        dW = dW .* (w_bound .- abs.(W))             # soft bound
    end

    dW_reg = zeros(eltype(W), size(W))
    pt = lowercase(String(prior_type))
    if pt == "l2" || pt == "ridge"
        dW_reg = -W .* prior_lmbda
    elseif pt == "l1" || pt == "lasso"
        dW_reg = -sign.(W) .* prior_lmbda
    elseif pt == "l1l2" || pt == "elastic_net"
        prior_scale, l1_ratio = prior_lmbda
        dW_reg = (.-sign.(W) .* l1_ratio .- W .* ((1 - l1_ratio) / 2)) .* prior_scale
    end

    dW = dW .+ dW_reg
    return dW .* sign_value, db .* sign_value
end

"""
    _enforce_constraints(W; w_bound=0.0, is_nonnegative=false) -> W_clipped

Apply hard-clip constraints to weights. Mirrors `_enforce_constraints`
(hebbianSynapse.py:71-93). With `w_bound <= 0` returns `W` unchanged.
"""
function _enforce_constraints(W; w_bound::Real=0.0, is_nonnegative::Bool=false)
    w_bound <= 0.0 && return W
    return is_nonnegative ? clamp.(W, 0.0, w_bound) : clamp.(W, -w_bound, w_bound)
end

# ── Component type ──────────────────────────────────────────────────────────

"""
    HebbianSynapse <: JaxComponent

Dense synaptic cable that adapts its weights via a two-factor Hebbian rule
plus an embedded optimizer (SGD or Adam).

| Input compartments:  `inputs`, `pre`, `post`
| State compartments:  `weights`, `biases`, `mask`, `key`, `opt_params`
| Output compartments: `outputs`, `dWeights`, `dBiases`

Forward pass (`advance_state!`) is identical to `DenseSynapse`:
`outputs = (inputs * weights) * resist_scale + biases`.

Update pass:
  - [`compute_update!`](@ref): fills `dWeights` / `dBiases` from `pre`, `post`.
  - [`evolve!`](@ref): does compute_update then steps `weights` / `biases`
    through the embedded optimizer; also re-applies bound + mask constraints.

Construct with [`HebbianSynapse(; ...)`](@ref). Required: `name`, `shape`.
"""
mutable struct HebbianSynapse <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters.
    shape::Tuple{Int, Int}
    batch_size::Int
    resist_scale::Float64
    eta::Float64
    w_bound::Float64
    is_nonnegative::Bool
    sign_value::Float64
    pre_wght::Float64
    post_wght::Float64
    prior_type::String          # "constant" | "l1"/"lasso" | "l2"/"ridge" | "l1l2"/"elastic_net"
    prior_lmbda::Any            # scalar OR (scale, l1_ratio) tuple for elastic_net
    optim_type::String          # "sgd" | "adam"
    has_bias::Bool

    # Per-instance Adam/SGD step function (pre-bound with eta).
    opt_step::Function

    # Compartments — superset of DenseSynapse.
    key::NGCSimLib.Compartment
    inputs::NGCSimLib.Compartment
    outputs::NGCSimLib.Compartment
    weights::NGCSimLib.Compartment
    biases::NGCSimLib.Compartment
    mask::NGCSimLib.Compartment
    pre::NGCSimLib.Compartment
    post::NGCSimLib.Compartment
    dWeights::NGCSimLib.Compartment
    dBiases::NGCSimLib.Compartment
    opt_params::NGCSimLib.Compartment  # NamedTuple state (Adam/SGD)
end

"""
    HebbianSynapse(; name, shape, eta=0.0, weight_init=("uniform", -0.3, 0.3),
                   bias_init=nothing, w_bound=1.0, is_nonnegative=false,
                   prior=("constant", 0.0), sign_value=1.0, optim_type="sgd",
                   pre_wght=1.0, post_wght=1.0, p_conn=1.0, mask=nothing,
                   resist_scale=1.0, batch_size=1, key=nothing,
                   context_path="", args=Any[],
                   kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `HebbianSynapse.__init__`
(hebbianSynapse.py:170-219). Defaults track upstream verbatim:

  - `eta=0.0`: learning rate (set > 0 to actually learn).
  - `weight_init`: same `(name, args...)` form as `DenseSynapse`. Default
    differs from `DenseSynapse`'s `("uniform", 0.025, 0.8)` — upstream's
    pcn_model.py uses `dist.uniform(-0.3, 0.3)` for Hebbian cables, so we
    use that as a more pc_discrim-friendly default.
  - `bias_init`: `nothing` ⇒ no biases (1×n_out zeros, not learned).
  - `optim_type`: `"sgd"` or `"adam"` (Adam matches pc_discrim).
  - `prior`: `(name, lmbda)` regularizer; supported names listed in
    [`_calc_update`](@ref). `"gaussian"` and `"laplacian"` are mapped to
    `"ridge"` / `"lasso"` (upstream alias).
"""
function HebbianSynapse(;
    name::AbstractString,
    shape::Tuple{<:Integer, <:Integer},
    eta::Real=0.0,
    weight_init=("uniform", -0.3, 0.3),
    bias_init=nothing,
    w_bound::Real=1.0,
    is_nonnegative::Bool=false,
    prior=("constant", 0.0),
    sign_value::Real=1.0,
    optim_type::AbstractString="sgd",
    pre_wght::Real=1.0,
    post_wght::Real=1.0,
    p_conn::Real=1.0,
    mask::Union{Nothing, AbstractMatrix}=nothing,
    resist_scale::Real=1.0,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    n_in, n_out = Int(shape[1]), Int(shape[2])

    # Reuse the `_init_array` helper (defined in dense_synapse.jl, in the
    # same module).
    seed = make_prng_key(key)
    rng = Xoshiro(seed)
    weights = _init_array((n_in, n_out), rng, weight_init)
    if 0.0 < Float64(p_conn) < 1.0
        p_mask = (rand(rng, Float64, n_in, n_out) .< Float64(p_conn)) .* 1.0
        weights = weights .* p_mask
    end
    _mask = mask === nothing ? ones(Float64, 1, 1) : Matrix{Float64}(mask)
    has_bias = bias_init !== nothing
    biases = has_bias ?
             _init_array((1, n_out), rng, bias_init) :
             zeros(Float64, 1, n_out)

    # Prior alias mapping (upstream: "gaussian" → "ridge", "laplacian" → "lasso").
    prior_name, prior_l = prior
    pn = lowercase(String(prior_name))
    if pn == "gaussian"
        pn = "ridge"
    elseif pn == "laplacian"
        pn = "lasso"
    end

    # Build the bound step fn (eta pre-bound) and init optimizer state.
    opt_step = get_opt_step_fn(optim_type; eta=Float64(eta))
    init_fn = get_opt_init_fn(optim_type)
    init_theta = has_bias ? Matrix{Float64}[weights, biases] :
                 Matrix{Float64}[weights]
    opt_state = init_fn(init_theta)

    advanced = rand(rng, UInt64)
    pre = zeros(Float64, batch_size, n_in)
    post = zeros(Float64, batch_size, n_out)

    return HebbianSynapse(
        String(name),
        String(context_path),
        args,
        kwargs,
        (n_in, n_out),
        Int(batch_size),
        Float64(resist_scale),
        Float64(eta),
        Float64(w_bound),
        is_nonnegative,
        Float64(sign_value),
        Float64(pre_wght),
        Float64(post_wght),
        pn,
        prior_l,
        String(optim_type),
        has_bias,
        opt_step,
        NGCSimLib.Compartment(advanced),
        NGCSimLib.Compartment(copy(pre); display_name="Inputs"),
        NGCSimLib.Compartment(copy(post); display_name="Outputs"),
        NGCSimLib.Compartment(weights; display_name="Weights"),
        NGCSimLib.Compartment(biases; display_name="Biases"),
        NGCSimLib.Compartment(_mask; display_name="Connectivity Mask"),
        NGCSimLib.Compartment(copy(pre); display_name="Pre-Synaptic Statistic"),
        NGCSimLib.Compartment(copy(post); display_name="Post-Synaptic Statistic"),
        NGCSimLib.Compartment(zeros(Float64, n_in, n_out); display_name="dW"),
        NGCSimLib.Compartment(zeros(Float64, 1, n_out); display_name="db"),
        NGCSimLib.Compartment(opt_state)
    )
end

# ── Dynamics ─────────────────────────────────────────────────────────────────

# Forward pass (identical to DenseSynapse.advance_state!). Mirrors
# `DenseSynapse.advance_state` indirectly via inheritance upstream.
NGCSimLib.@compilable function advance_state!(c::HebbianSynapse)
    W = NGCSimLib.get_value(c.weights) .* NGCSimLib.get_value(c.mask)
    out =
        (NGCSimLib.get_value(c.inputs) * W) .* c.resist_scale .+
        NGCSimLib.get_value(c.biases)
    NGCSimLib.set!(c.outputs, out)
    return c
end

# compute_update!(c) — fill `dWeights`/`dBiases` from current `pre`/`post`/`W`
# without stepping the parameters. Mirrors `calc_update`
# (hebbianSynapse.py:242-259).
NGCSimLib.@compilable function compute_update!(c::HebbianSynapse)
    pre = NGCSimLib.get_value(c.pre)
    post = NGCSimLib.get_value(c.post)
    W = NGCSimLib.get_value(c.weights)
    dW, db = _calc_update(pre, post, W;
        w_bound=c.w_bound, sign_value=c.sign_value,
        prior_type=c.prior_type, prior_lmbda=c.prior_lmbda,
        pre_wght=c.pre_wght, post_wght=c.post_wght)
    NGCSimLib.set!(c.dWeights, dW)
    NGCSimLib.set!(c.dBiases, db)
    return c
end

# evolve!(c, dt) — compute the Hebbian update AND step the optimizer to
# actually evolve `weights`/`biases` (also re-applies bound + mask). Mirrors
# `evolve` (hebbianSynapse.py:261-300).
NGCSimLib.@compilable function evolve!(c::HebbianSynapse, dt)
    pre = NGCSimLib.get_value(c.pre)
    post = NGCSimLib.get_value(c.post)
    W = NGCSimLib.get_value(c.weights)
    bias = NGCSimLib.get_value(c.biases)
    state = NGCSimLib.get_value(c.opt_params)

    dW, db = _calc_update(pre, post, W;
        w_bound=c.w_bound, sign_value=c.sign_value,
        prior_type=c.prior_type, prior_lmbda=c.prior_lmbda,
        pre_wght=c.pre_wght, post_wght=c.post_wght)

    theta = c.has_bias ?
            Matrix{Float64}[W, bias] :
            Matrix{Float64}[W]
    updates = c.has_bias ?
              Matrix{Float64}[dW, db] :
              Matrix{Float64}[dW]

    new_state, new_theta = c.opt_step(state, theta, updates)

    W_new = _enforce_constraints(new_theta[1];
        w_bound=c.w_bound,
        is_nonnegative=c.is_nonnegative)
    W_new = W_new .* NGCSimLib.get_value(c.mask)

    NGCSimLib.set!(c.opt_params, new_state)
    NGCSimLib.set!(c.weights, W_new)
    if c.has_bias
        NGCSimLib.set!(c.biases, new_theta[2])
    end
    NGCSimLib.set!(c.dWeights, dW)
    NGCSimLib.set!(c.dBiases, db)
    return c
end

NGCSimLib.@compilable function reset_state!(c::HebbianSynapse)
    pre = zeros(Float64, c.batch_size, c.shape[1])
    post = zeros(Float64, c.batch_size, c.shape[2])
    if !NGCSimLib.targeted(c.inputs)
        NGCSimLib.set!(c.inputs, copy(pre))
    end
    NGCSimLib.set!(c.outputs, copy(post))
    NGCSimLib.set!(c.pre, copy(pre))
    NGCSimLib.set!(c.post, copy(post))
    NGCSimLib.set!(c.dWeights, zeros(Float64, c.shape[1], c.shape[2]))
    NGCSimLib.set!(c.dBiases, zeros(Float64, 1, c.shape[2]))
    return c
end
