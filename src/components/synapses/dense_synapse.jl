# dense_synapse.jl — dense linear synaptic cable (no in-built learning).
#
# 1:1 port of ngclearn/components/synapses/denseSynapse.py.
#
# Forward dynamics:
#   outputs = (inputs * weights) * resist_scale + biases
# where `weights` is element-multiplied by an optional `mask` (zero entries
# implement sparsity / connectivity constraints).
#
# Spec: docs/specs/05_dense_synapse.md.
# Decisions: NGCLearn §1 (no @ngc_component), §2 (UInt64 PRNG seed).

# ── Distribution generators (subset of upstream DistributionGenerator) ───────
# Upstream `DistributionGenerator` is a small class with `.uniform()`,
# `.gaussian()`, etc. each returning a closure `(shape, key) -> samples`. We
# express the same with `(name, args...)`-tuples consumed by `_init_array`;
# this keeps the synapse constructor's `weight_init=("uniform", 0.025, 0.8)`
# form 1:1 with upstream `weight_init=DistributionGenerator.uniform(0.025, 0.8)`.

# Apply an optional structural modifier to a freshly-initialized matrix.
# Mirrors the `hollow` / `eye` branches of upstream `_process_params_jax`
# (distribution_generator.py:330-334):
#   :hollow → (1 - I) ⊙ A  (zero the diagonal)
#   :eye    → I ⊙ A         (keep only the diagonal)
function _apply_structure(ary::AbstractMatrix, mod::Symbol)
    size(ary, 1) == size(ary, 2) ||
        error("_apply_structure: `:$mod` requires a square matrix, got $(size(ary))")
    out = copy(ary)
    if mod === :hollow          # zero the diagonal: (1 - I) ⊙ A
        for i in axes(out, 1)
            out[i, i] = zero(eltype(out))
        end
    elseif mod === :eye         # keep only the diagonal: I ⊙ A
        diag = [out[i, i] for i in axes(out, 1)]
        fill!(out, zero(eltype(out)))
        for i in axes(out, 1)
            out[i, i] = diag[i]
        end
    else
        error("_apply_structure: unknown modifier `$mod` (supported: :hollow, :eye)")
    end
    return out
end
_apply_structure(ary, ::Nothing) = ary

"""
    _init_array(shape, rng, init) -> Array

Initialize an array of given `shape` using the named distribution `init` of the
form `("uniform", amin, amax)`, `("gaussian", mu, sigma)`, or `("constant", c)`.
A `("constant", c, :hollow)` / `("constant", c, :eye)` 3-tuple additionally zeros
the diagonal / keeps only the diagonal (square matrices; mirrors upstream's
`constant(value, hollow=True)` / `constant(value, eye=True)`). `("fan_in_gaussian",)`
draws `N(0, √(1/fan_in))` with `fan_in = shape[1]` (He-style).
Mirrors the call surface of upstream's `DistributionGenerator.*` constructors.
"""
function _init_array(shape::Tuple, rng, init)
    name = init[1]
    if name == "uniform"
        amin = Float64(init[2])
        amax = Float64(init[3])
        return rand(rng, Float64, shape...) .* (amax - amin) .+ amin
    elseif name == "gaussian" || name == "normal"
        mu = Float64(init[2])
        sigma = Float64(init[3])
        return randn(rng, Float64, shape...) .* sigma .+ mu
    elseif name == "constant"
        c = Float64(init[2])
        ary = fill(c, shape...)
        modifier = length(init) >= 3 ? init[3] : nothing
        return _apply_structure(ary, modifier)
    elseif name == "fan_in_gaussian"
        # He-style fan-in Gaussian: N(0, sqrt(1/fan_in)), fan_in = shape[1]
        # (number of input units, i.e. rows). Mirrors upstream
        # DistributionGenerator.fan_in_gaussian (distribution_generator.py:253).
        length(shape) >= 2 ||
            error("_init_array: fan_in_gaussian requires a ≥2-D shape, got $shape")
        fan_in = shape[1]
        sigma = sqrt(1.0 / fan_in)
        return randn(rng, Float64, shape...) .* sigma
    else
        error(
            "_init_array: unsupported distribution `$name` " *
            "(supported: uniform, gaussian/normal, constant, fan_in_gaussian)"
        )
    end
end

# ── Component type ───────────────────────────────────────────────────────────

"""
    DenseSynapse <: JaxComponent

A dense synaptic cable with no in-built learning. Performs

| `outputs = (inputs * weights) * resist_scale + biases`

| Input compartments:  `inputs`
| State compartments:  `weights`, `biases`, `mask`, `key`
| Output compartments: `outputs`

For Hebbian learning, see `HebbianSynapse` (future port). The `mask`
compartment is a multiplicative gate over `weights` (zero entries → no
connection); useful for both fixed sparsity (`p_conn < 1` at construction)
and runtime pruning.

Construct with the keyword constructor [`DenseSynapse(; ...)`](@ref). Required:
`name`, `shape`.
"""
mutable struct DenseSynapse <: JaxComponent
    # Standard component fields (decisions.md §1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters (scalars / static config — NOT compartments).
    shape::Tuple{Int, Int}
    batch_size::Int
    resist_scale::Float64

    # State / I-O compartments.
    key::NGCSimLib.Compartment      # UInt64 PRNG seed (decisions.md §2)
    inputs::NGCSimLib.Compartment
    outputs::NGCSimLib.Compartment
    weights::NGCSimLib.Compartment
    biases::NGCSimLib.Compartment
    mask::NGCSimLib.Compartment     # multiplicative gate over weights
end

"""
    DenseSynapse(; name, shape, weight_init=("uniform", 0.025, 0.8),
                  bias_init=nothing, resist_scale=1.0, p_conn=1.0,
                  mask=nothing, batch_size=1, key=nothing,
                  context_path="", args=Any[],
                  kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `DenseSynapse.__init__` (denseSynapse.py:43-96).

  - `shape::Tuple{Int,Int}`: `(n_inputs, n_outputs)`.
  - `weight_init`: distribution tuple; defaults to `("uniform", 0.025, 0.8)`
    to match upstream's `DistributionGenerator.uniform(0.025, 0.8)` default.
  - `bias_init`: distribution tuple; `nothing` disables biases (zero scalar).
  - `resist_scale`: multiplicative scaling on the linear output.
  - `p_conn`: probability a connection exists; `0 < p_conn < 1` sparsifies
    the weight matrix at construction via a Bernoulli mask.
  - `mask`: explicit override mask (else `ones(1,1)` broadcast-passthrough).
  - `key`: optional `UInt64` PRNG seed; `nothing` ⇒ OS-random
    (`make_prng_key`).
"""
function DenseSynapse(;
    name::AbstractString,
    shape::Tuple{<:Integer, <:Integer},
    weight_init=("uniform", 0.025, 0.8),
    bias_init=nothing,
    resist_scale::Real=1.0,
    p_conn::Real=1.0,
    mask::Union{Nothing, AbstractMatrix}=nothing,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    n_in, n_out = Int(shape[1]), Int(shape[2])
    seed = make_prng_key(key)
    rng = Xoshiro(seed)

    # Weight initialization. Upstream splits the JAX key into 4 sub-keys; we
    # advance the single RNG sequentially (different `rand` calls draw distinct
    # values), giving the same "fresh-key-per-init-call" property.
    weights = _init_array((n_in, n_out), rng, weight_init)

    # Bernoulli sparsity gate (denseSynapse.py:75-77).
    if 0.0 < Float64(p_conn) < 1.0
        p_mask = (rand(rng, Float64, n_in, n_out) .< Float64(p_conn)) .* 1.0
        weights = weights .* p_mask
    end

    # mask: explicit override else (1,1) broadcast-passthrough. Mirrors
    # denseSynapse.py:86-89.
    _mask = mask === nothing ? ones(Float64, 1, 1) : Matrix{Float64}(mask)

    # biases: distribution tuple else scalar 0.0. Mirrors denseSynapse.py:92-93.
    biases = if bias_init === nothing
        zeros(Float64, 1, n_out)
    else
        _init_array((1, n_out), rng, bias_init)
    end

    pre = zeros(Float64, batch_size, n_in)
    post = zeros(Float64, batch_size, n_out)

    # Advance the RNG once after init and stash the advanced seed; subsequent
    # draws (if any) start from a fresh state.
    advanced = rand(rng, UInt64)

    return DenseSynapse(
        String(name),
        String(context_path),
        args,
        kwargs,
        (n_in, n_out),
        Int(batch_size),
        Float64(resist_scale),
        NGCSimLib.Compartment(advanced),
        NGCSimLib.Compartment(copy(pre); display_name="Inputs"),
        NGCSimLib.Compartment(copy(post); display_name="Outputs"),
        NGCSimLib.Compartment(weights; display_name="Weights"),
        NGCSimLib.Compartment(biases; display_name="Biases"),
        NGCSimLib.Compartment(_mask; display_name="Connectivity Mask")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (denseSynapse.py:98-102). Marked `@compilable` for
# the Parser; eager dispatch is the ground-truth path.
NGCSimLib.@compilable function advance_state!(c::DenseSynapse)
    W = NGCSimLib.get_value(c.weights) .* NGCSimLib.get_value(c.mask)
    out =
        (NGCSimLib.get_value(c.inputs) * W) .* c.resist_scale .+
        NGCSimLib.get_value(c.biases)
    NGCSimLib.set!(c.outputs, out)
    return c
end

# Mirrors `reset` (denseSynapse.py:104-109). Only resets `inputs` when it
# is NOT externally wired (matches LIFCell.reset semantics).
NGCSimLib.@compilable function reset_state!(c::DenseSynapse)
    n_in = c.shape[1]
    n_out = c.shape[2]
    if !NGCSimLib.targeted(c.inputs)
        NGCSimLib.set!(c.inputs, zeros(Float64, c.batch_size, n_in))
    end
    NGCSimLib.set!(c.outputs, zeros(Float64, c.batch_size, n_out))
    return c
end
