# sparse_coding.jl — Olshausen & Field (1996) sparse coding exhibit.
#
# Faithful port of ngc-museum/exhibits/olshausen_sc/sparse_coding.py.
#
#   Olshausen, B. & Field, D. (1996). "Emergence of simple-cell receptive field
#   properties by learning a sparse code for natural images." Nature 381:607-609.
#
# A single-latent-layer generative (predictive-coding) model that learns a sparse
# dictionary `W1` over natural-image patches. Under the NGC naming convention
# (Ororbia & Kifer 2022) this is the GNCN-t1/SC.
#
# Topology:
#   z1 (RateCell, sparse prior) ─W1─▶ e0.mu        (generative prediction)
#   obs ─clamp─▶ e0.target                          (the patch to reconstruct)
#   e0.dmu ─E1 (= W1ᵀ)─▶ z1.j                        (error feedback drives latents)
#   Hebbian W1: pre = z1.zF, post = e0.dmu
#
# Two variants (`model_type`):
#   - "sc_cauchy" (default): Cauchy prior on z1 (prior=("cauchy", λ=0.14)) induces
#     sparsity through the leak term.
#   - "ista": no prior (prior=("gaussian", 0)) + soft-threshold on z1
#     (threshold=("soft_threshold", λ=5e-3)) — emulates ISTA (Daubechies 2004).
#
# `process!` runs a T-step inference window (reset → tie E1=W1ᵀ → clamp obs →
# T E-steps of settling → one Hebbian M-step → L2-normalize W1 rows). Returns the
# settled reconstruction `e0.mu` and the loss `e0.L`.
#
# Decisions: model layer (§7 — setup-before-wire, eager hand-ordered loop,
# per-model Context name).

"""
    SparseCoding

An Olshausen & Field (1996) sparse-coding model: a single latent layer `z1` with
a sparsity-inducing prior generates a reconstruction of the input through a
learned dictionary `W1`, trained by predictive-coding inference + a 2-factor
Hebbian update. Supports a Cauchy-prior variant (`"sc_cauchy"`) and an ISTA
soft-threshold variant (`"ista"`).

Construct with [`SparseCoding(; ...)`](@ref); drive with [`process!`](@ref).
"""
mutable struct SparseCoding
    in_dim::Int
    hid_dim::Int
    T::Int
    dt::Float64
    model_type::String

    z1::RateCell           # sparse latent codebook
    e0::GaussianErrorCell  # reconstruction error at the input layer
    W1::HebbianSynapse     # generative dictionary (z1 → prediction)
    E1::DenseSynapse       # error feedback (= W1ᵀ, re-pinned each process)
end

"""
    SparseCoding(; in_dim, hid_dim=100, T=200, dt=1.0, batch_size=1,
                 model_type="sc_cauchy", key=nothing, name="Circuit")

Build a sparse-coding model. Topology, hyperparameters, and wiring mirror
`SparseCoding.__init__` (sparse_coding.py:70-176). `model_type` selects the
Cauchy-prior (`"sc_cauchy"`) or ISTA soft-threshold (`"ista"`) variant. All
compartments are `post_init!`-ed before any `>>` wiring (decisions §7); pass a
distinct `name` to build independent models in one process.
"""
function SparseCoding(;
    in_dim::Integer,
    hid_dim::Integer=100,
    T::Integer=200,
    dt::Real=1.0,
    batch_size::Integer=1,
    model_type::AbstractString="sc_cauchy",
    key::Union{Integer, Nothing}=nothing,
    name::AbstractString="Circuit"
)
    eta_w = 1e-2
    tau_m = 20.0
    act_fx = "identity"
    # Variant-specific prior / threshold config (sparse_coding.py:97-109).
    if model_type == "sc_cauchy"
        prior_type = "cauchy"
        threshold_type = "none"
        lmbda = 0.14
    elseif model_type == "ista"
        prior_type = "gaussian"
        threshold_type = "soft_threshold"
        lmbda = 5e-3
    else
        error(
            "SparseCoding: model_type must be \"sc_cauchy\" or \"ista\", got \"$model_type\""
        )
    end
    k = key === nothing ? 0 : Int(key)

    local model
    NGCSimLib.Context(name) do _ctx
        # NB: RateCell is deterministic — no `key` kwarg (unlike spiking cells).
        z1 = RateCell(; name="z1", n_units=hid_dim, tau_m=tau_m, act_fx=act_fx,
            prior=(prior_type, lmbda), threshold=(threshold_type, lmbda),
            integration_type="euler", batch_size=batch_size)
        e0 = GaussianErrorCell(; name="e0", n_units=in_dim, batch_size=batch_size)
        W1 = HebbianSynapse(; name="W1", shape=(hid_dim, in_dim), eta=eta_w,
            weight_init=("fan_in_gaussian",), bias_init=nothing, w_bound=0.0,
            optim_type="sgd", sign_value=-1.0, batch_size=batch_size, key=k + 1)
        E1 = DenseSynapse(; name="E1", shape=(in_dim, hid_dim),
            weight_init=("uniform", -0.2, 0.2), resist_scale=1.0,
            batch_size=batch_size, key=k + 2)

        # SETUP-BEFORE-WIRE (decisions §7).
        post_init!.((z1, e0, W1, E1))

        # Forward: z1.zF -(W1)-> e0.mu (sparse_coding.py:149-150).
        z1.zF >> W1.inputs
        W1.outputs >> e0.mu
        # Feedback: e0.dmu -(E1)-> z1.j (sparse_coding.py:153-154).
        e0.dmu >> E1.inputs
        E1.outputs >> z1.j
        # Hebbian (pre, post) wiring (sparse_coding.py:156-157).
        z1.zF >> W1.pre
        e0.dmu >> W1.post

        model = SparseCoding(
            Int(in_dim), Int(hid_dim), Int(T), Float64(dt), String(model_type),
            z1, e0, W1, E1
        )
    end
    return model
end

# L2-normalize each row of W1 to unit norm (sparse_coding.py:178-179).
"""
    norm!(m::SparseCoding) -> SparseCoding

Rescale each row of the dictionary `W1` to unit L2 norm — the dictionary-
normalization step that keeps sparse-coding bases bounded. Mirrors
`SparseCoding.norm` (sparse_coding.py:178-179).
"""
function norm!(m::SparseCoding)
    W = NGCSimLib.get_value(m.W1.weights)
    NGCSimLib.set!(m.W1.weights, normalize_matrix(W, 1.0; order=2, axis=2))
    return m
end

# One inference step (advance ordering mirrors advance_process, lines 160-164).
function _advance!(m::SparseCoding)
    advance_state!(m.W1)
    advance_state!(m.E1)
    advance_state!(m.z1, m.dt)        # RateCell sig: (dt)
    advance_state!(m.e0, m.dt)        # GaussianErrorCell sig: (dt)
    return m
end

function _reset!(m::SparseCoding)
    reset_state!(m.W1)
    reset_state!(m.E1)
    reset_state!(m.z1)
    reset_state!(m.e0)
    return m
end

"""
    process!(m::SparseCoding, obs; adapt=true) -> (obs_mu, L0)

Process one observation (`batch×in_dim`) over a T-step inference window. Mirrors
`SparseCoding.process` (sparse_coding.py:246-291): tie `E1 = W1ᵀ` → reset →
clamp `obs` into `e0.target` → `T` E-steps of latent settling → (if `adapt`) one
Hebbian M-step + L2 dictionary normalization. Returns the settled reconstruction
`e0.mu` and the local loss `e0.L`.
"""
function process!(m::SparseCoding, obs::AbstractMatrix; adapt::Bool=true)
    # Tie feedback weights to the transpose of the forward dictionary
    # (sparse_coding.py:273), then reset to resting state.
    NGCSimLib.set!(m.E1.weights, permutedims(NGCSimLib.get_value(m.W1.weights)))
    _reset!(m)

    NGCSimLib.set!(m.e0.target, obs)   # clamp the patch to reconstruct
    for _ in 1:(m.T)
        _advance!(m)
    end

    if adapt
        evolve!(m.W1, m.dt)            # one Hebbian M-step
        norm!(m)                       # post-update dictionary normalization
    end

    return NGCSimLib.get_value(m.e0.mu), NGCSimLib.get_value(m.e0.L)
end
