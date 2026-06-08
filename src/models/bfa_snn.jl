# bfa_snn.jl — broadcast/feedback-alignment spiking neural network exhibit.
#
# Faithful port of ngc-museum/exhibits/bfa_snn/bfasnn_model.py.
#
#   Samadi, A., Lillicrap, T.P., & Tweed, D.B. (2017). "Deep learning with dynamic
#   spiking neurons and fixed feedback weights." Neural Computation 29.3: 578-602.
#
# Topology (a 2-layer spiking classifier trained WITHOUT backprop):
#   z0 (Bernoulli encoder) -(W1, Hebbian)-> z1 (sLIF) -(W2, Hebbian)-> z2 (sLIF)
#   z2.s -> e2 (output error vs label).  The output error is broadcast back through
#   a FIXED random matrix E2 (the feedback-alignment trick — NOT W2ᵀ) and gated by
#   z1's surrogate derivative into the hidden teaching signal d1, which drives W1.
#       e2.dmu -> E2 -> d1.target ;   z1.surrogate -> d1.modulator
#       W1.post = d1.dmu ;  W2.post = e2.dmu      (pre = the presynaptic spikes)
#
# `process!` runs a T-step stimulus window (reset → clamp → advance → evolve the
# two Hebbian synapses after a burn-in) and accumulates an output label estimate.
#
# Decisions: model layer (§7 — setup-before-wire, eager hand-ordered loop, no
# MethodProcess); per-model Context `name` for isolation.

# Row-wise softmax (model_utils.softmax is not yet ported; port-on-demand).
function _softmax_rows(x::AbstractMatrix)
    e = exp.(x .- maximum(x; dims=2))
    return e ./ sum(e; dims=2)
end

"""
    BFA_SNN

A Samadi-et-al. (2017) broadcast-feedback-alignment spiking network: a Bernoulli-
encoded input drives two sLIF layers through Hebbian-adapted weights `W1`/`W2`,
with credit assigned by broadcasting the output error through a FIXED random
feedback matrix `E2` (gated by the hidden surrogate derivative) — no backprop.

Construct with [`BFA_SNN(; ...)`](@ref); drive with [`process!`](@ref).
"""
mutable struct BFA_SNN
    in_dim::Int
    hid_dim::Int
    out_dim::Int
    T::Int
    dt::Float64
    burnin_T::Float64

    z0::BernoulliCell         # input encoder
    W1::HebbianSynapse        # input → hidden (adapted)
    z1::SLIFCell              # hidden layer
    W2::HebbianSynapse        # hidden → output (adapted)
    z2::SLIFCell              # output layer
    e2::GaussianErrorCell     # output error (vs label)
    E2::DenseSynapse          # FIXED random feedback (out → hidden)
    d1::GaussianErrorCell     # hidden teaching signal
end

"""
    BFA_SNN(; in_dim, out_dim, hid_dim=1024, T=100, dt=0.25, tau_m=20.0,
            key=nothing, name="Circuit")

Build a BFA-SNN. Topology, hyperparameters, and wiring mirror `BFA_SNN.__init__`
(bfasnn_model.py:58-161). Layer-wise learning rates fold into the Hebbian
`post_wght` (eta1=1/in_dim, eta2=1/hid_dim) with `sign_value=-1`, per Samadi et
al. All compartments are `post_init!`-ed BEFORE any `>>` wiring (decisions §7).
"""
function BFA_SNN(;
    in_dim::Integer,
    out_dim::Integer,
    hid_dim::Integer=1024,
    T::Integer=100,
    dt::Real=0.25,
    tau_m::Real=20.0,
    key::Union{Integer, Nothing}=nothing,
    name::AbstractString="Circuit"
)
    _dt = Float64(dt)
    R_m = 1.0
    v_thr = 0.4
    refract_T = 1.0
    eta1_w = 1.0 / in_dim
    eta2_w = 1.0 / hid_dim
    wI = ("gaussian", 0.0, 0.055)         # centered-Gaussian synapse init
    bI = ("constant", 0.0)                # zero biases
    k = key === nothing ? 0 : Int(key)

    local model
    NGCSimLib.Context(name) do _ctx
        z0 = BernoulliCell(; name="z0", n_units=in_dim, key=k + 0)
        W1 = HebbianSynapse(; name="W1", shape=(in_dim, hid_dim), eta=1.0,
            weight_init=wI, bias_init=bI, sign_value=-1.0, optim_type="sgd",
            w_bound=0.0, pre_wght=1.0, post_wght=eta1_w, is_nonnegative=false, key=k + 1)
        z1 = SLIFCell(; name="z1", n_units=hid_dim, tau_m=tau_m, resist_m=R_m,
            thr=v_thr, resist_inh=0.0, sticky_spikes=true, refract_time=refract_T,
            thr_gain=0.0, thr_leak=0.0, thr_jitter=0.0, key=k + 2)
        W2 = HebbianSynapse(; name="W2", shape=(hid_dim, out_dim), eta=1.0,
            weight_init=wI, bias_init=bI, sign_value=-1.0, optim_type="sgd",
            w_bound=0.0, pre_wght=1.0, post_wght=eta2_w, is_nonnegative=false, key=k + 3)
        z2 = SLIFCell(; name="z2", n_units=out_dim, tau_m=tau_m, resist_m=R_m,
            thr=v_thr, resist_inh=0.0, sticky_spikes=true, refract_time=refract_T,
            thr_gain=0.0, thr_leak=0.0, thr_jitter=0.0, key=k + 4)
        e2 = GaussianErrorCell(; name="e2", n_units=out_dim)
        E2 = DenseSynapse(; name="E2", shape=(out_dim, hid_dim), weight_init=wI,
            bias_init=nothing, key=k + 5)
        d1 = GaussianErrorCell(; name="d1", n_units=hid_dim)

        # SETUP-BEFORE-WIRE (decisions §7).
        post_init!.((z0, W1, z1, W2, z2, e2, E2, d1))

        # Forward path (bfasnn_model.py:121-126).
        z0.outputs >> W1.inputs
        W1.outputs >> z1.j
        z1.s >> W2.inputs
        W2.outputs >> z2.j
        z2.s >> e2.mu
        # Fixed-feedback credit assignment (bfasnn_model.py:128-136).
        e2.dmu >> E2.inputs
        E2.outputs >> d1.target
        z1.surrogate >> d1.modulator       # hidden teaching signal gated by surrogate
        z0.outputs >> W1.pre
        d1.dmu >> W1.post
        z1.s >> W2.pre
        e2.dmu >> W2.post

        model = BFA_SNN(
            Int(in_dim), Int(hid_dim), Int(out_dim), Int(T), _dt, 20.0 * _dt,
            z0, W1, z1, W2, z2, e2, E2, d1
        )
    end
    return model
end

# One simulation step. Advance order mirrors advance_process
# (bfasnn_model.py:149-157): encoder, W1, z1, W2, z2, e2, E2, d1.
function _advance!(m::BFA_SNN, t)
    advance_state!(m.z0, t)               # BernoulliCell sig: (t)
    advance_state!(m.W1)                  # HebbianSynapse forward
    advance_state!(m.z1, m.dt, t)         # SLIFCell sig: (dt, t)
    advance_state!(m.W2)
    advance_state!(m.z2, m.dt, t)
    advance_state!(m.e2, m.dt)            # GaussianErrorCell sig: (dt)
    advance_state!(m.E2)                  # DenseSynapse forward
    advance_state!(m.d1, m.dt)
    return m
end

function _reset!(m::BFA_SNN)
    reset_state!(m.z0)
    reset_state!(m.W1)
    reset_state!(m.z1)
    reset_state!(m.W2)
    reset_state!(m.z2)
    reset_state!(m.e2)
    reset_state!(m.E2)
    reset_state!(m.d1)
    return m
end

"""
    process!(m::BFA_SNN, obs, lab; adapt=true, input_gain=0.25) -> (latent, yMu, yCnt)

Process one observation/label pair over a T-step window (bfasnn_model.py:219-299).
Reset → for each step clamp the gain-scaled `obs` and the label, advance, and
(after the `burnin_T` warm-up, when `adapt`) evolve `W1`/`W2`. Returns the summed
hidden spikes (`latent`), the softmax label estimate (`yMu`, `1×out_dim`), and the
raw output spike counts (`yCnt`). Online only (batch size 1).
"""
function process!(m::BFA_SNN, obs::AbstractMatrix, lab::AbstractMatrix;
    adapt::Bool=true, input_gain::Real=0.25)
    size(obs, 1) == 1 || error("BFA_SNN.process!: batch size must be 1")
    _obs = obs .* input_gain
    _reset!(m)

    latent = zeros(Float64, 1, m.hid_dim)
    yMu = zeros(Float64, 1, m.out_dim)
    yCnt = zeros(Float64, 1, m.out_dim)
    T_learn = 0.0
    for ts in 1:(m.T - 1)
        NGCSimLib.set!(m.z0.inputs, _obs)        # clamp obs (encoder consumes it)
        NGCSimLib.set!(m.e2.target, lab)         # clamp label into the output error
        _advance!(m, Float64(ts) * m.dt)
        curr_t = ts * m.dt
        if adapt && curr_t > m.burnin_T
            evolve!(m.W1, m.dt)
            evolve!(m.W2, m.dt)
        end
        yCnt = yCnt .+ NGCSimLib.get_value(m.z2.s)
        if curr_t > m.burnin_T
            T_learn += 1.0
            yMu = yMu .+ NGCSimLib.get_value(m.z2.s)
        end
        latent = latent .+ NGCSimLib.get_value(m.z1.s)
    end
    yMu = _softmax_rows(yMu ./ max(T_learn, 1.0))
    return latent, yMu, yCnt
end
