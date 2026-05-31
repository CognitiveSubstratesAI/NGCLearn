# pcn.jl — Predictive Coding Network (the pc_discrim exhibit), faithful port of
# ngc-museum/exhibits/pc_discrim/pcn_model.py.
#
# Implements the discriminative PCN of:
#   Whittington, J.C.R. & Bogacz, R. (2017). "An approximation of the error
#   backpropagation algorithm in a predictive coding network with local hebbian
#   synaptic plasticity." Neural Computation 29.5: 1229-1262.
#
# Two coupled sub-networks (mirrors pcn_model.py:86-209):
#   - generative / forward:  z0 -(W1)-> e1, z1 -(W2)-> e2, z2 -(W3)-> e3
#                            e2 -(E2)-> z1 <- e1,  e3 -(E3)-> z2 <- e2
#                            W1..W3 are Hebbian-adapted; E2/E3 are static feedback.
#   - inference / projection: q0 -(Q1)-> q1 -(Q2)-> q2 -(Q3)-> q3, plus eq3.
#                            Q1..Q3 are static; used only to *initialise* latents.
#
# `process!` runs upstream's "PEM" cycle (pcn_model.py:312-375):
#   Projection  — tie Q=W and E=Wᵀ, feed-forward project to seed latents,
#   Expectation — T steps of error-driven latent settling,
#   Maximization — one Hebbian-Adam weight update.
#
# Decisions: this is the model layer (docs/decisions.md §7). EAGER path only —
# the components run via direct dispatch inside a Context (the wired-but-eager
# path the substrate supports today, see §4); no Reactant tracing here.

"""
    PCN

A discriminative predictive-coding network (pc_discrim exhibit). Hold the full
component graph (both the generative and inference sub-networks) plus the
hyperparameters needed by [`process!`](@ref).

Construct with [`PCN(; ...)`](@ref); drive with [`process!`](@ref).
"""
mutable struct PCN
    # Hyperparameters.
    in_dim::Int
    hid1_dim::Int
    hid2_dim::Int
    out_dim::Int
    T::Int
    dt::Float64

    # Generative / forward network.
    z0::RateCell
    z1::RateCell
    z2::RateCell
    z3::RateCell
    e1::GaussianErrorCell
    e2::GaussianErrorCell
    e3::GaussianErrorCell
    W1::HebbianSynapse
    W2::HebbianSynapse
    W3::HebbianSynapse
    E2::DenseSynapse
    E3::DenseSynapse

    # Inference / projection network.
    q0::RateCell
    q1::RateCell
    q2::RateCell
    q3::RateCell
    eq3::GaussianErrorCell
    Q1::DenseSynapse
    Q2::DenseSynapse
    Q3::DenseSynapse
end

"""
    PCN(; in_dim, out_dim, hid1_dim=128, hid2_dim=64, T=10, dt=1.0, tau_m=10.0,
        act_fx="tanh", eta=0.001, wlb=-0.3, wub=0.3, key=nothing, name="Circuit")

Build a pc_discrim PCN. Topology, defaults, and wiring mirror
`PCN.__init__` (pcn_model.py:50-260). All compartments are created inside a
`Context` and `post_init!`-ed so the `>>` wiring propagates on the eager path.

`name` is the Context namespace. Two PCNs with the same name share global-state
slots (Context reuse is faithful upstream behavior) — pass distinct names to
build independent models in one process.
"""
function PCN(;
    in_dim::Integer,
    out_dim::Integer,
    hid1_dim::Integer=128,
    hid2_dim::Integer=64,
    T::Integer=10,
    dt::Real=1.0,
    tau_m::Real=10.0,
    act_fx::AbstractString="tanh",
    eta::Real=0.001,
    wlb::Real=-0.3,
    wub::Real=0.3,
    key::Union{Integer, Nothing}=nothing
)
    optim_type = "adam"
    winit = ("uniform", Float64(wlb), Float64(wub))
    k = key === nothing ? 0 : Int(key)

    local model
    NGCSimLib.Context("Circuit") do _ctx
        # ── Generative / forward network ─────────────────────────────────────
        z0 = RateCell(; name="z0", n_units=in_dim, tau_m=0.0, act_fx="identity")
        z1 = RateCell(; name="z1", n_units=hid1_dim, tau_m=tau_m, act_fx=act_fx,
            prior=("gaussian", 0.0), integration_type="euler")
        e1 = GaussianErrorCell(; name="e1", n_units=hid1_dim)
        z2 = RateCell(; name="z2", n_units=hid2_dim, tau_m=tau_m, act_fx=act_fx,
            prior=("gaussian", 0.0), integration_type="euler")
        e2 = GaussianErrorCell(; name="e2", n_units=hid2_dim)
        z3 = RateCell(; name="z3", n_units=out_dim, tau_m=0.0, act_fx="identity")
        e3 = GaussianErrorCell(; name="e3", n_units=out_dim)

        W1 = HebbianSynapse(; name="W1", shape=(in_dim, hid1_dim), eta=eta,
            weight_init=winit, bias_init=("constant", 0.0), w_bound=0.0,
            optim_type=optim_type, sign_value=-1.0, key=k + 4)
        W2 = HebbianSynapse(; name="W2", shape=(hid1_dim, hid2_dim), eta=eta,
            weight_init=winit, bias_init=("constant", 0.0), w_bound=0.0,
            optim_type=optim_type, sign_value=-1.0, key=k + 5)
        W3 = HebbianSynapse(; name="W3", shape=(hid2_dim, out_dim), eta=eta,
            weight_init=winit, bias_init=("constant", 0.0), w_bound=0.0,
            optim_type=optim_type, sign_value=-1.0, key=k + 6)

        E2 = DenseSynapse(; name="E2", shape=(hid2_dim, hid1_dim),
            weight_init=winit, key=k + 4)
        E3 = DenseSynapse(; name="E3", shape=(out_dim, hid2_dim),
            weight_init=winit, key=k + 5)

        # ── Inference / projection network (pcn_model.py:189-209) ────────────
        q0 = RateCell(; name="q0", n_units=in_dim, tau_m=0.0, act_fx="identity")
        q1 = RateCell(; name="q1", n_units=hid1_dim, tau_m=0.0, act_fx=act_fx)
        q2 = RateCell(; name="q2", n_units=hid2_dim, tau_m=0.0, act_fx=act_fx)
        q3 = RateCell(; name="q3", n_units=out_dim, tau_m=0.0, act_fx="identity")
        eq3 = GaussianErrorCell(; name="eq3", n_units=out_dim)
        Q1 = DenseSynapse(; name="Q1", shape=(in_dim, hid1_dim),
            bias_init=("constant", 0.0), key=k)
        Q2 = DenseSynapse(; name="Q2", shape=(hid1_dim, hid2_dim),
            bias_init=("constant", 0.0), key=k)
        Q3 = DenseSynapse(; name="Q3", shape=(hid2_dim, out_dim),
            bias_init=("constant", 0.0), key=k)

        # CRITICAL ordering: setup-before-wire. Every compartment must be
        # `post_init!`-ed (which assigns its global-state root key) BEFORE any
        # `>>` wiring. `wire!` snapshots the SOURCE's target key at call time
        # (mirrors upstream __rrshift__, compartment.py:150-165); if the source
        # isn't set up yet its target is `nothing`, the wire copies `nothing`,
        # and the later setup! points the dest at its own slot — silently
        # severing the connection. Upstream gets this for free because its
        # metaclass runs _setup at construction; we must order it by hand.
        post_init!.((
            z0, z1, z2, z3, e1, e2, e3, W1, W2, W3, E2, E3,
            q0, q1, q2, q3, eq3, Q1, Q2, Q3
        ))

        # Forward wiring (pcn_model.py:155-166).
        z0.zF >> W1.inputs
        W1.outputs >> e1.mu
        z1.z >> e1.target
        z1.zF >> W2.inputs
        W2.outputs >> e2.mu
        z2.z >> e2.target
        z2.zF >> W3.inputs
        W3.outputs >> e3.mu
        z3.z >> e3.target

        # Feedback wiring (pcn_model.py:167-174).
        e2.dmu >> E2.inputs
        E2.outputs >> z1.j
        e1.dtarget >> z1.j_td
        e3.dmu >> E3.inputs
        E3.outputs >> z2.j
        e2.dtarget >> z2.j_td

        # Hebbian (pre, post) wiring (pcn_model.py:178-186).
        z0.zF >> W1.pre
        e1.dmu >> W1.post
        z1.zF >> W2.pre
        e2.dmu >> W2.post
        z2.zF >> W3.pre
        e3.dmu >> W3.post

        # Inference/projection wiring (pcn_model.py:204-209).
        q0.zF >> Q1.inputs
        Q1.outputs >> q1.j
        q1.zF >> Q2.inputs
        Q2.outputs >> q2.j
        q2.zF >> Q3.inputs
        Q3.outputs >> q3.j

        model = PCN(
            Int(in_dim), Int(hid1_dim), Int(hid2_dim), Int(out_dim),
            Int(T), Float64(dt),
            z0, z1, z2, z3, e1, e2, e3, W1, W2, W3, E2, E3,
            q0, q1, q2, q3, eq3, Q1, Q2, Q3
        )
    end
    return model
end

# Reset every cell (NOT the synapse weights) — mirrors reset_process
# (pcn_model.py:228-240).
function _reset_cells!(m::PCN)
    for c in (m.q0, m.q1, m.q2, m.q3, m.z0, m.z1, m.z2, m.z3)
        reset_state!(c)
    end
    for e in (m.eq3, m.e1, m.e2, m.e3)
        reset_state!(e)
    end
    return m
end

# Tie inference weights to forward weights and feedback weights to their
# transpose (pcn_model.py:320-329).
function _tie_weights!(m::PCN)
    NGCSimLib.set!(m.Q1.weights, NGCSimLib.get_value(m.W1.weights))
    NGCSimLib.set!(m.Q1.biases, NGCSimLib.get_value(m.W1.biases))
    NGCSimLib.set!(m.Q2.weights, NGCSimLib.get_value(m.W2.weights))
    NGCSimLib.set!(m.Q2.biases, NGCSimLib.get_value(m.W2.biases))
    NGCSimLib.set!(m.Q3.weights, NGCSimLib.get_value(m.W3.weights))
    NGCSimLib.set!(m.Q3.biases, NGCSimLib.get_value(m.W3.biases))
    NGCSimLib.set!(m.E2.weights, permutedims(NGCSimLib.get_value(m.W2.weights)))
    NGCSimLib.set!(m.E3.weights, permutedims(NGCSimLib.get_value(m.W3.weights)))
    return m
end

# P-step: pure feed-forward projection through the q-network
# (pcn_model.py:247-255).
function _project!(m::PCN)
    advance_state!(m.q0, m.dt)
    advance_state!(m.Q1)
    advance_state!(m.q1, m.dt)
    advance_state!(m.Q2)
    advance_state!(m.q2, m.dt)
    advance_state!(m.Q3)
    advance_state!(m.q3, m.dt)
    advance_state!(m.eq3, m.dt)
    return m
end

# One E-step: error-driven latent settling (pcn_model.py:214-226).
function _advance!(m::PCN, t)
    advance_state!(m.E2)
    advance_state!(m.E3)
    advance_state!(m.z0, m.dt)
    advance_state!(m.z1, m.dt)
    advance_state!(m.z2, m.dt)
    advance_state!(m.z3, m.dt)
    advance_state!(m.W1)
    advance_state!(m.W2)
    advance_state!(m.W3)
    advance_state!(m.e1, m.dt)
    advance_state!(m.e2, m.dt)
    advance_state!(m.e3, m.dt)
    return m
end

"""
    process!(m::PCN, obs, lab; adapt=true) -> (y_mu_inf, y_mu, EFE)

Run one PEM cycle on a single `(obs, lab)` pair (each a `1×dim` row).
Faithful to `PCN.process` (pcn_model.py:312-375):

  1. reset cells, tie `Q=W` and `E=Wᵀ`;
  2. **P-step** — clamp `obs`/`lab`, feed-forward project, seed latents
     `z1.z=q1.z`, `z2.z=q2.z`, and the output-layer error `e3=eq3`;
  3. **E-step** — `T` rounds of latent settling (clamping `obs`/`lab` each round);
  4. **M-step** (if `adapt`) — one Hebbian-Adam weight update on `W1..W3`.

Returns the projected prediction `y_mu_inf` (= `q3.z` from the P-step), the
settled prediction `y_mu` (= `e3.mu`), and the approximate expected free energy
`EFE = L1 + L2 + L3`.
"""
function process!(m::PCN, obs::AbstractMatrix, lab::AbstractMatrix; adapt::Bool=true)
    eps = 0.001
    _lab = clamp.(lab, eps, 1.0 - eps)

    _reset_cells!(m)
    _tie_weights!(m)

    # P-step: clamp input to z0 & q0, infer-target to eq3, then project.
    NGCSimLib.set!(m.z0.j, obs)
    NGCSimLib.set!(m.q0.j, obs)
    NGCSimLib.set!(m.eq3.target, _lab)
    _project!(m)

    # Seed generative latents from the projected inference states
    # (pcn_model.py:341-346).
    NGCSimLib.set!(m.z1.z, NGCSimLib.get_value(m.q1.z))
    NGCSimLib.set!(m.z2.z, NGCSimLib.get_value(m.q2.z))
    NGCSimLib.set!(m.e3.dmu, NGCSimLib.get_value(m.eq3.dmu))
    NGCSimLib.set!(m.e3.dtarget, NGCSimLib.get_value(m.eq3.dtarget))

    y_mu_inf = NGCSimLib.get_value(m.q3.z)
    EFE = 0.0
    y_mu = zeros(Float64, 1, m.out_dim)

    if adapt
        for ts in 0:(m.T - 1)
            NGCSimLib.set!(m.z0.j, obs)
            NGCSimLib.set!(m.q0.j, obs)
            NGCSimLib.set!(m.z3.j, _lab)   # clamp target to output layer
            _advance!(m, ts)
        end
        y_mu = NGCSimLib.get_value(m.e3.mu)
        EFE =
            _scalar(NGCSimLib.get_value(m.e1.L)) +
            _scalar(NGCSimLib.get_value(m.e2.L)) +
            _scalar(NGCSimLib.get_value(m.e3.L))

        # M-step: one scheduled Hebbian-Adam update.
        evolve!(m.W1, m.dt)
        evolve!(m.W2, m.dt)
        evolve!(m.W3, m.dt)
    end

    return y_mu_inf, y_mu, EFE
end

_scalar(x::AbstractArray) = sum(x)
_scalar(x) = x

"""
    project(m::PCN, obs) -> y

Test-time inference: clamp `obs`, project feed-forward, return the prediction
`q3.z` (no settling, no learning). Mirrors the `adapt_synapses=false` path of
`PCN.process`.
"""
function project(m::PCN, obs::AbstractMatrix)
    _reset_cells!(m)
    _tie_weights!(m)
    NGCSimLib.set!(m.q0.j, obs)
    _project!(m)
    return NGCSimLib.get_value(m.q3.z)
end
