# dc_snn.jl — Diehl & Cook (2015) spiking neural network exhibit.
#
# Faithful port of ngc-museum/exhibits/diehl_cook_snn/dcsnn_model.py.
#
#   Diehl, P.U. & Cook, M. (2015). "Unsupervised learning of digit recognition
#   using spike-timing-dependent plasticity." Front. Comput. Neurosci. 9:99.
#
# Topology:
#   z0 (Poisson encoder) -(W1, STDP)-> z1e (excitatory LIF)
#   z1e -(W1ei, fixed eye)-> z1i (inhibitory LIF) -(W1ie, fixed hollow)-> z1e
#   z1e.j = Summation(W1.outputs, W1ie.outputs)   # feedforward + lateral inhib
#   STDP traces: tr0 over z0 spikes (pre), tr1 over z1e spikes (post)
#
# `process!` runs a T-step stimulus window (reset → clamp → advance loop →
# evolve STDP each step → L1-normalize W1), online (batch size 1).
#
# Decisions: model layer (§7 — setup-before-wire, eager hand-ordered loop, no
# MethodProcess); per-model Context `name` for isolation.

"""
    DC_SNN

A Diehl & Cook (2015) unsupervised-STDP spiking network: a Poisson-encoded input
drives an excitatory LIF layer through STDP-adapted weights `W1`, with fixed
lateral inhibition (`W1ei`/`W1ie`) implementing winner-take-all competition.

Construct with [`DC_SNN(; ...)`](@ref); drive with [`process!`](@ref).
"""
mutable struct DC_SNN
    in_dim::Int
    hid_dim::Int
    T::Int
    dt::Float64
    wNorm::Float64

    z0::PoissonCell           # input encoder
    W1::TraceSTDPSynapse      # feedforward, STDP-adapted
    z1e::LIFCell              # excitatory layer
    z1i::LIFCell              # inhibitory layer
    W1ie::DenseSynapse        # inhibitory→excitatory (fixed, hollow, negative)
    W1ei::DenseSynapse        # excitatory→inhibitory (fixed, eye, positive)
    tr0::VarTrace             # pre-synaptic trace (over z0 spikes)
    tr1::VarTrace             # post-synaptic trace (over z1e spikes)
end

"""
    DC_SNN(; in_dim, hid_dim=100, T=200, dt=1.0, key=nothing, name="Circuit")

Build a DC-SNN. Topology, hyperparameters, and wiring mirror `DC_SNN.__init__`
(dcsnn_model.py:46-196). All compartments are `post_init!`-ed BEFORE any `>>`
wiring (decisions §7). Pass distinct `name`s to build independent models in one
process (Context reuse shares global-state slots otherwise).
"""
function DC_SNN(;
    in_dim::Integer,
    hid_dim::Integer=100,
    T::Integer=200,
    dt::Real=1.0,
    key::Union{Integer, Nothing}=nothing,
    name::AbstractString="Circuit"
)
    tau_m_e = 100.500896468
    tau_m_i = 100.500896468
    tau_tr = 20.0
    Aplus = 1e-2
    Aminus = 1e-4
    wNorm = 78.4
    k = key === nothing ? 0 : Int(key)
    _dt = Float64(dt)

    local model
    NGCSimLib.Context(name) do _ctx
        z0 = PoissonCell(; name="z0", n_units=in_dim, target_freq=63.75, key=k + 0)
        W1 = TraceSTDPSynapse(; name="W1", shape=(in_dim, hid_dim),
            A_plus=Aplus, A_minus=Aminus, eta=1.0, pretrace_target=0.0,
            weight_init=("uniform", 0.0, 0.3), key=k + 1)
        z1e = LIFCell(; name="z1e", n_units=hid_dim, tau_m=tau_m_e,
            resist_m=tau_m_e / _dt, thr=-52.0, v_rest=-65.0, v_reset=-60.0,
            tau_theta=1e7, theta_plus=0.05, refract_time=5.0, one_spike=true,
            key=k + 2)
        z1i = LIFCell(; name="z1i", n_units=hid_dim, tau_m=tau_m_i,
            resist_m=tau_m_i / _dt, thr=-40.0, v_rest=-60.0, v_reset=-45.0,
            tau_theta=0.0, refract_time=5.0, one_spike=false, key=k + 3)
        # Fixed lateral synapses: ie = hollow negative (all-to-all but self),
        # ei = eye positive (one-to-one). eta=0 ⇒ never adapted (DenseSynapse).
        W1ie = DenseSynapse(; name="W1ie", shape=(hid_dim, hid_dim),
            weight_init=("constant", -120.0, :hollow), key=k + 4)
        W1ei = DenseSynapse(; name="W1ei", shape=(hid_dim, hid_dim),
            weight_init=("constant", 22.5, :eye), key=k + 5)
        tr0 = VarTrace(; name="tr0", n_units=in_dim, tau_tr=tau_tr,
            decay_type="exp", a_delta=0.0, key=k + 6)
        tr1 = VarTrace(; name="tr1", n_units=hid_dim, tau_tr=tau_tr,
            decay_type="exp", a_delta=0.0, key=k + 7)

        # SETUP-BEFORE-WIRE (decisions §7): post_init! all, THEN `>>`.
        post_init!.((z0, W1, z1e, z1i, W1ie, W1ei, tr0, tr1))

        # Forward + lateral wiring (dcsnn_model.py:153-166).
        z0.outputs >> W1.inputs
        z1i.s >> W1ie.inputs
        NGCSimLib.Summation(W1.outputs, W1ie.outputs) >> z1e.j  # ff + lateral inhib
        z1e.s >> W1ei.inputs
        W1ei.outputs >> z1i.j

        # STDP plumbing: traces + spikes into W1's plasticity compartments.
        z0.outputs >> tr0.inputs
        z1e.s >> tr1.inputs
        tr0.trace >> W1.preTrace
        z0.outputs >> W1.preSpike
        tr1.trace >> W1.postTrace
        z1e.s >> W1.postSpike

        model = DC_SNN(
            Int(in_dim), Int(hid_dim), Int(T), _dt, wNorm,
            z0, W1, z1e, z1i, W1ie, W1ei, tr0, tr1
        )
    end
    return model
end

"""
    norm!(m::DC_SNN) -> DC_SNN

Rescale each column of the feedforward weight matrix `W1` to L1-norm `m.wNorm`
(78.4 by default). Called at the end of [`process!`](@ref) when adapting, to
keep total incoming weight per excitatory unit bounded — the homeostatic
constraint of Diehl & Cook (2015). Mirrors `DC_SNN.norm` (dcsnn_model.py:197-198).
"""
function norm!(m::DC_SNN)
    W = NGCSimLib.get_value(m.W1.weights)
    NGCSimLib.set!(m.W1.weights, normalize_matrix(W, m.wNorm; order=1, axis=1))
    return m
end

# One simulation step: advance order mirrors advance_proc (dcsnn_model.py:168-176).
# Synapses first (so outputs feed the Summation into z1e.j this step), then
# encoder, neurons, traces.
function _advance!(m::DC_SNN, t)
    advance_state!(m.W1)
    advance_state!(m.W1ie)
    advance_state!(m.W1ei)
    advance_state!(m.z0, t, m.dt)        # PoissonCell sig: (t, dt)
    advance_state!(m.z1e, m.dt, t)       # LIFCell sig: (dt, t)
    advance_state!(m.z1i, m.dt, t)
    advance_state!(m.tr0, m.dt)          # VarTrace sig: (dt)
    advance_state!(m.tr1, m.dt)
    return m
end

function _reset!(m::DC_SNN)
    reset_state!(m.z0)
    reset_state!(m.z1e)
    reset_state!(m.z1i)
    reset_state!(m.tr0)
    reset_state!(m.tr1)
    reset_state!(m.W1)
    reset_state!(m.W1ie)
    reset_state!(m.W1ei)
    return m
end

"""
    process!(m::DC_SNN, obs; adapt=true) -> Matrix

Process one observation (`1×in_dim` row) over a T-step stimulus window. Mirrors
`DC_SNN.process` (dcsnn_model.py:265-309): reset → clamp `obs` into the Poisson
encoder → T advance steps (each runs STDP `evolve!` when `adapt`) → L1-normalize
`W1`. Returns the excitatory spike counts (`1×hid_dim`) accumulated over the
window. Online only (batch size 1).
"""
function process!(m::DC_SNN, obs::AbstractMatrix; adapt::Bool=true)
    size(obs, 1) == 1 || error("DC_SNN.process!: batch size must be 1, got $(size(obs,1))")
    _reset!(m)
    NGCSimLib.set!(m.z0.inputs, obs)

    spike_counts = zeros(Float64, 1, m.hid_dim)
    for ts in 0:(m.T - 1)
        NGCSimLib.set!(m.z0.inputs, obs)   # re-clamp each step (encoder consumes it)
        _advance!(m, Float64(ts))
        if adapt
            evolve!(m.W1)
        end
        spike_counts = spike_counts .+ NGCSimLib.get_value(m.z1e.s)
    end
    if adapt
        norm!(m)
    end
    return spike_counts
end
