# 01_pc_discrim_topology.jl — forward-inference subset of the pc_discrim
# ngc-museum exhibit.
#
# Builds the full PCN topology from
# `ngc-museum/exhibits/pc_discrim/pcn_model.py` using the components we've
# ported so far (RateCell + GaussianErrorCell + DenseSynapse) and runs T=10
# inference steps with random weights. This is the **substrate-validation**
# half of pc_discrim — it proves we can construct and thread signals through
# the multi-layer PC architecture. The **learning** half needs
# `HebbianSynapse` + Adam (next porting target); without it the weights stay
# at their random init so the network output isn't meaningful.
#
# Topology (Whittington & Bogacz 2017):
#
#     z0 ──W1──▶ e1.mu      z1 ──W2──▶ e2.mu      z2 ──W3──▶ e3.mu
#                   ▲                     ▲                     ▲
#     z1.z ─────────┘       z2.z ─────────┘       z3.z ─────────┘
#                   │                     │                     │
#                   ▼                     ▼                     ▼
#                  e1                    e2                    e3
#                   │                     │                     │
#                   │ (dtarget)           │ (dtarget)           │ (dtarget)
#                   ▼                     ▼                     ▼
#                 z1.j_td               z2.j_td               (clamped)
#
#                          ┌─e2.dmu─▶E2─▶z1.j
#                          └─e3.dmu─▶E3─▶z2.j
#
# Run: `julia --project=. examples/01_pc_discrim_topology.jl`

using NGCLearn
using NGCSimLib: get_value, set!, Compartment, Context, post_init!
using Random

# ── Hyperparameters (match upstream pcn_model.py defaults) ───────────────────
in_dim, hid1_dim, hid2_dim, out_dim = 4, 8, 6, 3
T, dt = 10, 1.0     # inference loop length, integration step
tau_m = 10.0
act_fx = "tanh"

# Components, synapses, and wires all live inside a Context so that
# `setup!` fires for every Compartment — that's what assigns each one a
# `root_target` global-state key, which in turn is what makes `>>` wiring
# actually propagate values. Outside a Context, pre-setup Compartments are
# plain value holders (good for unit tests) but `wire!` can't thread them
# because the one-hop target chase returns `nothing`.
Context("pcn") do _ctx
    # ── Layers ───────────────────────────────────────────────────────────────
    z0 = RateCell(; name="z0", n_units=in_dim, tau_m=0.0, act_fx="identity")
    z1 = RateCell(; name="z1", n_units=hid1_dim, tau_m=tau_m, act_fx=act_fx)
    z2 = RateCell(; name="z2", n_units=hid2_dim, tau_m=tau_m, act_fx=act_fx)
    z3 = RateCell(; name="z3", n_units=out_dim, tau_m=0.0, act_fx="identity")
    post_init!.((z0, z1, z2, z3))

    # Gaussian error cells at every layer.
    e1 = GaussianErrorCell(; name="e1", n_units=hid1_dim)
    e2 = GaussianErrorCell(; name="e2", n_units=hid2_dim)
    e3 = GaussianErrorCell(; name="e3", n_units=out_dim)
    post_init!.((e1, e2, e3))

    # Forward generative + feedback synapses. All DenseSynapse for now;
    # the HebbianSynapse port will replace W1..W3 to add the learning rule.
    W1 = DenseSynapse(; name="W1", shape=(in_dim, hid1_dim), key=1)
    W2 = DenseSynapse(; name="W2", shape=(hid1_dim, hid2_dim), key=2)
    W3 = DenseSynapse(; name="W3", shape=(hid2_dim, out_dim), key=3)
    E2 = DenseSynapse(; name="E2", shape=(hid2_dim, hid1_dim), key=4)
    E3 = DenseSynapse(; name="E3", shape=(out_dim, hid2_dim), key=5)
    post_init!.((W1, W2, W3, E2, E3))

    # ── Wiring (mirrors pcn_model.py:153-178) ──────────────────────────────
    # Forward path
    z0.zF >> W1.inputs
    W1.outputs >> e1.mu
    z1.z >> e1.target

    z1.zF >> W2.inputs
    W2.outputs >> e2.mu
    z2.z >> e2.target

    z2.zF >> W3.inputs
    W3.outputs >> e3.mu
    z3.z >> e3.target

    # Feedback path
    e2.dmu >> E2.inputs
    E2.outputs >> z1.j
    e1.dtarget >> z1.j_td

    e3.dmu >> E3.inputs
    E3.outputs >> z2.j
    e2.dtarget >> z2.j_td

    # ── Drive: random input at z0, random label at z3 (clamped) ───────────
    rng = Random.Xoshiro(42)
    x = randn(rng, Float64, 1, in_dim)
    y = randn(rng, Float64, 1, out_dim)
    set!(z0.j, x)
    set!(z3.j, y)    # stateless ⇒ z3.z = j; e3.target reads z3.z

    # ── Run T inference steps ─────────────────────────────────────────────
    println("=== pc_discrim forward inference (T=$T, random weights) ===")
    for t in 1:T
        # Forward synapses
        advance_state!(W1)
        advance_state!(W2)
        advance_state!(W3)
        # Stateless layers reflect their inputs immediately
        advance_state!(z0, dt)
        advance_state!(z3, dt)
        # Error cells gate dmu/dtarget on (target - mu) mismatch
        advance_state!(e1, dt)
        advance_state!(e2, dt)
        advance_state!(e3, dt)
        # Feedback synapses
        advance_state!(E2)
        advance_state!(E3)
        # Hidden rate cells integrate bottom-up + top-down currents over time
        advance_state!(z1, dt)
        advance_state!(z2, dt)

        println(
            "t=$t  ‖e3.dmu‖² = ", round(sum(abs2, get_value(e3.dmu)); digits=4),
            "   ‖z2.z‖² = ", round(sum(abs2, get_value(z2.z)); digits=4),
            "   ‖z1.z‖² = ", round(sum(abs2, get_value(z1.z)); digits=4)
        )
    end

    # ── Final state report ────────────────────────────────────────────────
    println()
    println("=== Final state (T=$T) ===")
    println("z0.zF (input forwarded): ", round.(get_value(z0.zF); digits=3))
    println("z1.zF (hidden 1):        ", round.(get_value(z1.zF); digits=3))
    println("z2.zF (hidden 2):        ", round.(get_value(z2.zF); digits=3))
    println("z3.zF (output / label):  ", round.(get_value(z3.zF); digits=3))
    println()
    println("Free energies (per layer):")
    println("  e1.L = ", round.(get_value(e1.L); digits=4))
    println("  e2.L = ", round.(get_value(e2.L); digits=4))
    println("  e3.L = ", round.(get_value(e3.L); digits=4))
    println()
    println(
        "Substrate validated ✓ — pc_discrim topology constructed + threaded ",
        "$T inference steps inside a Context."
    )
    println(
        "Next: port HebbianSynapse (~365 LOC + Adam) to replace W1/W2/W3 ",
        "and enable end-to-end learning."
    )
end
