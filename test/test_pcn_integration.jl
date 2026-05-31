using NGCLearn
using NGCSimLib: get_value, targeted
using Test

# Acceptance test for the faithful pc_discrim PCN (Whittington-Bogacz 2017).
# Deterministic (fixed PRNG key). Drives the full PEM loop (project → settle →
# Hebbian-Adam update) on a tiny linearly-separable 2-class task.
#
# Every asserted number was OBSERVED from a real run before being written here
# (no aspirational thresholds): with eta=0.002, key=7, 60 epochs the mean output
# error falls 0.707 → 0.0009 and EFE rises -0.386 → -0.0006; test-time
# projection classifies all four training samples (4/4).
#
# Critical substrate detail this guards (see docs/decisions.md §7): the model
# must `post_init!` every component BEFORE any `>>` wiring — `wire!` snapshots
# the source's target key at call time, so wiring pre-setup silently severs the
# connection (output stays 0, nothing learns). The "wires are live" testset
# below is the regression guard for exactly that.

# label = one-hot(sum(x[1:2]) vs sum(x[3:4])).
const _PCN_X = [
    reshape(Float64[1, 1, 0, 0], 1, 4),
    reshape(Float64[0, 0, 1, 1], 1, 4),
    reshape(Float64[1, 0, 0, 1], 1, 4),
    reshape(Float64[0, 1, 1, 0], 1, 4)
]
const _PCN_Y = [
    reshape(Float64[1, 0], 1, 2),
    reshape(Float64[0, 1], 1, 2),
    reshape(Float64[1, 0], 1, 2),
    reshape(Float64[0, 1], 1, 2)
]

function _train_pcn(; epochs=60, eta=0.002, name="train")
    m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, tau_m=10.0,
        act_fx="tanh", eta=eta, key=7, name=name)
    hist = NamedTuple{(:err, :efe), Tuple{Float64, Float64}}[]
    for _ in 1:epochs
        errs = Float64[]
        efes = Float64[]
        for i in 1:length(_PCN_X)
            _, y_mu, EFE = process!(m, _PCN_X[i], _PCN_Y[i])
            push!(errs, sum(abs2, _PCN_Y[i] .- y_mu))
            push!(efes, EFE)
        end
        push!(hist, (err=sum(errs) / length(errs), efe=sum(efes) / length(efes)))
    end
    return m, hist
end

@testset "PCN integration (pc_discrim)" begin
    @testset "construction + shapes" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7, name="ctor")
        @test m isa PCN
        @test (m.in_dim, m.hid1_dim, m.hid2_dim, m.out_dim) == (4, 8, 6, 2)
        y = project(m, _PCN_X[1])
        @test size(y) == (1, 2)
        @test all(isfinite, y)
    end

    @testset "wires are live (setup-before-wire regression guard)" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7, name="wires")
        # If wiring happened before post_init!, these would all read `false`.
        @test targeted(m.W1.inputs)    # z0.zF -> W1.inputs
        @test targeted(m.e1.mu)        # W1.outputs -> e1.mu
        @test targeted(m.W1.pre)       # z0.zF -> W1.pre
        @test targeted(m.Q1.inputs)    # q0.zF -> Q1.inputs
        @test targeted(m.z1.j)         # E2.outputs -> z1.j
    end

    @testset "process! returns finite outputs + correct shapes" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7, name="proc")
        y_inf, y_mu, EFE = process!(m, _PCN_X[1], _PCN_Y[1])
        @test size(y_inf) == (1, 2)
        @test size(y_mu) == (1, 2)
        @test isfinite(EFE)
        @test EFE <= 0.0    # EFE = sum of Gaussian log-densities (each ≤ 0)
        @test all(isfinite, y_mu)
        @test sum(abs, y_mu) > 0.0   # forward path actually propagates (not zeros)
    end

    @testset "weight tying: E = Wᵀ, Q = W (no M-step)" begin
        # Tie holds when no weight update runs (adapt=false). With adapt=true the
        # M-step evolve! moves W AFTER the tie, so E/Q intentionally diverge from
        # the post-update W — hence this checks the projection-only path.
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7, name="tie")
        process!(m, _PCN_X[1], _PCN_Y[1]; adapt=false)
        @test get_value(m.E3.weights) ≈ permutedims(get_value(m.W3.weights))
        @test get_value(m.Q3.weights) ≈ get_value(m.W3.weights)
        @test size(get_value(m.E3.weights)) == (m.out_dim, m.hid2_dim)
    end

    @testset "learning reduces output error + EFE rises toward 0" begin
        _, hist = _train_pcn(; epochs=60, eta=0.002, name="learn")
        @test all(h -> isfinite(h.err) && isfinite(h.efe), hist)  # no divergence
        # Observed: err 0.707 → 0.0009, EFE -0.386 → -0.0006.
        @test hist[1].err > 0.5
        @test hist[end].err < 0.05            # error collapses by >10x
        @test hist[end].err < 0.1 * hist[1].err
        @test hist[end].efe > hist[1].efe     # free energy climbs toward 0
        @test hist[end].efe > -0.05
    end

    @testset "trained network classifies the training set" begin
        m, _ = _train_pcn(; epochs=60, eta=0.002, name="classify")
        correct = 0
        for i in 1:length(_PCN_X)
            p = project(m, _PCN_X[i])
            correct += (argmax(vec(p)) == argmax(vec(_PCN_Y[i])))
        end
        @test correct == length(_PCN_X)   # deterministic ⇒ exact 4/4
    end
end
