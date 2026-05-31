using NGCLearn
using NGCSimLib: get_value
using Random
using Test

# Acceptance test for the faithful pc_discrim PCN (Whittington-Bogacz 2017).
# Deterministic (fixed PRNG key) — drives the full PEM loop (project → settle →
# Hebbian-Adam update) on a tiny linearly-separable 2-class task and checks that
# (a) free energy / output error genuinely decrease with training, and
# (b) the trained network classifies every training sample correctly.
#
# Expected numbers were observed empirically before being asserted (see the
# session notes): eta=0.002, 50 epochs ⇒ mean ‖lab−y_mu‖² falls from ~0.42 to
# <0.08 and test-time projection accuracy is 4/4.

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

@testset "PCN integration (pc_discrim)" begin
    @testset "construction + shapes" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7)
        @test m isa PCN
        @test (m.in_dim, m.hid1_dim, m.hid2_dim, m.out_dim) == (4, 8, 6, 2)
        y = project(m, _PCN_X[1])
        @test size(y) == (1, 2)
        @test all(isfinite, y)
    end

    @testset "process! returns finite outputs + correct shapes" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7)
        y_inf, y_mu, EFE = process!(m, _PCN_X[1], _PCN_Y[1])
        @test size(y_inf) == (1, 2)
        @test size(y_mu) == (1, 2)
        @test isfinite(EFE)
        @test EFE <= 0.0    # EFE = sum of Gaussian log-densities (each ≤ 0)
    end

    @testset "weight tying: E = Wᵀ, Q = W after a step" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, key=7)
        process!(m, _PCN_X[1], _PCN_Y[1])
        # After process! the ties were (re)applied at the start of the call.
        @test get_value(m.E3.weights) ≈ permutedims(get_value(m.W3.weights))
        @test get_value(m.Q3.weights) ≈ get_value(m.W3.weights)
        @test size(get_value(m.E3.weights)) == (m.out_dim, m.hid2_dim)
    end

    @testset "learning reduces output error + EFE rises toward 0" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, tau_m=10.0,
            act_fx="tanh", eta=0.002, key=7)
        epochs = 50
        first_err = 0.0
        last_err = 0.0
        first_efe = 0.0
        last_efe = 0.0
        for epoch in 1:epochs
            errs = Float64[]
            efes = Float64[]
            for i in 1:length(_PCN_X)
                _, y_mu, EFE = process!(m, _PCN_X[i], _PCN_Y[i])
                push!(errs, sum(abs2, _PCN_Y[i] .- y_mu))
                push!(efes, EFE)
            end
            @test all(isfinite, errs)      # no divergence at the chosen eta
            me = sum(errs) / length(errs)
            mf = sum(efes) / length(efes)
            if epoch == 1
                first_err, first_efe = me, mf
            elseif epoch == epochs
                last_err, last_efe = me, mf
            end
        end
        # Output prediction error at least halves over training.
        @test last_err < 0.5 * first_err
        # Free energy (negative) climbs toward 0 as errors shrink.
        @test last_efe > first_efe
    end

    @testset "trained network classifies the training set" begin
        m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, tau_m=10.0,
            act_fx="tanh", eta=0.002, key=7)
        for _ in 1:50, i in 1:length(_PCN_X)
            process!(m, _PCN_X[i], _PCN_Y[i])
        end
        correct = 0
        for i in 1:length(_PCN_X)
            p = project(m, _PCN_X[i])
            correct += (argmax(vec(p)) == argmax(vec(_PCN_Y[i])))
        end
        @test correct == length(_PCN_X)   # deterministic ⇒ exact 4/4
    end
end
