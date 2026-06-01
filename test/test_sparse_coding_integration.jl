using NGCLearn
using NGCSimLib: get_value, targeted
using Test

# Integration test for the Olshausen & Field (1996) sparse-coding exhibit.
# Deterministic (fixed key, RateCell is deterministic). Verifies structural
# invariants + that the model reconstructs its input and learns a normalized
# dictionary. Numbers OBSERVED from real runs before being asserted (in=8,
# hid=12, key=3 ⇒ recon tracks the input pattern, row-L2 norms → 1).

const _SC_OBS = reshape(Float64[1, 0, 1, 0, 1, 0, 1, 0], 1, 8)

function _train_sc(model_type; iters=20, key=3)
    m = SparseCoding(; in_dim=8, hid_dim=12, T=50, model_type=model_type,
        key=key, name="sc_" * model_type * "_" * string(key))
    local mu, L
    for _ in 1:iters
        mu, L = process!(m, _SC_OBS; adapt=true)
    end
    return m, mu, L
end

@testset "SparseCoding integration (olshausen_sc)" begin
    @testset "construction + shapes (both variants)" begin
        for mt in ("sc_cauchy", "ista")
            m = SparseCoding(; in_dim=8, hid_dim=12, T=50, model_type=mt, key=1,
                name="sc_ctor_" * mt)
            @test m isa SparseCoding
            @test m.model_type == mt
            @test size(get_value(m.W1.weights)) == (12, 8)   # hid × in
            @test size(get_value(m.E1.weights)) == (8, 12)   # in × hid (feedback)
        end
    end

    @testset "invalid model_type errors" begin
        @test_throws ErrorException SparseCoding(; in_dim=4, hid_dim=4,
            model_type="bogus", name="sc_bad")
    end

    @testset "wires are live (setup-before-wire guard)" begin
        m = SparseCoding(; in_dim=8, hid_dim=12, T=10, key=1, name="sc_wires")
        @test targeted(m.W1.inputs)    # z1.zF → W1.inputs
        @test targeted(m.e0.mu)        # W1.outputs → e0.mu
        @test targeted(m.E1.inputs)    # e0.dmu → E1.inputs
        @test targeted(m.z1.j)         # E1.outputs → z1.j
        @test targeted(m.W1.pre)       # z1.zF → W1.pre
        @test targeted(m.W1.post)      # e0.dmu → W1.post
    end

    @testset "process! ties E1 = W1ᵀ" begin
        m = SparseCoding(; in_dim=8, hid_dim=12, T=20, key=2, name="sc_tie")
        process!(m, _SC_OBS; adapt=true)
        # E1 is pinned to W1ᵀ at the START of process!, before the M-step moves W1.
        # So check the tie on a no-adapt run (W1 unchanged through the call).
        m2 = SparseCoding(; in_dim=8, hid_dim=12, T=20, key=2, name="sc_tie2")
        process!(m2, _SC_OBS; adapt=false)
        @test get_value(m2.E1.weights) ≈ permutedims(get_value(m2.W1.weights))
    end

    @testset "reconstruction tracks the input + finite loss" begin
        m, mu, L = _train_sc("sc_cauchy"; iters=20)
        @test size(mu) == (1, 8)
        @test all(isfinite, mu)
        @test isfinite(L)
        # Observed: recon tracks obs — "on" units (idx 1,3,5,7) end clearly above
        # "off" units (idx 2,4,6,8).
        on_mean = (mu[1] + mu[3] + mu[5] + mu[7]) / 4
        off_mean = (mu[2] + mu[4] + mu[6] + mu[8]) / 4
        @test on_mean > off_mean + 0.3
    end

    @testset "learning + L2 dictionary normalization" begin
        m = SparseCoding(; in_dim=8, hid_dim=12, T=50, key=3, name="sc_norm")
        W0 = copy(get_value(m.W1.weights))
        process!(m, _SC_OBS; adapt=true)
        @test get_value(m.W1.weights) != W0                      # W1 learns
        rownorms = vec(sqrt.(sum(abs2, get_value(m.W1.weights); dims=2)))
        @test all(≈(1.0; atol=1e-6), rownorms)                   # rows unit-L2
    end

    @testset "adapt=false leaves W1 unchanged" begin
        m = SparseCoding(; in_dim=8, hid_dim=12, T=30, key=4, name="sc_infer")
        W0 = copy(get_value(m.W1.weights))
        process!(m, _SC_OBS; adapt=false)
        @test get_value(m.W1.weights) == W0
    end

    @testset "ista variant also reconstructs" begin
        m, mu, L = _train_sc("ista"; iters=20)
        @test all(isfinite, mu)
        on_mean = (mu[1] + mu[3] + mu[5] + mu[7]) / 4
        off_mean = (mu[2] + mu[4] + mu[6] + mu[8]) / 4
        @test on_mean > off_mean + 0.3
    end

    @testset "determinism under fixed key" begin
        _, mu_a, _ = _train_sc("sc_cauchy"; iters=15, key=9)
        _, mu_b, _ = _train_sc("sc_cauchy"; iters=15, key=9)
        @test mu_a ≈ mu_b
    end
end
