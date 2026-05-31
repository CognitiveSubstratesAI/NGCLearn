using NGCLearn
using NGCSimLib: get_value, set!
using Test

@testset "HebbianSynapse" begin

    # ── Construction defaults + shapes ───────────────────────────────────────
    @testset "construction shapes + Hebbian compartments" begin
        s = HebbianSynapse(; name="H", shape=(3, 2), eta=0.01,
            weight_init=("constant", 0.5), key=1)
        @test s isa HebbianSynapse
        @test s isa NGCLearn.JaxComponent
        @test s.shape == (3, 2)
        @test size(get_value(s.weights)) == (3, 2)
        @test size(get_value(s.biases)) == (1, 2)
        @test size(get_value(s.dWeights)) == (3, 2)
        @test size(get_value(s.dBiases)) == (1, 2)
        @test size(get_value(s.pre)) == (1, 3)
        @test size(get_value(s.post)) == (1, 2)
        @test s.optim_type == "sgd"
        @test s.has_bias == false
    end

    # ── Forward pass identical to DenseSynapse ──────────────────────────────
    @testset "advance_state! forward pass" begin
        s = HebbianSynapse(; name="H", shape=(3, 2), eta=0.01,
            weight_init=("constant", 0.5), key=2)
        set!(s.inputs, [1.0 2.0 3.0])
        advance_state!(s)
        # (1+2+3) * 0.5 = 3 for each output
        @test get_value(s.outputs) ≈ [3.0 3.0]
    end

    # ── compute_update! produces dW = pre' * post ───────────────────────────
    @testset "compute_update! follows the 2-factor Hebbian rule" begin
        s = HebbianSynapse(; name="H", shape=(2, 2), eta=0.0,
            weight_init=("constant", 0.0),
            prior=("constant", 0.0), w_bound=0.0, key=3)
        # pre  = [1 2],  post = [3 4]
        # dW   = pre' * post = [[1*3, 1*4], [2*3, 2*4]] = [[3,4],[6,8]]
        # db   = sum(post, dims=1) = [3 4]
        set!(s.pre, [1.0 2.0])
        set!(s.post, [3.0 4.0])
        compute_update!(s)
        @test get_value(s.dWeights) ≈ [3.0 4.0; 6.0 8.0]
        @test get_value(s.dBiases) ≈ [3.0 4.0]
    end

    # ── sign_value flips the direction (descent) ────────────────────────────
    @testset "sign_value = -1 flips dW sign" begin
        s = HebbianSynapse(; name="H", shape=(2, 2),
            weight_init=("constant", 0.0),
            sign_value=-1.0, w_bound=0.0,
            prior=("constant", 0.0), key=4)
        set!(s.pre, [1.0 1.0])
        set!(s.post, [1.0 1.0])
        compute_update!(s)
        @test all(get_value(s.dWeights) .== -1.0)
    end

    # ── l2 prior contributes -W * λ ─────────────────────────────────────────
    @testset "l2 prior subtracts W * lmbda from dW" begin
        # pre = post = zeros ⇒ Hebbian part = 0, prior part = -W * λ.
        # With W ≡ 0.5 and λ = 0.1, dW = -0.05 everywhere.
        s = HebbianSynapse(; name="H", shape=(2, 2),
            weight_init=("constant", 0.5),
            prior=("l2", 0.1), w_bound=0.0, key=5)
        set!(s.pre, zeros(1, 2))
        set!(s.post, zeros(1, 2))
        compute_update!(s)
        @test all(get_value(s.dWeights) .≈ -0.05)
    end

    # ── evolve! actually moves weights via SGD ──────────────────────────────
    @testset "evolve! with SGD descends along dW (sign_value=-1)" begin
        # With Hebbian rule + sign=-1: dW = -pre' * post = -[[1]] for
        # pre=post=[1]. SGD: W_new = W - η * dW = 0.5 - 0.01 * (-1) = 0.51.
        s = HebbianSynapse(; name="H", shape=(1, 1), eta=0.01,
            weight_init=("constant", 0.5),
            sign_value=-1.0, w_bound=0.0,
            prior=("constant", 0.0), key=6)
        set!(s.pre, [1.0;;])
        set!(s.post, [1.0;;])
        evolve!(s, 1.0)
        @test get_value(s.weights) ≈ [0.51;;]
        @test get_value(s.dWeights) ≈ [-1.0;;]
    end

    # ── evolve! with Adam: first step ≈ W - eta on nonzero grad ─────────────
    @testset "evolve! with Adam steps in the right direction" begin
        # Hebbian rule with sign_value=-1: dW = -1. Adam first step
        # ≈ W - eta * sign(dW) = 0.5 - 0.001 * (-1) ≈ 0.501.
        s = HebbianSynapse(; name="H", shape=(1, 1), eta=0.001,
            weight_init=("constant", 0.5),
            sign_value=-1.0, w_bound=0.0,
            optim_type="adam",
            prior=("constant", 0.0), key=7)
        set!(s.pre, [1.0;;])
        set!(s.post, [1.0;;])
        evolve!(s, 1.0)
        @test get_value(s.weights) ≈ [0.501;;] atol=1e-6
    end

    # ── Bound: w_bound > 0 caps the weights ─────────────────────────────────
    @testset "w_bound + is_nonnegative clamps weights post-update" begin
        # Big Hebbian update with sign_value=-1 would push W well above 1;
        # w_bound=1.0 + is_nonnegative=true clamps to [0, 1].
        s = HebbianSynapse(; name="H", shape=(1, 1), eta=10.0,
            weight_init=("constant", 0.5),
            sign_value=-1.0, w_bound=1.0,
            is_nonnegative=true,
            prior=("constant", 0.0), key=8)
        set!(s.pre, [1.0;;])
        set!(s.post, [10.0;;])
        evolve!(s, 1.0)
        # The unclamped update would be 0.5 + 10*10 = 100.5; clamp to 1.0.
        @test get_value(s.weights) == [1.0;;]
    end

    # ── reset_state! zeros pre/post/dW/db (and outputs) ─────────────────────
    @testset "reset_state! zeros plasticity compartments" begin
        s = HebbianSynapse(; name="H", shape=(2, 2), key=9)
        set!(s.pre, [3.0 3.0])
        set!(s.post, [4.0 4.0])
        set!(s.dWeights, [1.0 1.0; 1.0 1.0])
        set!(s.dBiases, [2.0 2.0])
        set!(s.outputs, [9.0 9.0])
        reset_state!(s)
        @test all(get_value(s.pre) .== 0.0)
        @test all(get_value(s.post) .== 0.0)
        @test all(get_value(s.dWeights) .== 0.0)
        @test all(get_value(s.dBiases) .== 0.0)
        @test all(get_value(s.outputs) .== 0.0)
    end

    # ── prior_type alias: "gaussian" → "ridge", "laplacian" → "lasso" ───────
    @testset "prior_type aliases map upstream-style names" begin
        a = HebbianSynapse(; name="A", shape=(1, 1),
            prior=("gaussian", 0.1), key=10)
        b = HebbianSynapse(; name="B", shape=(1, 1),
            prior=("laplacian", 0.1), key=11)
        @test a.prior_type == "ridge"
        @test b.prior_type == "lasso"
    end
end
