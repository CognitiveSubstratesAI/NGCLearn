using NGCLearn
using Test

@testset "Optim (SGD + Adam)" begin

    # ── SGD ──────────────────────────────────────────────────────────────────
    @testset "sgd_init returns time_step=0" begin
        state = sgd_init([zeros(2, 3)])
        @test state.time_step == 0.0
    end

    @testset "sgd_step descends: θ_new = θ - η * update" begin
        theta = Matrix{Float64}[[1.0 2.0; 3.0 4.0]]
        upd = Matrix{Float64}[[1.0 1.0; 1.0 1.0]]
        state = sgd_init(theta)
        new_state, new_theta = sgd_step(state, theta, upd; eta=0.1)
        @test new_theta[1] ≈ [0.9 1.9; 2.9 3.9]
        @test new_state.time_step == 1.0
        # repeat: keeps stepping
        new_state2, new_theta2 = sgd_step(new_state, new_theta, upd; eta=0.1)
        @test new_theta2[1] ≈ [0.8 1.8; 2.8 3.8]
        @test new_state2.time_step == 2.0
    end

    # ── Adam ─────────────────────────────────────────────────────────────────
    @testset "adam_init returns zeroed moments + time_step=0" begin
        theta = [zeros(2, 2), zeros(1, 2)]
        state = adam_init(theta)
        @test all(state.g1[1] .== 0.0)
        @test all(state.g1[2] .== 0.0)
        @test all(state.g2[1] .== 0.0)
        @test state.time_step == 0.0
    end

    @testset "adam_step first step ≈ -eta * sign(grad) for nonzero grad" begin
        # At step 1 with g1=g2=0:
        #   _g1 = (1-β1)*upd; g1_unb = _g1 / (1 - β1^1) = upd
        #   _g2 = (1-β2)*upd^2; g2_unb = _g2 / (1 - β2^1) = upd^2
        #   step = eta * upd / (sqrt(upd^2) + eps) ≈ eta * sign(upd)
        theta = Matrix{Float64}[fill(1.0, 1, 2)]
        upd = Matrix{Float64}[fill(0.5, 1, 2)]
        state = adam_init(theta)
        new_state, new_theta = adam_step(state, theta, upd; eta=0.01)
        # ≈ 1.0 - 0.01 = 0.99 (with eps tiny, sign(upd)=1)
        @test all(new_theta[1] .≈ 1.0 - 0.01)
        @test new_state.time_step == 1.0
        # 1st moment populated
        @test all(new_state.g1[1] .≈ 0.1 * 0.5)
    end

    @testset "adam_step multi-param round-trip" begin
        theta = Matrix{Float64}[ones(2, 3), ones(1, 3)]
        upd = Matrix{Float64}[ones(2, 3), ones(1, 3)]
        state = adam_init(theta)
        new_state, new_theta = adam_step(state, theta, upd; eta=0.001)
        @test length(new_theta) == 2
        @test size(new_theta[1]) == (2, 3)
        @test size(new_theta[2]) == (1, 3)
    end

    # ── Dispatch ────────────────────────────────────────────────────────────
    @testset "get_opt_{init,step}_fn dispatch" begin
        @test get_opt_init_fn("sgd") === sgd_init
        @test get_opt_init_fn("adam") === adam_init
        @test get_opt_step_fn("sgd") isa Function
        @test get_opt_step_fn("adam") isa Function
        @test_throws ErrorException get_opt_init_fn("nope")
        @test_throws ErrorException get_opt_step_fn("nope")
    end
end
