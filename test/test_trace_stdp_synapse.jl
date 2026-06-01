using NGCLearn
using NGCSimLib: get_value, set!, targeted
using Test

# Trace-STDP is deterministic — assert exact matrix products from
#   dW = A_plus*(preTrace - x_tar)ᵀ·postSpike  -  A_minus*preSpikeᵀ·postTrace
# and the evolve! update w ← clip(W + dW*eta - decay, w_eps, w_bound).

@testset "TraceSTDPSynapse" begin
    @testset "construction + shapes" begin
        c = TraceSTDPSynapse(; name="W", shape=(2, 3), A_plus=1.0, A_minus=1.0, key=1)
        @test c isa TraceSTDPSynapse
        @test c isa NGCLearn.JaxComponent
        @test c.shape == (2, 3)
        @test size(get_value(c.weights)) == (2, 3)
        @test size(get_value(c.preSpike)) == (1, 2)
        @test size(get_value(c.postSpike)) == (1, 3)
        @test size(get_value(c.dWeights)) == (2, 3)
    end

    @testset "forward pass: outputs = inputs * W * resist_scale" begin
        c = TraceSTDPSynapse(; name="W", shape=(2, 2), A_plus=1.0, A_minus=1.0,
            weight_init=("constant", 0.5), resist_scale=2.0, key=2)
        set!(c.inputs, [1.0 3.0])
        advance_state!(c)
        # [1 3] * [.5 .5; .5 .5] = [2 2]; *resist_scale 2 = [4 4]
        @test get_value(c.outputs) ≈ [4.0 4.0]
    end

    @testset "additive STDP update (mu = 0)" begin
        c = TraceSTDPSynapse(; name="W", shape=(2, 2), A_plus=1.0, A_minus=1.0,
            weight_init=("constant", 0.5), eta=1.0, w_bound=10.0, key=3)
        set!(c.preTrace, [1.0 2.0])
        set!(c.postSpike, [1.0 0.0])
        set!(c.preSpike, [0.0 0.0])      # zero LTD term
        set!(c.postTrace, [0.0 0.0])
        evolve!(c)
        # dWpost = (preTrace - 0)ᵀ · postSpike = [1;2]*[1 0] = [1 0; 2 0]
        # dWpre  = 0 (preSpike zero). dW = [1 0; 2 0]
        @test get_value(c.dWeights) ≈ [1.0 0.0; 2.0 0.0]
        # w = clip(0.5 + dW*1 - 0, 0, 10) = [1.5 0.5; 2.5 0.5]
        @test get_value(c.weights) ≈ [1.5 0.5; 2.5 0.5]
    end

    @testset "LTD term subtracts (A_minus)" begin
        c = TraceSTDPSynapse(; name="W", shape=(2, 2), A_plus=0.0, A_minus=1.0,
            weight_init=("constant", 5.0), eta=1.0, w_bound=10.0, key=4)
        set!(c.preSpike, [1.0 0.0])
        set!(c.postTrace, [2.0 1.0])
        set!(c.preTrace, [0.0 0.0])      # A_plus=0 anyway
        set!(c.postSpike, [0.0 0.0])
        evolve!(c)
        # dWpre = -(preSpikeᵀ · postTrace) = -([1;0]*[2 1]) = [-2 -1; 0 0]
        @test get_value(c.dWeights) ≈ [-2.0 -1.0; 0.0 0.0]
        @test get_value(c.weights) ≈ [3.0 4.0; 5.0 5.0]
    end

    @testset "x_tar (pretrace_target) shifts LTP" begin
        c = TraceSTDPSynapse(; name="W", shape=(1, 1), A_plus=1.0, A_minus=0.0,
            weight_init=("constant", 0.5), pretrace_target=0.3, eta=1.0,
            w_bound=10.0, key=5)
        set!(c.preTrace, [1.0;;])
        set!(c.postSpike, [1.0;;])
        evolve!(c)
        # dW = (1 - 0.3)*1 = 0.7
        @test get_value(c.dWeights) ≈ [0.7;;]
        @test get_value(c.weights) ≈ [1.2;;]
    end

    @testset "weights clipped to [w_eps, w_bound]" begin
        c = TraceSTDPSynapse(; name="W", shape=(1, 1), A_plus=100.0, A_minus=0.0,
            weight_init=("constant", 0.5), eta=1.0, w_bound=1.0, key=6)
        set!(c.preTrace, [1.0;;])
        set!(c.postSpike, [1.0;;])
        evolve!(c)
        # huge LTP push; clipped to w_bound - w_eps = 1.0
        @test get_value(c.weights) ≈ [1.0;;]
        @test all(get_value(c.weights) .<= 1.0)
    end

    @testset "weight decay (tau_w > 0)" begin
        c = TraceSTDPSynapse(; name="W", shape=(1, 1), A_plus=0.0, A_minus=0.0,
            weight_init=("constant", 4.0), eta=1.0, w_bound=10.0, tau_w=2.0, key=7)
        # no spikes ⇒ dW=0; decay = W/tau_w = 4/2 = 2 ⇒ w = 4 - 2 = 2
        evolve!(c)
        @test get_value(c.dWeights) ≈ [0.0;;]
        @test get_value(c.weights) ≈ [2.0;;]
    end

    @testset "power-law scaling (mu > 0)" begin
        c = TraceSTDPSynapse(; name="W", shape=(1, 1), A_plus=1.0, A_minus=0.0,
            weight_init=("constant", 0.5), mu=2.0, eta=1.0, w_bound=1.0, key=8)
        set!(c.preTrace, [1.0;;])
        set!(c.postSpike, [1.0;;])
        evolve!(c)
        # post_shift = (w_bound - W)^mu = (1-0.5)^2 = 0.25
        # dW = 0.25 * ((preTrace-0)ᵀ·postSpike) * A_plus = 0.25 * 1 = 0.25
        @test get_value(c.dWeights) ≈ [0.25;;]
        @test get_value(c.weights) ≈ [0.75;;]
    end

    @testset "reset_state! zeros spikes/traces/dW" begin
        c = TraceSTDPSynapse(; name="W", shape=(2, 2), A_plus=1.0, A_minus=1.0, key=9)
        set!(c.preTrace, [1.0 1.0])
        set!(c.postSpike, [1.0 1.0])
        evolve!(c)
        reset_state!(c)
        @test all(get_value(c.preSpike) .== 0.0)
        @test all(get_value(c.postTrace) .== 0.0)
        @test all(get_value(c.dWeights) .== 0.0)
    end
end
