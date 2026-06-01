using NGCLearn
using NGCSimLib: get_value, set!, targeted
using Test

# VarTrace is fully deterministic — assert exact values from the decay law:
#   decay = exp(-dt/tau) | 1 - dt/tau | 0   (exp | lin | step)
#   x_tr  = gamma_tr * trace * decay  then  +inputs*a_delta  OR  gated snap.

@testset "VarTrace" begin
    @testset "construction + shapes" begin
        c = VarTrace(; name="T", n_units=3, tau_tr=20.0, a_delta=1.0)
        @test c isa VarTrace
        @test c isa NGCLearn.JaxComponent
        @test (c.tau_tr, c.a_delta, c.P_scale, c.gamma_tr) == (20.0, 1.0, 1.0, 1.0)
        @test size(get_value(c.trace)) == (1, 3)
        @test all(get_value(c.trace) .== 0.0)
    end

    @testset "additive exp decay (a_delta > 0)" begin
        c = VarTrace(; name="T", n_units=2, tau_tr=10.0, a_delta=2.0, decay_type="exp")
        set!(c.inputs, [1.0 0.0])
        advance_state!(c, 1.0)
        # trace 0 → x_tr = 1*0*exp(-0.1) + [1 0]*2 = [2 0]
        @test get_value(c.trace) ≈ [2.0 0.0]
        @test get_value(c.outputs) ≈ [2.0 0.0]   # outputs mirrors trace
        # next step, no input: x_tr = 2 * exp(-1/10) = 2*0.904837...
        set!(c.inputs, [0.0 0.0])
        advance_state!(c, 1.0)
        @test get_value(c.trace) ≈ [2.0 * exp(-0.1) 0.0]
    end

    @testset "linear decay factor" begin
        c = VarTrace(; name="T", n_units=1, tau_tr=4.0, a_delta=1.0, decay_type="lin")
        set!(c.trace, [10.0;;])
        set!(c.outputs, [10.0;;])
        set!(c.inputs, [0.0;;])
        advance_state!(c, 1.0)
        # decay = 1 - 1/4 = 0.75 ⇒ 10 * 0.75 = 7.5
        @test get_value(c.trace) ≈ [7.5;;]
    end

    @testset "step decay zeros the leak (decay = 0)" begin
        c = VarTrace(; name="T", n_units=2, tau_tr=5.0, a_delta=3.0, decay_type="step")
        set!(c.trace, [9.0 9.0])
        set!(c.inputs, [1.0 0.0])
        advance_state!(c, 1.0)
        # leak term 0; additive: x_tr = 0 + [1 0]*3 = [3 0]
        @test get_value(c.trace) ≈ [3.0 0.0]
    end

    @testset "gated snap-to-P_scale (a_delta <= 0)" begin
        c = VarTrace(;
            name="T", n_units=3, tau_tr=10.0, a_delta=0.0, P_scale=1.0, decay_type="exp"
        )
        set!(c.trace, [0.5 0.5 0.5])
        set!(c.inputs, [1.0 0.0 1.0])
        advance_state!(c, 1.0)
        # x_tr = (0.5*exp(-0.1))*(1 - input) + input*P_scale
        d = 0.5 * exp(-0.1)
        @test get_value(c.trace) ≈ [1.0 d 1.0]
    end

    @testset "nearest-neighbor variant (k > 0)" begin
        c = VarTrace(; name="T", n_units=1, tau_tr=10.0, a_delta=2.0, n_nearest_spikes=2,
            decay_type="exp")
        set!(c.trace, [1.0;;])
        set!(c.inputs, [1.0;;])
        advance_state!(c, 1.0)
        # x_tr = 1*exp(-0.1) + 1*(a_delta - trace/k) = exp(-0.1) + (2 - 1/2)
        @test get_value(c.trace) ≈ [exp(-0.1) + 1.5;;]
    end

    @testset "reset_state! zeros trace + outputs" begin
        c = VarTrace(; name="T", n_units=2, tau_tr=10.0, a_delta=1.0)
        set!(c.inputs, [1.0 1.0])
        advance_state!(c, 1.0)
        @test any(get_value(c.trace) .!= 0.0)
        reset_state!(c)
        @test all(get_value(c.trace) .== 0.0)
        @test all(get_value(c.outputs) .== 0.0)
    end
end
