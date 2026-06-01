using NGCLearn
using NGCSimLib: get_value, set!, targeted
using Test

# IFCell is deterministic (no PRNG). Hand-computed from the IF dynamics:
#   dv/dt = j*resist_m/tau_m  (gated by refractory mask, NO leak term)
#   spike if v > thr; on spike v <- v_reset, rfr <- 0; tols <- t
# lower_clamp_voltage floors v at v_rest.

@testset "IFCell" begin
    mk(; kw...) = IFCell(; name="I", n_units=2, tau_m=10.0, resist_m=1.0,
        thr=-52.0, v_rest=-65.0, v_reset=-60.0, refract_time=0.0, kw...)

    @testset "construction + shapes" begin
        c = mk()
        @test c isa IFCell
        @test c isa NGCLearn.JaxComponent
        @test size(get_value(c.v)) == (1, 2)
        @test get_value(c.v) ≈ [-65.0 -65.0]      # v_rest
        @test all(get_value(c.s) .== 0.0)
        @test c.intg_flag == 0                     # default euler
        # no leak term ⇒ no g_L / tau_theta fields (struct is the IF subset)
        @test !hasproperty(c, :g_L)
        @test !hasproperty(c, :thr_theta)
    end

    @testset "pure integration (no leak), sub-threshold" begin
        c = mk(; lower_clamp_voltage=false)   # disable clamp to see raw integration
        set!(c.j, [100.0 100.0])
        advance_state!(c, 1.0, 1.0)
        # dv/dt = 100/10 = 10; v = -65 + 10 = -55 < thr(-52) ⇒ no spike.
        # (LIF would subtract a leak; IF does not — same here since rest==v start,
        #  but the point is the dynamics use ONLY current.)
        @test get_value(c.v) ≈ [-55.0 -55.0]
        @test all(get_value(c.s) .== 0.0)
    end

    @testset "supra-threshold spike + reset" begin
        c = mk()
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 4.0)
        # dv/dt = 100; v_raw = -65 + 100 = 35 > -52 ⇒ spike.
        @test all(get_value(c.s) .== 1.0)
        @test get_value(c.v) ≈ [-60.0 -60.0]       # v_reset
        @test get_value(c.rfr) ≈ [0.0 0.0]         # (0+1)*(1-1)
        @test get_value(c.tols) ≈ [4.0 4.0]        # records spike time
    end

    @testset "lower_clamp_voltage floors at v_rest" begin
        c = mk()   # clamp on by default
        set!(c.j, [-1000.0 -1000.0])               # strong negative current
        advance_state!(c, 1.0, 1.0)
        @test all(get_value(c.v) .>= -65.0)        # never below v_rest
    end

    @testset "refractory mask gates integration" begin
        c = mk(; refract_time=5.0)
        # rfr starts at refract_T=5, so mask=(5>=5)=1 first step (can integrate)
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 1.0)                # spikes → rfr resets to 0
        @test all(get_value(c.s) .== 1.0)
        @test get_value(c.rfr) ≈ [0.0 0.0]
        # next step: rfr=0 < refract_T=5 ⇒ mask=0 ⇒ no integration, v stays v_reset-clamped
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 2.0)
        # mask 0 ⇒ dv=0 ⇒ v stays at v_reset (-60), which is > thr? -60 < -52 ⇒ no spike
        @test all(get_value(c.s) .== 0.0)
        @test get_value(c.v) ≈ [-60.0 -60.0]
    end

    @testset "rk2 integrator selected" begin
        c = mk(; integration_type="rk2", lower_clamp_voltage=false)
        @test c.intg_flag == 1
        set!(c.j, [100.0 100.0])
        advance_state!(c, 1.0, 1.0)
        # IF dynamics are current-only & independent of v, so midpoint == euler: -55.
        @test get_value(c.v) ≈ [-55.0 -55.0]
    end

    @testset "reset_state! restores initial values" begin
        c = mk()
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 1.0)
        reset_state!(c)
        @test get_value(c.v) ≈ [-65.0 -65.0]
        @test all(get_value(c.s) .== 0.0)
        @test all(get_value(c.tols) .== 0.0)
    end
end
