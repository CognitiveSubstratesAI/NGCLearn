using NGCLearn
using NGCSimLib: get_value, set!
using Test

# These tests drive the EAGER path: pre-`setup!` compartments behave as plain
# mutable value holders (get_value → initial_value, set! → writes it), so
# `advance_state!`/`reset_state!` can be verified standalone with no Context and
# no GlobalState contamination across tests. The expected numbers are computed
# by hand from the upstream dynamics. See docs/decisions.md #5 for why the JIT
# (Parser/Reactant) path is a separate, later phase.

@testset "LIFCell" begin
    # Common, threshold-dynamics-OFF config (tau_theta = 0) for clean arithmetic.
    mk(; kw...) = LIFCell(;
        name="L", n_units=2, tau_m=10.0, resist_m=1.0, thr=-52.0, v_rest=-65.0,
        v_reset=-60.0, conduct_leak=1.0, tau_theta=0.0, refract_time=5.0, kw...
    )

    @testset "construction + shapes" begin
        c = mk()
        @test c isa LIFCell
        @test c isa NGCLearn.JaxComponent
        @test size(get_value(c.v)) == (1, 2)
        @test get_value(c.v) ≈ [-65.0 -65.0]          # rest + v_rest
        @test get_value(c.rfr) ≈ [5.0 5.0]            # rest + refract_T
        @test all(get_value(c.s) .== 0.0)
        @test c.intg_flag == 0                        # default "euler"
    end

    @testset "sub-threshold integration (no spike)" begin
        c = mk()
        set!(c.j, [100.0 100.0])
        advance_state!(c, 1.0, 1.0)                   # dt = 1, t = 1
        # dv/dt = (v_rest - v)*g_L + j  = (−65−(−65))*1 + 100 = 100; /tau_m → 10
        # v_next = −65 + 10*1 = −55, which is < threshold (−52) ⇒ no spike.
        @test get_value(c.v) ≈ [-55.0 -55.0]
        @test all(get_value(c.s) .== 0.0)
        @test get_value(c.rfr) ≈ [6.0 6.0]            # (5 + 1)*(1 − 0)
        @test all(get_value(c.tols) .== 0.0)          # no spike ⇒ unchanged
    end

    @testset "supra-threshold spike + reset-to-v_reset" begin
        c = mk()
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 7.0)                   # t = 7
        # dv/dt = 1000/10 = 100; v_raw = −65 + 100 = 35 > −52 ⇒ spike.
        @test all(get_value(c.s) .== 1.0)
        @test get_value(c.v) ≈ [-60.0 -60.0]          # v_reset
        @test get_value(c.rfr) ≈ [0.0 0.0]            # (5 + 1)*(1 − 1)
        @test get_value(c.tols) ≈ [7.0 7.0]           # records spike time t
    end

    @testset "adaptive threshold raises after spikes" begin
        c = mk(; tau_theta=1e7, theta_plus=0.05)      # threshold dynamics ON
        @test get_value(c.thr_theta) ≈ [0.0 0.0]
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 1.0)
        # thr_theta = 0*exp(-dt/tau_theta) + s*theta_plus = 1*0.05
        @test get_value(c.thr_theta) ≈ [0.05 0.05]
    end

    @testset "midpoint (rk2) integrator selected" begin
        c = mk(; integration_type="rk2")
        @test c.intg_flag == 1
        set!(c.j, [100.0 100.0])
        advance_state!(c, 1.0, 1.0)
        # Midpoint diverges from Euler because the leak term (v_rest − v) is
        # nonzero at the half-step: stage 1 slope = (0 + 100)/10 = 10 ⇒ v_mid =
        # −65 + 10*0.5 = −60; stage 2 slope = (−5 + 100)/10 = 9.5 ⇒ v_next =
        # −65 + 9.5*1 = −55.5 (vs Euler's −55.0). Below threshold (−52) ⇒ no spike.
        @test get_value(c.v) ≈ [-55.5 -55.5]
    end

    @testset "reset_state! restores initial values" begin
        c = mk()
        set!(c.j, [1000.0 1000.0])
        advance_state!(c, 1.0, 3.0)
        @test all(get_value(c.s) .== 1.0)
        reset_state!(c)
        @test get_value(c.v) ≈ [-65.0 -65.0]
        @test all(get_value(c.s) .== 0.0)
        @test get_value(c.rfr) ≈ [5.0 5.0]
        @test all(get_value(c.tols) .== 0.0)
    end

    @testset "v_min clamps voltage floor" begin
        c = mk(; v_min=-62.0)
        set!(c.j, [-1000.0 -1000.0])                  # strong hyperpolarizing current
        advance_state!(c, 1.0, 1.0)
        @test all(get_value(c.v) .>= -62.0)
    end
end
