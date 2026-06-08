using NGCLearn
using NGCSimLib: get_value, set!
using Test

# EAGER-path tests (see test_lif_cell.jl header). Expected numbers are hand-
# computed from the upstream sLIFCell dynamics. Constructed with `thr_jitter=0`
# and `resist_inh=0` for deterministic arithmetic unless a test needs otherwise.

@testset "SLIFCell" begin
    # Default: adaptive-threshold OFF (gain=leak=0), no jitter, no inhibition,
    # refract_time=0 (so the refractory mask is always 1) — clean arithmetic.
    mk(; kw...) = SLIFCell(;
        name="S", n_units=2, tau_m=1.0, resist_m=1.0, thr=0.5, thr_jitter=0.0, kw...
    )

    @testset "surrogate functions" begin
        # Heaviside spike emission (strict >).
        @test secant_spike_fx([0.5 1.5], [1.0 1.0]) ≈ [0.0 1.0]
        @test secant_spike_fx([1.0 1.0], [1.0 1.0]) ≈ [0.0 0.0]   # not strictly greater
        # secant derivative: sech(c2·j) for j>0, 0 for j≤0 (CODE = sech, not sech²).
        d = secant_d_spike_fx([1.0 2.0 -1.0 0.0])
        @test d[1] ≈ 0.99680851 atol = 1e-6     # sech(0.08)
        @test d[2] ≈ 0.98733513 atol = 1e-6     # sech(0.16)
        @test d[3] == 0.0                        # j < 0 → masked
        @test d[4] == 0.0                        # j = 0 → masked (strict >)
        # omit_scale=false multiplies by c1·c2.
        @test secant_d_spike_fx([1.0]; omit_scale=false)[1] ≈ 0.99680851 * 0.82 * 0.08 atol =
            1e-8
        # the estimator factory returns the pair.
        sfx, dfx = secant_lif_estimator()
        @test sfx === secant_spike_fx && dfx === secant_d_spike_fx
    end

    @testset "construction + shapes" begin
        c = mk()
        @test c isa SLIFCell
        @test c isa NGCLearn.JaxComponent
        @test size(get_value(c.v)) == (1, 2)
        @test get_value(c.v) ≈ [0.0 0.0]               # rest voltage is 0 (not v_rest)
        @test get_value(c.thr) ≈ [0.5 0.5]             # threshold0 = thr (jitter 0)
        @test get_value(c.rfr) ≈ [0.0 0.0]             # rest + refract_T (0)
        @test get_value(c.surrogate) ≈ [1.0 1.0]       # surrogate init = 1
        @test all(get_value(c.s) .== 0.0)
        @test size(c.inh_weights) == (2, 2)
        @test c.inh_weights[1, 1] == 0.0 && c.inh_weights[2, 2] == 0.0   # hollow
    end

    @testset "sub-threshold step (no spike)" begin
        c = mk(; thr=10.0)
        set!(c.j, [1.0 2.0])
        advance_state!(c, 1.0, 1.0)
        # j·R_m = [1 2]; dv = (-0 + j)·(1/1)·mask(=1) = [1 2]; v = 0 + 1·[1 2] = [1 2]
        @test get_value(c.v) ≈ [1.0 2.0]
        @test all(get_value(c.s) .== 0.0)              # 1,2 < 10 ⇒ no spike
        @test get_value(c.rfr) ≈ [1.0 1.0]             # (0 + 1)·(1 − 0)
        @test all(get_value(c.tols) .== 0.0)
        @test get_value(c.surrogate) ≈ secant_d_spike_fx([1.0 2.0])   # wired from drive
        @test get_value(c.j) ≈ [1.0 2.0]               # processed drive written back
    end

    @testset "supra-threshold spike: hyperpolarize + adaptive thr + tols" begin
        c = mk(; thr=0.5, thr_gain=0.1, thr_leak=0.01)
        set!(c.j, [1.0 0.0])
        advance_state!(c, 1.0, 5.0)
        # _v = [1 0]; spikes = ([1 0] > 0.5) = [1 0]; hyperpolarize ⇒ v = [0 0]
        @test get_value(c.v) ≈ [0.0 0.0]
        @test get_value(c.s) ≈ [1.0 0.0]
        # thr = thr + s·gain − thr·leak = [0.5 0.5] + [0.1 0] − [0.005 0.005]
        @test get_value(c.thr) ≈ [0.595 0.495]
        @test get_value(c.rfr) ≈ [1.0 1.0]             # spiked: 0+1·dt; other: (0+1)·1
        @test get_value(c.tols) ≈ [5.0 0.0]            # only the spiking unit records t
    end

    @testset "sticky spikes pin through the refractory window" begin
        c = SLIFCell(;
            name="S", n_units=2, tau_m=1.0, resist_m=1.0, thr=0.5,
            refract_time=2.0, sticky_spikes=true, thr_jitter=0.0
        )
        set!(c.j, [1.0 1.0])
        advance_state!(c, 1.0, 1.0)                    # step 1: natural spike
        @test get_value(c.s) ≈ [1.0 1.0]
        @test get_value(c.rfr) ≈ [1.0 1.0]             # (2+1)·0 + 1·1 ; now < refract_T
        set!(c.j, [0.0 0.0])
        advance_state!(c, 1.0, 2.0)                    # step 2: no drive, but refractory
        # old rfr = 1.0 < refract_T(2) ⇒ mask 0 ⇒ sticky pins s = 0·0 + (1−0) = 1
        @test get_value(c.s) ≈ [1.0 1.0]
        @test get_value(c.rfr) ≈ [2.0 2.0]             # (1+1)·(1−0) + 0
    end

    @testset "rho_b sparsity-threshold mode" begin
        # dthr = sum(spikes) − 1; v_thr ← max(v_thr + dthr·rho_b, 0.025).
        @test NGCLearn._update_threshold_slif(
            1.0, [0.5 0.5 0.5], [1.0 1.0 0.0], 0.0, 0.0, 0.1
        ) ≈ [0.6 0.6 0.6]                              # 2 spikes ⇒ dthr=+1
        @test NGCLearn._update_threshold_slif(
            1.0, [0.5 0.5 0.5], [0.0 0.0 0.0], 0.0, 0.0, 0.1
        ) ≈ [0.4 0.4 0.4]                              # 0 spikes ⇒ dthr=−1
        @test NGCLearn._update_threshold_slif(
            1.0, [0.05 0.05], [0.0 0.0], 0.0, 0.0, 0.1
        ) ≈ [0.025 0.025]                              # floored at 0.025
    end

    @testset "lateral inhibition subtracts recurrent drive" begin
        c = SLIFCell(;
            name="S", n_units=2, tau_m=1.0, resist_m=1.0, thr=10.0, resist_inh=0.5,
            thr_jitter=0.0
        )
        c.inh_weights .= [0.0 1.0; 1.0 0.0]            # known hollow matrix
        set!(c.s, [1.0 0.0])                           # previous-step spikes
        set!(c.j, [2.0 2.0])
        advance_state!(c, 1.0, 1.0)
        # j = [2 2]·R_m − (s·Wi)·inh_R = [2 2] − ([0 1])·0.5 = [2 1.5]
        @test get_value(c.j) ≈ [2.0 1.5]
    end

    @testset "reset restores state; threshold persistence honored" begin
        c = mk(; thr_gain=0.1)
        set!(c.j, [1.0 1.0])
        advance_state!(c, 1.0, 3.0)                    # mutate v/s/thr/rfr/tols
        reset_state!(c)
        @test get_value(c.v) ≈ [0.0 0.0]
        @test all(get_value(c.s) .== 0.0)
        @test get_value(c.rfr) ≈ [0.0 0.0]
        @test get_value(c.surrogate) ≈ [1.0 1.0]
        @test all(get_value(c.tols) .== 0.0)
        @test get_value(c.thr) ≈ [0.5 0.5]             # non-persistent ⇒ back to threshold0

        cp = mk(; thr_gain=0.1, thr_persist=true)
        set!(cp.j, [1.0 0.0])
        advance_state!(cp, 1.0, 3.0)
        thr_after = copy(get_value(cp.thr))
        reset_state!(cp)
        @test get_value(cp.thr) ≈ thr_after            # persistent ⇒ threshold survives reset
    end
end
