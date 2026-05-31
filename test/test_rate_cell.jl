using NGCLearn
using NGCSimLib: get_value, set!
using Test

# RateCell — EAGER-path tests. Pre-`setup!` Compartments behave as plain mutable
# value holders, so we exercise the dynamics standalone and compare against
# hand-computed values derived from the upstream docstring's ODE
# (RateCell.py:107-116).

@testset "RateCell" begin

    # ── Shape + defaults ─────────────────────────────────────────────────────
    @testset "construction + shape defaults" begin
        c = RateCell(; name="R", n_units=3, tau_m=10.0)
        @test c isa RateCell
        @test c isa NGCLearn.JaxComponent
        @test size(get_value(c.j)) == (1, 3)
        @test size(get_value(c.j_td)) == (1, 3)
        @test size(get_value(c.z)) == (1, 3)
        @test size(get_value(c.zF)) == (1, 3)
        @test all(get_value(c.j) .== 0.0)
        @test all(get_value(c.zF) .== 0.0)
        @test c.tau_m == 10.0
        @test c.prior_type == 0           # gaussian default
        @test c.prior_leak_rate == 0.0
        @test c.threshold_type == "none"
        @test c.intg_flag == 0           # euler default
        @test c.is_stateful                     # tau_m > 0
        # Identity is the default activation
        @test c.fx(2.0) == 2.0
        @test c.dfx([1.0, 2.0, 3.0]) ≈ [1.0, 1.0, 1.0]
    end

    # ── Stateful Euler step, zero leak ───────────────────────────────────────
    @testset "advance_state! Euler, gaussian prior, zero leak" begin
        # tau_m = 10, leak_gamma = 0 ⇒ dz/dt = (j + j_td) / tau_m
        # j = 1.0, j_td = 2.0 ⇒ dz/dt = 3/10 = 0.3
        # z_new = 0 + 0.3 * dt(=1) = 0.3
        c = RateCell(; name="R", n_units=2, tau_m=10.0,
            prior=("gaussian", 0.0), act_fx="identity")
        set!(c.j, [1.0 1.0])
        set!(c.j_td, [2.0 2.0])
        advance_state!(c, 1.0)
        @test get_value(c.z) ≈ [0.3 0.3]
        @test get_value(c.zF) ≈ [0.3 0.3]    # identity activation
    end

    # ── Stateful Euler step, non-zero leak (gaussian prior z_leak = z) ──────
    @testset "advance_state! gaussian prior leak" begin
        # gaussian z_leak = z (so the term is just `-leak_gamma * z`).
        # Pre-step z = 5.0, j = 1.0, j_td = 0, leak_gamma = 2.0, tau_m = 10.0.
        # dz/dt = (-2 * 5 + 1) / 10 = -0.9; z_new = 5 + 1 * (-0.9) = 4.1
        c = RateCell(; name="R", n_units=1, tau_m=10.0,
            prior=("gaussian", 2.0), act_fx="identity")
        set!(c.j, [1.0;;])    # 1×1 matrix
        set!(c.z, [5.0;;])
        advance_state!(c, 1.0)
        @test get_value(c.z) ≈ [4.1;;]
    end

    # ── Laplacian prior (z_leak = sign(z)) ──────────────────────────────────
    @testset "advance_state! laplacian prior" begin
        # z = -3, leak_gamma = 4, tau_m = 10, j = 0, j_td = 0
        # sign(z) = -1; dz/dt = (-(-1)*4 + 0)/10 = 0.4; z_new = -3 + 0.4 = -2.6
        c = RateCell(; name="R", n_units=1, tau_m=10.0,
            prior=("laplacian", 4.0), act_fx="identity")
        set!(c.z, [-3.0;;])
        advance_state!(c, 1.0)
        @test get_value(c.z) ≈ [-2.6;;]
    end

    # ── Stateless mode ──────────────────────────────────────────────────────
    @testset "stateless mode (tau_m <= 0)" begin
        c = RateCell(; name="R", n_units=2, tau_m=-1.0,
            act_fx="identity")
        @test c.is_stateful == false
        set!(c.j, [1.0 2.0])
        set!(c.j_td, [10.0 20.0])
        advance_state!(c, 1.0)
        # stateless: z = j + j_td  (no integration, no leak, no modulation)
        @test get_value(c.z) ≈ [11.0 22.0]
        @test get_value(c.zF) ≈ [11.0 22.0]
    end

    # ── Output scale ────────────────────────────────────────────────────────
    @testset "output_scale multiplies zF only, not z" begin
        c = RateCell(; name="R", n_units=2, tau_m=-1.0,
            act_fx="identity", output_scale=3.0)
        set!(c.j, [1.0 1.0])
        set!(c.j_td, [0.0 0.0])
        advance_state!(c, 1.0)
        @test get_value(c.z) ≈ [1.0 1.0]
        @test get_value(c.zF) ≈ [3.0 3.0]
    end

    # ── Tanh activation pair (dfx modulates j on stateful path) ─────────────
    @testset "tanh activation: dfx modulates j" begin
        # Pre-step z = 0 ⇒ dfx(0) = 1 - tanh(0)^2 = 1; so the modulation is a
        # no-op on the first step. After one Euler step:
        #   j_eff = j * dfx(z) * resist_scale = 1 * 1 * 1 = 1
        #   dz/dt = (j_eff + j_td)/tau_m = 1/10 = 0.1
        #   z_new = 0 + 0.1 = 0.1; zF = tanh(0.1) ≈ 0.09966799
        c = RateCell(; name="R", n_units=1, tau_m=10.0, act_fx="tanh")
        set!(c.j, [1.0;;])
        advance_state!(c, 1.0)
        @test get_value(c.z) ≈ [0.1;;]
        @test get_value(c.zF) ≈ [tanh(0.1);;]
    end

    # ── Soft thresholding ───────────────────────────────────────────────────
    @testset "soft-threshold post-integration step" begin
        # First step the dynamics, THEN apply soft-threshold with λ = 0.2.
        # j = 5.0, tau_m = 10, leak = 0 ⇒ z_raw = 0.5
        # threshold_soft(0.5, 0.2) = max(0.5 - 0.2, 0) - max(-0.5 - 0.2, 0)
        #                          = 0.3 - 0 = 0.3
        c = RateCell(; name="R", n_units=1, tau_m=10.0,
            threshold=("soft_threshold", 0.2), act_fx="identity")
        set!(c.j, [5.0;;])
        advance_state!(c, 1.0)
        @test get_value(c.z) ≈ [0.3;;]
    end

    # ── reset_state! zeros every compartment ────────────────────────────────
    @testset "reset_state! zeros every compartment" begin
        c = RateCell(; name="R", n_units=2, tau_m=10.0)
        set!(c.j, [7.0 7.0])
        set!(c.j_td, [3.0 3.0])
        set!(c.z, [9.0 9.0])
        set!(c.zF, [4.0 4.0])
        reset_state!(c)
        @test all(get_value(c.j) .== 0.0)
        @test all(get_value(c.j_td) .== 0.0)
        @test all(get_value(c.z) .== 0.0)
        @test all(get_value(c.zF) .== 0.0)
    end

    # ── Integrator selection (rk2 == euler at zero curvature) ───────────────
    @testset "rk2 path executes" begin
        c = RateCell(; name="R", n_units=1, tau_m=10.0,
            prior=("gaussian", 0.0), integration_type="rk2",
            act_fx="identity")
        @test c.intg_flag == 1
        set!(c.j, [1.0;;])
        advance_state!(c, 1.0)
        # With leak = 0 the rhs is constant in z (`dz/dt = j/tau_m`), so RK2's
        # midpoint correction matches Euler exactly: z_new = 0.1.
        @test get_value(c.z) ≈ [0.1;;]
    end

    # ── Unknown activation raises (preserves upstream behaviour) ────────────
    @testset "create_function rejects unknown activation" begin
        @test_throws ErrorException RateCell(; name="R", n_units=1, tau_m=1.0,
            act_fx="nope_not_real")
    end
end
