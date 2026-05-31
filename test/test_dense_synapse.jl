using NGCLearn
using NGCSimLib: get_value, set!
using Test

# DenseSynapse — EAGER-path tests. We verify the linear forward pass against
# hand-computed values + check the init wiring (weight shape, bias defaults,
# Bernoulli sparsity).

@testset "DenseSynapse" begin

    # ── Construction shapes + default initializers ───────────────────────────
    @testset "construction defaults + shapes" begin
        s = DenseSynapse(; name="W", shape=(3, 2), key=42)
        @test s isa DenseSynapse
        @test s isa NGCLearn.JaxComponent
        @test s.shape == (3, 2)
        @test s.batch_size == 1
        @test s.resist_scale == 1.0
        @test size(get_value(s.weights)) == (3, 2)
        @test size(get_value(s.biases)) == (1, 2)
        @test size(get_value(s.inputs)) == (1, 3)
        @test size(get_value(s.outputs)) == (1, 2)
        # default mask is a (1,1) broadcast-passthrough
        @test get_value(s.mask) == ones(1, 1)
        # default weight init = uniform(0.025, 0.8)
        W = get_value(s.weights)
        @test all((W .>= 0.025) .& (W .<= 0.8))
        # bias_init=nothing ⇒ zeros
        @test all(get_value(s.biases) .== 0.0)
    end

    # ── Forward pass: outputs = (inputs @ weights) * resist_scale + biases ──
    @testset "advance_state! linear forward pass" begin
        # Build with constant weights so the math is easy.
        s = DenseSynapse(; name="W", shape=(3, 2),
            weight_init=("constant", 0.5), key=1)
        @test all(get_value(s.weights) .== 0.5)
        set!(s.inputs, [1.0 2.0 3.0])     # 1×3
        advance_state!(s)
        # inputs @ W = [1+2+3] = 6 → out = [3.0, 3.0]
        @test get_value(s.outputs) ≈ [3.0 3.0]
    end

    # ── resist_scale multiplies the output ──────────────────────────────────
    @testset "resist_scale scales output" begin
        s = DenseSynapse(; name="W", shape=(2, 1),
            weight_init=("constant", 1.0),
            resist_scale=10.0, key=2)
        set!(s.inputs, [3.0 4.0])
        advance_state!(s)
        # (3+4) * 10 = 70
        @test get_value(s.outputs) ≈ [70.0;;]
    end

    # ── biases get added after the scaled matmul ────────────────────────────
    @testset "bias_init adds bias after scaling" begin
        s = DenseSynapse(; name="W", shape=(2, 3),
            weight_init=("constant", 0.0),
            bias_init=("constant", 5.0), key=3)
        # All weights zero ⇒ out = bias before any scaling
        set!(s.inputs, [1.0 1.0])
        advance_state!(s)
        @test get_value(s.outputs) ≈ [5.0 5.0 5.0]
    end

    # ── Multiplicative mask zero ⇒ output zero ──────────────────────────────
    @testset "mask zero suppresses the synapse" begin
        s = DenseSynapse(; name="W", shape=(2, 2),
            weight_init=("constant", 1.0),
            bias_init=("constant", 0.0), key=4)
        set!(s.mask, zeros(2, 2))
        set!(s.inputs, [3.0 4.0])
        advance_state!(s)
        @test all(get_value(s.outputs) .== 0.0)
    end

    # ── Bernoulli sparsification (p_conn < 1) zeros some entries ────────────
    @testset "p_conn < 1 produces a sparse weight matrix" begin
        # Reproducible RNG (seed=7); with p_conn=0.3 over a 20×20 grid we
        # expect ~120 nonzeros — well below the 400 cell count.
        s = DenseSynapse(; name="W", shape=(20, 20),
            weight_init=("constant", 1.0),
            p_conn=0.3, key=7)
        W = get_value(s.weights)
        n_nonzero = sum(W .!= 0.0)
        @test 0 < n_nonzero < 400          # genuine sparsification
        @test n_nonzero <= 400 * 0.5       # under half the cells
    end

    # ── Reset semantics: zero outputs unconditionally; inputs are zeroed only
    #    when NOT externally wired. Mirrors upstream denseSynapse.py:104-109. ─
    # Note: a pre-`setup!` Compartment has `target === nothing` which
    # `NGCSimLib.targeted` reports as TRUE ("not a string ⇒ wired") — so in
    # this standalone EAGER test the inputs are NOT zeroed (matches the
    # protect-foreign-wires intent of the guard). The outputs reset is
    # unconditional and that's what we assert here.
    @testset "reset_state! zeros outputs (unconditional)" begin
        s = DenseSynapse(; name="W", shape=(2, 2), key=9)
        set!(s.outputs, [9.0 9.0])
        reset_state!(s)
        @test all(get_value(s.outputs) .== 0.0)
    end

    # ── Reproducible init from explicit seed ────────────────────────────────
    @testset "explicit key gives reproducible weights" begin
        a = DenseSynapse(; name="W", shape=(4, 4), key=12345)
        b = DenseSynapse(; name="W", shape=(4, 4), key=12345)
        @test get_value(a.weights) == get_value(b.weights)
    end

    # ── Unknown distribution raises ─────────────────────────────────────────
    @testset "unsupported init distribution raises" begin
        @test_throws ErrorException DenseSynapse(;
            name="W", shape=(2, 2),
            weight_init=("not_a_dist", 0.0, 1.0), key=11
        )
    end
end
