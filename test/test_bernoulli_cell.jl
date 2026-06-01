using NGCLearn
using NGCSimLib: get_value, set!, targeted
using Test

# BernoulliCell is stochastic; tests pin the deterministic envelope of
# Bernoulli(p=inputs): p=0 ⇒ never spike, p=1 ⇒ always spike, binary outputs,
# tols recording, and fixed-key reproducibility.

@testset "BernoulliCell" begin
    @testset "construction + shapes" begin
        c = BernoulliCell(; name="B", n_units=4)
        @test c isa BernoulliCell
        @test c isa NGCLearn.JaxComponent
        @test size(get_value(c.inputs)) == (1, 4)
        @test all(get_value(c.outputs) .== 0.0)
    end

    @testset "p=0 ⇒ no spikes, p=1 ⇒ all spikes" begin
        c = BernoulliCell(; name="B", n_units=4, key=1)
        set!(c.inputs, [0.0 0.0 0.0 0.0])
        advance_state!(c, 1.0)
        @test all(get_value(c.outputs) .== 0.0)
        set!(c.inputs, [1.0 1.0 1.0 1.0])
        advance_state!(c, 3.0)
        @test all(get_value(c.outputs) .== 1.0)
        @test all(get_value(c.tols) .== 3.0)        # tols records spike time
    end

    @testset "mixed probabilities ⇒ binary output, deterministic gates honored" begin
        c = BernoulliCell(; name="B", n_units=3, key=2)
        set!(c.inputs, [1.0 0.0 1.0])               # forced on/off/on
        advance_state!(c, 1.0)
        @test get_value(c.outputs) == [1.0 0.0 1.0]
    end

    @testset "tols holds previous time when no spike" begin
        c = BernoulliCell(; name="B", n_units=2, key=3)
        set!(c.inputs, [1.0 1.0])
        advance_state!(c, 5.0)
        @test all(get_value(c.tols) .== 5.0)
        set!(c.inputs, [0.0 0.0])
        advance_state!(c, 9.0)
        @test all(get_value(c.tols) .== 5.0)        # unchanged
    end

    @testset "fixed key ⇒ reproducible + binary" begin
        mk() = (c=BernoulliCell(; name="B", n_units=16, key=42);
            set!(c.inputs, fill(0.5, 1, 16)); c)
        a = mk()
        b = mk()
        advance_state!(a, 1.0)
        advance_state!(b, 1.0)
        @test get_value(a.outputs) == get_value(b.outputs)
        @test all(o -> o == 0.0 || o == 1.0, get_value(a.outputs))
    end

    @testset "key advances between steps" begin
        c = BernoulliCell(; name="B", n_units=64, key=7)
        set!(c.inputs, fill(0.5, 1, 64))
        advance_state!(c, 1.0)
        s1 = copy(get_value(c.outputs))
        set!(c.inputs, fill(0.5, 1, 64))
        advance_state!(c, 2.0)
        @test s1 != get_value(c.outputs)            # different draw (key advanced)
    end

    @testset "reset_state! zeros outputs + tols" begin
        c = BernoulliCell(; name="B", n_units=3, key=5)
        set!(c.inputs, [1.0 1.0 1.0])
        advance_state!(c, 2.0)
        reset_state!(c)
        @test all(get_value(c.outputs) .== 0.0)
        @test all(get_value(c.tols) .== 0.0)
    end
end
