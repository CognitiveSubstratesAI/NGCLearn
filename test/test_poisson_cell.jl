using NGCLearn
using NGCSimLib: get_value, set!, targeted
using Test

# PoissonCell is stochastic; tests pin the deterministic envelope of its
# spike-probability law (pspike = inputs * dt/1000 * target_freq) plus the
# reproducibility guaranteed by a fixed PRNG seed.

@testset "PoissonCell" begin
    @testset "construction + shapes" begin
        c = PoissonCell(; name="P", n_units=3, target_freq=63.75)
        @test c isa PoissonCell
        @test c isa NGCLearn.JaxComponent
        @test c.target_freq == 63.75
        @test size(get_value(c.inputs)) == (1, 3)
        @test all(get_value(c.outputs) .== 0.0)
        @test all(get_value(c.tols) .== 0.0)
    end

    @testset "zero input ⇒ no spikes (pspike = 0)" begin
        c = PoissonCell(; name="P", n_units=4, target_freq=100.0, key=1)
        set!(c.inputs, zeros(1, 4))
        advance_state!(c, 1.0, 1.0)
        @test all(get_value(c.outputs) .== 0.0)
    end

    @testset "saturating input ⇒ all spike (pspike ≥ 1 > any uniform)" begin
        # pspike = inputs * dt/1000 * target_freq. With inputs huge, pspike ≫ 1,
        # so eps (∈[0,1)) is always < pspike ⇒ every unit spikes.
        c = PoissonCell(; name="P", n_units=5, target_freq=1000.0, key=2)
        set!(c.inputs, fill(1e6, 1, 5))
        advance_state!(c, 3.0, 3.0)
        @test all(get_value(c.outputs) .== 1.0)
        @test all(get_value(c.tols) .== 3.0)   # tols records spike time t
    end

    @testset "tols holds previous time when no spike" begin
        c = PoissonCell(; name="P", n_units=2, target_freq=1000.0, key=3)
        set!(c.inputs, fill(1e6, 1, 2))
        advance_state!(c, 5.0, 1.0)            # all spike → tols = 5
        @test all(get_value(c.tols) .== 5.0)
        set!(c.inputs, zeros(1, 2))            # now none spike
        advance_state!(c, 9.0, 1.0)
        @test all(get_value(c.tols) .== 5.0)   # unchanged (1-0)*5 + 0*9
    end

    @testset "fixed key ⇒ reproducible spike pattern" begin
        mk() = (c=PoissonCell(; name="P", n_units=8, target_freq=50.0, key=42);
            set!(c.inputs, fill(0.5, 1, 8)); c)
        a = mk()
        b = mk()
        advance_state!(a, 1.0, 1.0)
        advance_state!(b, 1.0, 1.0)
        @test get_value(a.outputs) == get_value(b.outputs)
        @test all(o -> o == 0.0 || o == 1.0, get_value(a.outputs))  # binary
    end

    @testset "key advances between steps (stream not frozen)" begin
        c = PoissonCell(; name="P", n_units=64, target_freq=50.0, key=7)
        set!(c.inputs, fill(0.5, 1, 64))
        advance_state!(c, 1.0, 1.0)
        s1 = copy(get_value(c.outputs))
        set!(c.inputs, fill(0.5, 1, 64))
        advance_state!(c, 2.0, 1.0)
        s2 = copy(get_value(c.outputs))
        @test s1 != s2   # different draws (key advanced) — overwhelmingly likely at n=64
    end

    @testset "reset_state! zeros outputs + tols" begin
        c = PoissonCell(; name="P", n_units=3, target_freq=1000.0, key=5)
        set!(c.inputs, fill(1e6, 1, 3))
        advance_state!(c, 2.0, 1.0)
        @test any(get_value(c.outputs) .== 1.0)
        reset_state!(c)
        @test all(get_value(c.outputs) .== 0.0)
        @test all(get_value(c.tols) .== 0.0)
    end
end
