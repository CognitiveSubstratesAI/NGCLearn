using NGCLearn
using NGCSimLib: get_value, set!
using Test

@testset "GaussianErrorCell" begin
    @testset "construction + shapes" begin
        c = GaussianErrorCell(; name="E", n_units=2)
        @test c isa GaussianErrorCell
        @test c isa NGCLearn.JaxComponent
        @test size(get_value(c.mu)) == (1, 2)
        @test get_value(c.Sigma) ≈ [1.0;;]            # (1,1) block, sigma default 1
        @test get_value(c.L) == 0.0
        @test all(get_value(c.modulator) .== 1.0)
        @test all(get_value(c.mask) .== 1.0)
    end

    @testset "mismatch, precision-scaling, log-likelihood (sigma = 1)" begin
        c = GaussianErrorCell(; name="E", n_units=2)
        set!(c.mu, [0.0 0.0])
        set!(c.target, [1.0 2.0])
        advance_state!(c, 1.0)
        # e = target − mu = [1 2]; Sigma = 1
        @test get_value(c.dmu) ≈ [1.0 2.0]            # e / Sigma, gated by mod*mask=1
        @test get_value(c.dtarget) ≈ [-1.0 -2.0]      # −dmu
        @test get_value(c.dSigma) ≈ [1.0;;]           # constant 1 (no Sigma grad yet)
        @test get_value(c.L) ≈ -2.5                   # −sum(e^2)*0.5/Sigma = −(1+4)*0.5
        @test get_value(c.L) isa Real                 # squeezed to scalar
    end

    @testset "precision scaling with sigma != 1" begin
        c = GaussianErrorCell(; name="E", n_units=2, sigma=2.0)
        @test get_value(c.Sigma) ≈ [2.0;;]
        set!(c.mu, [0.0 0.0])
        set!(c.target, [4.0 0.0])
        advance_state!(c, 1.0)
        @test get_value(c.dmu) ≈ [2.0 0.0]            # [4 0] / 2
        @test get_value(c.L) ≈ -4.0                   # −(16)*0.5/2
    end

    @testset "modulator + one-shot mask gate the error" begin
        c = GaussianErrorCell(; name="E", n_units=2)
        set!(c.mu, [0.0 0.0])
        set!(c.target, [1.0 1.0])
        set!(c.modulator, [0.5 0.5])
        set!(c.mask, [1.0 0.0])
        advance_state!(c, 1.0)
        # dmu = (e/Sigma) * modulator * mask = [1 1]*0.5*[1 0] = [0.5 0]
        @test get_value(c.dmu) ≈ [0.5 0.0]
        # mask is "eaten" — reset to all-ones after the step.
        @test all(get_value(c.mask) .== 1.0)
    end

    @testset "reset_state! restores initial values" begin
        c = GaussianErrorCell(; name="E", n_units=2)
        set!(c.mu, [9.0 9.0])
        set!(c.target, [1.0 2.0])
        advance_state!(c, 1.0)
        @test get_value(c.L) != 0.0
        reset_state!(c)
        @test all(get_value(c.mu) .== 0.0)
        @test all(get_value(c.target) .== 0.0)
        @test all(get_value(c.modulator) .== 1.0)
        @test all(get_value(c.mask) .== 1.0)
        @test get_value(c.L) == 0.0
    end
end
