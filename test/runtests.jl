using NGCLearn
using Test

@testset "NGCLearn.jl" begin
    @testset "version" begin
        @test NGCLearn.NGCLEARN_VERSION == v"0.1.0"
    end
    include("test_ode_utils.jl")
    include("test_lif_cell.jl")
    include("test_gaussian_error_cell.jl")
end
