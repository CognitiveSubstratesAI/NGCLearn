using NGCLearn
using Test

@testset "NGCLearn.jl" begin
    @testset "version" begin
        @test NGCLearn.NGCLEARN_VERSION == v"0.1.0"
    end
    include("test_ode_utils.jl")
    include("test_lif_cell.jl")
    include("test_gaussian_error_cell.jl")
    include("test_rate_cell.jl")
    include("test_dense_synapse.jl")
    include("test_optim.jl")
    include("test_hebbian_synapse.jl")
    include("test_poisson_cell.jl")
    include("test_var_trace.jl")
    include("test_pcn_integration.jl")
end
