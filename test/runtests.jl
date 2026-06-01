using NGCLearn
using Test

@testset "NGCLearn.jl" begin
    @testset "version" begin
        @test NGCLearn.NGCLEARN_VERSION == v"0.1.0"
    end
    include("test_ode_utils.jl")
    include("test_lif_cell.jl")
    include("test_if_cell.jl")
    include("test_gaussian_error_cell.jl")
    include("test_rate_cell.jl")
    include("test_dense_synapse.jl")
    include("test_optim.jl")
    include("test_hebbian_synapse.jl")
    include("test_poisson_cell.jl")
    include("test_var_trace.jl")
    include("test_trace_stdp_synapse.jl")
    include("test_pcn_integration.jl")
    include("test_dc_snn_integration.jl")
    include("test_sparse_coding_integration.jl")
    include("test_jit_integration.jl")
end
