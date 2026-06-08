using NGCLearn
using NGCSimLib: get_value
using LinearAlgebra: norm
using Test

# Integration test for the BFA-SNN exhibit (bfa_snn.jl). Drives the full
# broadcast-feedback-alignment loop end-to-end on a tiny deterministic task
# (fixed seed ⇒ reproducible). Asserts structural validity, that adaptation is
# gated by `adapt`, and that the network LEARNS to classify two patterns via the
# fixed-feedback credit assignment (no backprop). Full MNIST parity vs upstream
# `train_bfasnn.py` is a separate, non-CI acceptance run.

@testset "BFA_SNN integration" begin
    @testset "structure + forward pass" begin
        m = BFA_SNN(; in_dim=4, out_dim=2, hid_dim=16, T=30, dt=0.25, key=1)
        @test m isa BFA_SNN
        lat, yMu, yCnt = process!(m, [1.0 0.0 1.0 0.0], [0.0 1.0]; adapt=false)
        @test size(lat) == (1, 16)
        @test size(yMu) == (1, 2) && size(yCnt) == (1, 2)
        @test sum(yMu) ≈ 1.0 && all(yMu .>= 0.0)        # valid label distribution
        @test all(yCnt .>= 0.0)
    end

    @testset "adapt=false leaves weights fixed; adapt=true changes them" begin
        m = BFA_SNN(; in_dim=6, out_dim=2, hid_dim=24, T=40, dt=0.25, key=3)
        W1_0 = copy(get_value(m.W1.weights))
        W2_0 = copy(get_value(m.W2.weights))
        process!(m, [1.0 1.0 1.0 0.0 0.0 0.0], [1.0 0.0]; adapt=false)
        @test get_value(m.W1.weights) == W1_0           # no learning without adapt
        @test get_value(m.W2.weights) == W2_0
        E2_0 = copy(get_value(m.E2.weights))
        # A few epochs (W2's `pre` is hidden spikes, so W2 only adapts once z1 fires —
        # which needs W1 to grow first; not learnable in a single call).
        for _ in 1:10
            process!(m, [1.0 1.0 1.0 0.0 0.0 0.0], [1.0 0.0]; adapt=true)
        end
        @test get_value(m.W1.weights) != W1_0           # Hebbian synapses evolved
        @test get_value(m.W2.weights) != W2_0
        @test get_value(m.E2.weights) == E2_0           # feedback matrix stays FIXED
    end

    @testset "learns to discriminate two classes (feedback alignment)" begin
        m = BFA_SNN(; in_dim=6, out_dim=2, hid_dim=48, T=50, dt=0.25, key=11)
        A = [1.0 1.0 1.0 0.0 0.0 0.0]
        labA = [1.0 0.0]                                 # class 1
        B = [0.0 0.0 0.0 1.0 1.0 1.0]
        labB = [0.0 1.0]                                 # class 2
        for _ in 1:60
            process!(m, A, labA; adapt=true)
            process!(m, B, labB; adapt=true)
        end
        _, yA, cA = process!(m, A, labA; adapt=false)
        _, yB, cB = process!(m, B, labB; adapt=false)
        # each pattern's output layer fires preferentially for its own class
        @test argmax(vec(yA)) == 1
        @test argmax(vec(yB)) == 2
        @test cA[1] > cA[2]                              # class-1 unit wins for A
        @test cB[2] > cB[1]                              # class-2 unit wins for B
    end
end
