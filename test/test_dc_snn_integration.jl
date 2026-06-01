using NGCLearn
using NGCSimLib: get_value, targeted
using Test

# Integration test for the Diehl & Cook (2015) DC-SNN exhibit. Deterministic
# (fixed PRNG key). Verifies the assembled network's structural invariants and
# the winner-take-all + STDP behavior — all numbers OBSERVED from real runs
# before being asserted (in=64, hid=10, T=200, key=3 ⇒ 11 exc spikes, unit-2
# dominant; in=4 small config ⇒ 0 spikes / under-driven).

@testset "DC_SNN integration (diehl_cook_snn)" begin
    @testset "construction + fixed lateral synapses" begin
        m = DC_SNN(; in_dim=4, hid_dim=6, T=50, key=10, name="dc_ctor")
        @test m isa DC_SNN
        @test (m.in_dim, m.hid_dim, m.T) == (4, 6, 50)
        # W1ie: hollow (zero diagonal), all off-diagonal = -120.
        W1ie = get_value(m.W1ie.weights)
        @test all(W1ie[i, i] == 0.0 for i in 1:6)
        @test W1ie[1, 2] == -120.0
        # W1ei: eye (only diagonal), = 22.5.
        W1ei = get_value(m.W1ei.weights)
        @test all(W1ei[i, i] == 22.5 for i in 1:6)
        @test W1ei[1, 2] == 0.0
        # W1 feedforward init uniform(0, 0.3).
        W1 = get_value(m.W1.weights)
        @test size(W1) == (4, 6)
        @test all(0.0 .<= W1 .<= 0.3)
    end

    @testset "wires are live (setup-before-wire guard)" begin
        m = DC_SNN(; in_dim=4, hid_dim=6, T=10, key=10, name="dc_wires")
        # W1.inputs ← z0.outputs ; W1.preSpike ← z0.outputs ; z1i.j ← W1ei.outputs
        @test targeted(m.W1.inputs)
        @test targeted(m.W1.preSpike)
        @test targeted(m.W1.postSpike)
        @test targeted(m.W1ie.inputs)
        @test targeted(m.z1i.j)
        # z1e.j is wired to a Summation op (not a single compartment) — targeted.
        @test targeted(m.z1e.j)
    end

    @testset "process! normalizes W1 columns to wNorm" begin
        m = DC_SNN(; in_dim=8, hid_dim=5, T=30, key=1, name="dc_norm")
        process!(m, fill(0.5, 1, 8); adapt=true)
        colsums = vec(sum(abs, get_value(m.W1.weights); dims=1))
        @test all(≈(78.4; atol=1e-6), colsums)
    end

    @testset "STDP changes W1 under drive" begin
        m = DC_SNN(; in_dim=16, hid_dim=8, T=100, key=2, name="dc_stdp")
        W_before = copy(get_value(m.W1.weights))
        process!(m, fill(0.8, 1, 16); adapt=true)
        @test get_value(m.W1.weights) != W_before
    end

    @testset "excitatory layer spikes + winner-take-all under strong drive" begin
        m = DC_SNN(; in_dim=64, hid_dim=10, T=200, key=3, name="dc_wta")
        counts = process!(m, fill(0.9, 1, 64); adapt=true)
        @test size(counts) == (1, 10)
        total = sum(counts)
        @test total > 0.0                 # the network actually spikes
        @test total <= 200.0              # one_spike=true ⇒ ≤1 exc spike/step
        # Lateral inhibition ⇒ competition: one unit should dominate.
        @test maximum(counts) >= 0.5 * total
    end

    @testset "adapt=false leaves W1 unchanged (inference only)" begin
        m = DC_SNN(; in_dim=16, hid_dim=8, T=80, key=4, name="dc_infer")
        W_before = copy(get_value(m.W1.weights))
        process!(m, fill(0.8, 1, 16); adapt=false)
        @test get_value(m.W1.weights) == W_before   # no evolve!, no norm!
    end

    @testset "determinism under fixed key" begin
        a = DC_SNN(; in_dim=32, hid_dim=8, T=120, key=5, name="dc_detA")
        b = DC_SNN(; in_dim=32, hid_dim=8, T=120, key=5, name="dc_detB")
        ca = process!(a, fill(0.7, 1, 32); adapt=true)
        cb = process!(b, fill(0.7, 1, 32); adapt=true)
        @test ca == cb
        @test get_value(a.W1.weights) ≈ get_value(b.W1.weights)
    end
end
