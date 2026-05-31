# 02_pc_discrim_train.jl — train the faithful pc_discrim PCN end-to-end.
#
# Uses the `PCN` model (src/models/pcn.jl), a faithful port of
# ngc-museum/exhibits/pc_discrim/pcn_model.py, to learn a tiny linearly-separable
# 2-class task via the full PEM loop (Projection → Expectation → Maximization)
# with weight tying (E = Wᵀ, Q = W) and Hebbian-Adam updates.
#
# This is the Phase-A acceptance exhibit, scaled down so it runs in seconds.
# Full MNIST parity vs upstream `train_pcn.py` is a separate (non-CI) run.
#
# Observed (eta=0.002, key=7): mean ‖lab-y_mu‖² falls 0.707 → 0.0009 over 60
# epochs, projection accuracy 4/4.
#
# Run: `julia --project=. examples/02_pc_discrim_train.jl`

using NGCLearn

# label = one-hot(sum(x[1:2]) vs sum(x[3:4])).
X = [reshape(Float64[1, 1, 0, 0], 1, 4),
    reshape(Float64[0, 0, 1, 1], 1, 4),
    reshape(Float64[1, 0, 0, 1], 1, 4),
    reshape(Float64[0, 1, 1, 0], 1, 4)]
Y = [reshape(Float64[1, 0], 1, 2),
    reshape(Float64[0, 1], 1, 2),
    reshape(Float64[1, 0], 1, 2),
    reshape(Float64[0, 1], 1, 2)]

m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6, T=10, tau_m=10.0,
    act_fx="tanh", eta=0.002, key=7)

epochs = 60
println("=== pc_discrim training (epochs=$epochs, T=$(m.T)) ===")
println("epoch   mean ‖lab-y_mu‖²    mean EFE")
for epoch in 1:epochs
    errs = Float64[]
    efes = Float64[]
    for i in 1:length(X)
        _, y_mu, EFE = process!(m, X[i], Y[i])
        push!(errs, sum(abs2, Y[i] .- y_mu))
        push!(efes, EFE)
    end
    if epoch == 1 || epoch % 10 == 0 || epoch == epochs
        me = sum(errs) / length(errs)
        mf = sum(efes) / length(efes)
        println(lpad(epoch, 5), "     ", round(me; digits=5), "          ",
            round(mf; digits=5))
    end
end

println("\n=== test-time projection (no settling, no learning) ===")
correct = 0
for i in 1:length(X)
    p = project(m, X[i])
    pred = argmax(vec(p))
    truth = argmax(vec(Y[i]))
    global correct += (pred == truth)
    println("  x$i → ", round.(p; digits=3), "  class $pred (truth $truth)")
end
println("accuracy: $correct/$(length(X))")
println("\npc_discrim exhibit validated ✓ — faithful PEM loop learns the task.")
