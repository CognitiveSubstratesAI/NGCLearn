# jit_eager.jl — eager vs Reactant-JIT timing for a LIFCell step, across sizes.
#
# Times the two compiled-process paths head-to-head over a range of layer
# widths and prints a Markdown table + environment info:
#   - compile_process!        → eager-spliced ctx runner (plain Julia)
#   - compile_with_reactant!  → Reactant.@compile / XLA-traced runner
#
# Both paths produce identical output (asserted in test/test_jit_integration.jl);
# this measures the performance difference. Numbers are machine- and
# Reactant-version-dependent — see results.md for a captured reference run.
#
# Run:  julia --project=benchmark benchmark/jit_eager.jl

using NGCLearn
using NGCSimLib: Context, post_init!, MethodProcess, compile_process!,
    compile_with_reactant!, run, get_processes, get_context, get_components,
    set!, get_state
import Reactant
import InteractiveUtils

const SIZES = (64, 256, 1024, 4096)
const REPS = 1000

function bench_N(N::Int; reps::Int=REPS)
    nm = "bench$N"
    Context(nm) do _ctx
        c = LIFCell(; name="c", n_units=N, tau_m=10.0)
        post_init!(c)
        set!(c.j, fill(50.0, 1, N))
        p = MethodProcess(; name="step")
        p >> (c, :advance_state!)
        post_init!(p)
    end
    ctx = get_context(nm)
    p = get_processes(ctx)["step"]
    cell = get_components(ctx)["c"]

    compile_process!(p)
    run(p; dt=1.0, t=1.0)                       # warmup
    t0 = time_ns()
    for _ in 1:reps
        run(p; dt=1.0, t=1.0)
    end
    eager_us = (time_ns() - t0) / 1e3 / reps

    reset_state!(cell)
    set!(cell.j, fill(50.0, 1, N))
    compile_with_reactant!(p, copy(get_state()), Any[1.0 for _ in p.keyword_order])
    run(p; dt=1.0, t=1.0)                       # warmup the thunk
    t0 = time_ns()
    for _ in 1:reps
        run(p; dt=1.0, t=1.0)
    end
    jit_us = (time_ns() - t0) / 1e3 / reps

    return (N=N, eager=eager_us, jit=jit_us, speedup=eager_us / jit_us)
end

println("LIFCell single-step: eager-spliced vs Reactant-JIT\n")
println("| n_units | eager µs/step | JIT µs/step | speedup |")
println("|--------:|--------------:|------------:|--------:|")
for N in SIZES
    r = bench_N(N)
    println("| $(r.N) | $(round(r.eager; digits=1)) | $(round(r.jit; digits=1)) | ",
        round(r.speedup; digits=2), "× |")
end

println("\nEnvironment:")
io = IOBuffer()
InteractiveUtils.versioninfo(io)
for l in split(String(take!(io)), "\n")
    if occursin(r"Julia Version|OS:|CPU:|WORD_SIZE", l)
        println("  ", strip(l))
    end
end
println("  Reactant ", pkgversion(Reactant))
