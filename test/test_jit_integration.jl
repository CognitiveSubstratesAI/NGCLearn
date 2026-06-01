using NGCLearn
using NGCSimLib: Context, post_init!, MethodProcess, compile_process!,
    compile_with_reactant!, run, get_processes, get_context, get_components,
    get_value, set!, get_state
using Test

# End-to-end JIT test: a real NGCLearn component (LIFCell — scalar hyperparams,
# broadcasts, step_euler integration, conditionals) must
#   (1) compile + run through compile_process! (eager-spliced ctx runner), and
#   (2) compile + run through compile_with_reactant! (Reactant.@compile / XLA),
# with both producing output IDENTICAL to plain eager dispatch.
#
# This is the conformance property of decisions.md §4: the JIT path matches the
# eager ground truth. Reactant precompile + trace is heavy, so this whole file
# can be skipped by setting NGCLEARN_SKIP_JIT=1 (CI runs it by default).

if get(ENV, "NGCLEARN_SKIP_JIT", "0") == "1"
    @info "Skipping JIT integration test (NGCLEARN_SKIP_JIT=1)"
else
    import Reactant

    @testset "JIT integration (LIFCell through Reactant)" begin
        # --- eager ground truth: one LIF step under strong drive ---
        ce = LIFCell(; name="L", n_units=3, tau_m=10.0, v_min=-62.0)
        post_init!(ce)
        set!(ce.j, [1000.0 1000.0 1000.0])
        advance_state!(ce, 1.0, 1.0)
        eager_v = copy(get_value(ce.v))
        eager_s = copy(get_value(ce.s))
        @test eager_s == [1.0 1.0 1.0]      # supra-threshold ⇒ spikes
        @test eager_v == [-60.0 -60.0 -60.0]  # reset to v_reset

        # --- build the same cell inside a Context as a MethodProcess ---
        Context("jit") do _ctx
            c = LIFCell(; name="L", n_units=3, tau_m=10.0, v_min=-62.0)
            post_init!(c)
            set!(c.j, [1000.0 1000.0 1000.0])
            p = MethodProcess(; name="step")
            p >> (c, :advance_state!)
            post_init!(p)
        end
        ctx = get_context("jit")
        p = get_processes(ctx)["step"]
        cell = get_components(ctx)["L"]

        # (1) eager-spliced compiled runner
        compile_process!(p)
        @test p.keyword_order == [:dt, :t]
        out_c, _ = run(p; dt=1.0, t=1.0)
        @test out_c["jit:L:v"] == eager_v
        @test out_c["jit:L:s"] == eager_s

        # (2) Reactant-traced runner — reset state, re-clamp, compile, run.
        reset_state!(cell)
        set!(cell.j, [1000.0 1000.0 1000.0])
        sample_ctx = copy(get_state())
        sample_args = Any[1.0 for _ in p.keyword_order]
        compile_with_reactant!(p, sample_ctx, sample_args)
        out_j, _ = run(p; dt=1.0, t=1.0)
        @test Array(out_j["jit:L:v"]) == Array(eager_v)
        @test Array(out_j["jit:L:s"]) == Array(eager_s)
    end

    # Whole-zoo coverage: every component's @compilable advance_state! must
    # trace through Reactant.@compile AND match the eager-spliced result. Each
    # case = (label, build, drive!, kwargs Dict, compartment-key to compare).
    # Drive values are arbitrary but fixed; we assert JIT == eager, not specific
    # numbers (the per-component value semantics are covered by their unit tests).
    _zoo = [
        ("RateCell", () -> RateCell(; name="c", n_units=3, tau_m=10.0, act_fx="tanh"),
            c -> set!(c.j, [1.0 2.0 3.0]), Dict(:dt => 1.0), "z"),
        ("GaussianErrorCell", () -> GaussianErrorCell(; name="c", n_units=3),
            c -> (set!(c.mu, [0.0 0.0 0.0]); set!(c.target, [1.0 2.0 3.0])),
            Dict(:dt => 1.0), "dmu"),
        ("DenseSynapse",
            () -> DenseSynapse(; name="c", shape=(3, 2),
                weight_init=("constant", 0.5), key=1),
            c -> set!(c.inputs, [1.0 2.0 3.0]), Dict{Symbol, Float64}(), "outputs"),
        ("PoissonCell",
            () -> PoissonCell(; name="c", n_units=4, target_freq=50.0, key=1),
            c -> set!(c.inputs, [0.5 0.5 0.5 0.5]), Dict(:t => 1.0, :dt => 1.0),
            "outputs"),
        ("VarTrace",
            () -> VarTrace(; name="c", n_units=3, tau_tr=20.0, a_delta=1.0),
            c -> set!(c.inputs, [1.0 0.0 1.0]), Dict(:dt => 1.0), "trace"),
        ("TraceSTDPSynapse",
            () -> TraceSTDPSynapse(; name="c", shape=(3, 2), A_plus=1e-2,
                A_minus=1e-4, weight_init=("constant", 0.1), key=1),
            c -> set!(c.inputs, [1.0 2.0 3.0]), Dict{Symbol, Float64}(), "outputs")
    ]

    @testset "JIT coverage: $label traces + matches eager" for (
            label, build, drive!, kw, readkey
        ) in _zoo

        ctxname = "jitcov_" * label
        Context(ctxname) do _ctx
            c = build()
            post_init!(c)
            drive!(c)
            p = MethodProcess(; name="step")
            p >> (c, :advance_state!)
            post_init!(p)
        end
        ctx = get_context(ctxname)
        p = get_processes(ctx)["step"]
        cell = get_components(ctx)["c"]

        compile_process!(p)
        out_e, _ = run(p; kw...)
        eager = copy(Array(out_e["$ctxname:c:$readkey"]))

        reset_state!(cell)
        drive!(cell)
        sample_args = Any[kw[k] for k in p.keyword_order]
        compile_with_reactant!(p, copy(get_state()), sample_args)
        out_j, _ = run(p; kw...)
        @test Array(out_j["$ctxname:c:$readkey"]) == eager
    end
end
