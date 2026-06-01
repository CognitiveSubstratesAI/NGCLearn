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
end
