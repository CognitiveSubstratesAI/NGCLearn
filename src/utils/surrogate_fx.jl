# surrogate_fx.jl — surrogate derivative functions for (binary) spike emission.
#
# 1:1 port of the parts of ngclearn/utils/surrogate_fx.py that the consuming
# component needs. Upstream exposes several estimators (heaviside, triangular,
# arctan, secant_lif, ...); only `secant_lif_estimator` is ported here — it is
# the surrogate required by SLIFCell / the Samadi-et-al. (2017) BFA exhibit.
# Others land when a component that uses them is ported (same "port-on-demand"
# rule as model_utils.jl).
#
# Convention (mirrors upstream's estimator factories):
#   - `spike_fx(v, thr)`   = binary spike emission  (Heaviside, v > thr)
#   - `d_spike_fx(j; ...)` = surrogate derivative of the spike w.r.t. drive j
#   - `secant_lif_estimator()` returns the `(spike_fx, d_spike_fx)` pair, as the
#     upstream factory does (get_surr_fx defaults to False there).

# ── secant LIF estimator (Samadi et al., 2017) ───────────────────────────────

"""
    secant_spike_fx(v, thr)

Binary spike emission: `1.0` where `v > thr`, else `0.0`. Mirrors `spike_fx`
inside `secant_lif_estimator` (surrogate_fx.py:108-110) — note the strict `>`.
"""
@inline secant_spike_fx(v, thr) = (v .> thr) .* 1.0

"""
    secant_d_spike_fx(j; c1=0.82, c2=0.08, omit_scale=true)

Surrogate derivative of the spike function w.r.t. the electrical drive `j`
(Samadi et al., 2017). Returns `sech(c2·j)` for `j > 0` and `0` for `j ≤ 0`,
optionally scaled by `c1·c2`.

PORT FIDELITY: the upstream docstring advertises `sech²(c2·j)`, but the *code*
computes `dv_dj = sech_j` to the **first** power (the `* (c1 * c2)` is commented
out and gated behind `omit_scale=False`) — see surrogate_fx.py:148-153. This is a
1:1 port of the code, not the docstring. `thr` is accepted upstream but UNUSED.
"""
function secant_d_spike_fx(j; c1::Real=0.82, c2::Real=0.08, omit_scale::Bool=true)
    mask = (j .> 0.0) .* 1.0
    dj = j .* c2
    cosh_j = (exp.(dj) .+ exp.(-dj)) ./ 2.0
    sech_j = 1.0 ./ cosh_j
    dv_dj = sech_j                       # first power — matches the code, not the docstring
    if !omit_scale
        dv_dj = dv_dj .* (c1 * c2)
    end
    return dv_dj .* mask                 # 0 for j ≤ 0
end

"""
    secant_lif_estimator() -> (spike_fx, d_spike_fx)

Return the `(secant_spike_fx, secant_d_spike_fx)` pair, mirroring the upstream
`secant_lif_estimator(get_surr_fx=False)` factory (surrogate_fx.py:91-159).
"""
secant_lif_estimator() = (secant_spike_fx, secant_d_spike_fx)
