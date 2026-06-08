# slif_cell.jl — simplified leaky integrate-and-fire spiking neuron.
#
# 1:1 port of ngclearn/components/neurons/spiking/sLIFCell.py.
#
# This is the spiking cell of the Samadi-et-al. (2017) broadcast/feedback-
# alignment exhibit (bfa_snn). Beyond a plain LIF it adds, all per upstream:
#   - a SURROGATE spike derivative (the secant LIF estimator) emitted on `surrogate`
#   - an ADAPTIVE per-unit threshold `thr` (gain/leak, or an `rho_b` sparsity mode)
#   - optional "STICKY SPIKES" (spikes pinned to 1 through the refractory period)
#   - optional heuristic LATERAL INHIBITION via a fixed hollow `inh_weights` matrix
# Voltage integration is Euler-only upstream (no RK2 branch).
#
# Dynamics (sLIFCell.py:_dfv): tau_m * dv/dt = (-v + j) during the non-refractory
# window (j already scaled by resist_m and, if enabled, lateral inhibition).
#
# Spec: docs/specs/07_slif_cell.md.
#
# Upstream divergences:
#   - `reset` guards the input compartment `j` with `targeted(c.j)` so an
#     externally-wired drive is not clobbered — the same convention the LIFCell
#     port adopted (decisions.md #6 / the LIFCell header). Upstream sets `j`
#     unconditionally.
#   - `self.v_min = -3.` is set in upstream `__init__` but NEVER applied in
#     `advance_state` (dead there, like LIFCell's commented surrogate); ported as
#     a stored field for fidelity, not wired into the dynamics.
#   - The surrogate derivative ports the upstream CODE (`sech`, first power), not
#     its docstring (`sech²`) — see surrogate_fx.jl.

# ── Module-level co-routines (mirror the sLIFCell.py free functions) ──────────

# Voltage dynamics dv/dt. Mirrors `_dfv`/`_dfv_internal` (sLIFCell.py:12-24).
# params = (j, rfr, tau_m, refract_T). Note: NO v_rest / g_L / resist_m here —
# resist_m is applied to `j` before integration.
function _dfv_slif(t, v, params)
    j, rfr, tau_m, refract_T = params
    mask = (rfr .>= refract_T) .* 1.0          # refractory mask
    dv_dt = (.-v .+ j) .* (1.0 / tau_m) .* mask
    return dv_dt
end

# Adaptive-threshold update. Mirrors `_update_threshold` (sLIFCell.py:26-36).
function _update_threshold_slif(dt, v_thr, spikes, thr_gain, thr_leak, rho_b)
    if rho_b > 0.0                              # sparsity-enforcement mode (ignores gain/leak)
        dthr = sum(spikes; dims=2) .- 1.0       # (batch,1), keepdims
        return max.(v_thr .+ dthr .* rho_b, 0.025)
    else                                        # simple adaptive threshold
        return v_thr .+ spikes .* thr_gain .- v_thr .* thr_leak
    end
end

# Refractory + (optionally sticky) spike update. Mirrors
# `_update_refract_and_spikes` (sLIFCell.py:38-46).
function _update_refract_and_spikes_slif(dt, rfr, s, refract_T, sticky_spikes)
    _rfr = (rfr .+ dt) .* (1.0 .- s) .+ s .* dt # reset refractory to dt where spiked
    _s = s
    if sticky_spikes
        mask = (rfr .>= refract_T) .* 1.0
        _s = s .* mask .+ (1.0 .- mask)         # pin to 1 during refractory window
    end
    return _rfr, _s
end

# ── Component type ────────────────────────────────────────────────────────────

"""
    SLIFCell <: JaxComponent

A simplified leaky integrate-and-fire spiking cell (Samadi et al., 2017): LIF
voltage dynamics plus a secant surrogate spike derivative, a per-unit adaptive
threshold, optional sticky spikes, and optional heuristic lateral inhibition.

| Input compartments:  `j` (electrical current)
| State compartments:  `v` (voltage), `rfr` (refractory), `thr` (adaptive
|                      threshold), `key` (PRNG seed)
| Output compartments: `s` (spikes), `surrogate` (secant derivative), `tols`
|                      (time-of-last-spike)

Construct with [`SLIFCell(; ...)`](@ref). Required: `name`, `n_units`, `tau_m`,
`resist_m`, `thr`.
"""
mutable struct SLIFCell <: JaxComponent
    # Standard component fields (jax_component.jl, decisions.md #1).
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters (scalars / fixed arrays — NOT compartments).
    n_units::Int
    batch_size::Int
    tau_m::Float64
    R_m::Float64               # resist_m
    refract_T::Float64         # refract_time (ms)
    v_min::Float64             # set upstream but UNUSED in advance (see header)
    sticky_spikes::Bool
    inh_R::Float64             # resist_inh: lateral inhibition magnitude (0 ⇒ off)
    rho_b::Float64             # threshold sparsity factor
    thr_persist::Bool
    thrGain::Float64
    thrLeak::Float64
    inh_weights::Matrix{Float64}   # fixed hollow lateral-inhibition matrix
    threshold0::Matrix{Float64}    # per-unit threshold initial condition, (1, n_units)

    # State / I-O compartments.
    key::NGCSimLib.Compartment
    j::NGCSimLib.Compartment
    v::NGCSimLib.Compartment
    thr::NGCSimLib.Compartment
    rfr::NGCSimLib.Compartment
    s::NGCSimLib.Compartment
    surrogate::NGCSimLib.Compartment
    tols::NGCSimLib.Compartment
end

"""
    SLIFCell(; name, n_units, tau_m, resist_m, thr, resist_inh=0.0,
             thr_persist=false, thr_gain=0.0, thr_leak=0.0, rho_b=0.0,
             refract_time=0.0, sticky_spikes=false, thr_jitter=0.05,
             batch_size=1, key=nothing, context_path="", args=Any[],
             kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `SLIFCell.__init__` (sLIFCell.py:115-162). The
fixed `inh_weights` (hollow, `uniform(0.025, 1)`) and per-unit `threshold0`
(`thr ± thr_jitter`) are drawn once from the PRNG seed at construction.
"""
function SLIFCell(;
    name::AbstractString,
    n_units::Integer,
    tau_m::Real,
    resist_m::Real,
    thr::Real,
    resist_inh::Real=0.0,
    thr_persist::Bool=false,
    thr_gain::Real=0.0,
    thr_leak::Real=0.0,
    rho_b::Real=0.0,
    refract_time::Real=0.0,
    sticky_spikes::Bool=false,
    thr_jitter::Real=0.05,
    batch_size::Integer=1,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    resist_m > 0.0 || error("SLIFCell: resist_m must be > 0 (got $resist_m)")
    seed = make_prng_key(key)
    rng = Xoshiro(seed)
    rest = zeros(Float64, batch_size, n_units)

    # Hollow lateral-inhibition matrix: uniform(0.025, 1), zero diagonal.
    inh_weights = 0.025 .+ (1.0 - 0.025) .* rand(rng, Float64, n_units, n_units)
    @inbounds for i in 1:n_units
        inh_weights[i, i] = 0.0
    end
    # Per-unit threshold initial condition with uniform jitter in [-thr_jitter, thr_jitter].
    threshold0 =
        thr .+ (-thr_jitter .+ (2.0 * thr_jitter) .* rand(rng, Float64, 1, n_units))

    return SLIFCell(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        Int(batch_size),
        Float64(tau_m),
        Float64(resist_m),
        Float64(refract_time),
        -3.0,
        sticky_spikes,
        Float64(resist_inh),
        Float64(rho_b),
        thr_persist,
        Float64(thr_gain),
        Float64(thr_leak),
        inh_weights,
        threshold0,
        NGCSimLib.Compartment(seed),
        NGCSimLib.Compartment(copy(rest); display_name="Current", units="mA"),
        NGCSimLib.Compartment(copy(rest); display_name="Voltage", units="mV"),
        NGCSimLib.Compartment(threshold0 .+ 0.0; display_name="Threshold", units="mV"),
        NGCSimLib.Compartment(
            rest .+ refract_time; display_name="Refractory Time Period", units="ms"
        ),
        NGCSimLib.Compartment(copy(rest); display_name="Spikes"),
        NGCSimLib.Compartment(rest .+ 1.0; display_name="Surrogate State"),
        NGCSimLib.Compartment(copy(rest); display_name="Time-of-Last-Spike", units="ms")
    )
end

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (sLIFCell.py:164-217). Euler-only.
NGCSimLib.@compilable function advance_state!(c::SLIFCell, dt, t)
    # Modify drive by membrane resistance and (optional) lateral inhibition.
    j = NGCSimLib.get_value(c.j) .* c.R_m
    if c.inh_R > 0.0
        j = j .- (NGCSimLib.get_value(c.s) * c.inh_weights) .* c.inh_R
    end
    NGCSimLib.set!(c.j, j)                       # upstream writes processed j back

    surrogate = secant_d_spike_fx(j; c1=0.82, c2=0.08)

    v_params = (j, NGCSimLib.get_value(c.rfr), c.tau_m, c.refract_T)
    _, _v = step_euler(0.0, NGCSimLib.get_value(c.v), _dfv_slif, dt, v_params)

    thr = NGCSimLib.get_value(c.thr)
    spikes = secant_spike_fx(_v, thr)
    _v = (1.0 .- spikes) .* _v                   # hyperpolarize spiking cells
    new_thr = _update_threshold_slif(dt, thr, spikes, c.thrGain, c.thrLeak, c.rho_b)
    _rfr, spikes = _update_refract_and_spikes_slif(
        dt, NGCSimLib.get_value(c.rfr), spikes, c.refract_T, c.sticky_spikes
    )
    tols = (1.0 .- spikes) .* NGCSimLib.get_value(c.tols) .+ (spikes .* t)

    NGCSimLib.set!(c.v, _v)
    NGCSimLib.set!(c.s, spikes)
    NGCSimLib.set!(c.thr, new_thr)
    NGCSimLib.set!(c.rfr, _rfr)
    NGCSimLib.set!(c.surrogate, surrogate)
    NGCSimLib.set!(c.tols, tols)
    return c
end

# Mirrors `reset` (sLIFCell.py:219-238). Non-persistent thresholds reset to the
# per-unit `threshold0`; `j` is guarded so external wiring is preserved.
NGCSimLib.@compilable function reset_state!(c::SLIFCell)
    rest = zeros(Float64, c.batch_size, c.n_units)
    if !NGCSimLib.targeted(c.j)
        NGCSimLib.set!(c.j, copy(rest))
    end
    NGCSimLib.set!(c.v, copy(rest))
    NGCSimLib.set!(c.s, copy(rest))
    NGCSimLib.set!(c.tols, copy(rest))
    NGCSimLib.set!(c.rfr, rest .+ c.refract_T)
    NGCSimLib.set!(c.surrogate, rest .+ 1.0)
    if !c.thr_persist
        NGCSimLib.set!(c.thr, c.threshold0 .+ 0.0)
    end
    return c
end
