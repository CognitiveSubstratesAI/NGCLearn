# Port spec: `SLIFCell`

Upstream: `ngclearn/components/neurons/spiking/sLIFCell.py`
Port: `src/components/neurons/spiking/slif_cell.jl`
Surrogate: `ngclearn/utils/surrogate_fx.py` → `src/utils/surrogate_fx.jl`

The simplified-LIF spiking cell of the Samadi-et-al. (2017) broadcast/feedback-
alignment exhibit (`bfa_snn`). Adds a surrogate spike derivative, a per-unit
adaptive threshold, sticky spikes, and optional lateral inhibition on top of LIF.

## Dynamics

    j' = j * resist_m  − (s_prev · Wi) * resist_inh     (lateral term only if resist_inh > 0)
    tau_m * dv/dt = (-v + j') · mask            mask = 1[rfr ≥ refract_T]   (Euler only)
    s = 1[v > thr]                              (strict >, secant_spike_fx)
    on spike:  v <- (1-s)*v   (hyperpolarize to 0)
    surrogate = sech(0.08 · j')  for j' > 0, else 0      (secant_d_spike_fx)
    tols = (1-s)*tols + s*t

Adaptive threshold (`_update_threshold`):

    rho_b > 0:  thr <- max(thr + (Σ_units s − 1)·rho_b, 0.025)      (sparsity mode)
    else:       thr <- thr + s·thr_gain − thr·thr_leak

Refractory + sticky (`_update_refract_and_spikes`):

    rfr <- (rfr + dt)·(1-s) + s·dt
    if sticky_spikes:  s <- s·mask + (1 − mask)    (pin to 1 through refractory window)

## Compartments

| kind | name | init |
|---|---|---|
| input | `j` | zeros `(1, n_units)` |
| state | `v` | 0 (note: rest is 0, not a `v_rest`) |
| state | `thr` | `threshold0` = `thr ± thr_jitter` per unit, shape `(1, n_units)` |
| state | `rfr` | `refract_T` |
| state | `key` | PRNG seed (`UInt64`) |
| output | `s` | 0 |
| output | `surrogate` | 1 |
| output | `tols` | 0 |

Fixed (non-compartment) fields: `inh_weights` (hollow `uniform(0.025,1)`,
diagonal 0) and `threshold0`, both drawn once from the seed at construction.

## `advance_state!(c, dt, t)` — maps sLIFCell.py:164-217

1. `j = get(c.j) * R_m`; if `inh_R > 0` subtract `(get(c.s) · inh_weights) * inh_R`;
   write processed `j` back (upstream does too).
2. `surrogate = secant_d_spike_fx(j)`.
3. integrate `v` one Euler step with `_dfv_slif`, `params = (j, rfr, tau_m, refract_T)`.
4. `s = 1[_v > thr]`; hyperpolarize `_v = (1-s)*_v`.
5. `thr <- _update_threshold_slif(...)`; `(rfr, s) <- _update_refract_and_spikes_slif(...)`.
6. `tols = (1-s)*tols + s*t`; write back `v, s, thr, rfr, surrogate, tols`.

## `reset_state!(c)` — maps sLIFCell.py:219-238

Reset `v, s, tols → 0`, `rfr → refract_T`, `surrogate → 1`; `j` reset only if not
externally `targeted`. Threshold reset to `threshold0` **iff** `thr_persist` is false.

## Surrogate (`secant_lif_estimator`)

`spike_fx(v, thr) = 1[v > thr]`. `d_spike_fx(j; c1, c2, omit_scale=true) = mask·sech(c2·j)`,
optionally `·(c1·c2)`.

**PORT FIDELITY (comment-that-lies):** the upstream docstring claims `sech²(c2·j)`,
but the code computes `dv_dj = sech_j` to the **first** power (surrogate_fx.py:148-153,
the `* (c1*c2)` disabled behind `omit_scale=False`). The port follows the code.

## Divergences (see `docs/decisions.md`)

- `reset` guards `j` with `targeted(c.j)` (LIFCell convention, #6); upstream sets
  it unconditionally.
- `self.v_min = -3.` is set upstream but never applied in `advance_state`; ported
  as a stored field for fidelity, not wired into the dynamics.
- PRNG: `UInt64` seed via `Xoshiro` (decisions #2). The exact `inh_weights` /
  `threshold0` draws differ from JAX; only the dynamics are part of the contract.

## Verification

`test/test_slif_cell.jl`, eager path, hand-computed (44 tests): surrogate values
(`sech`, strict `>`, mask, scale); construction/shapes; sub-threshold step;
supra-threshold spike (hyperpolarize + adaptive `thr` + `tols`); sticky-spike
pinning across the refractory window; `rho_b` sparsity threshold (incl. 0.025
floor); lateral inhibition (known `inh_weights`); `reset_state!` with persistent
and non-persistent thresholds.
