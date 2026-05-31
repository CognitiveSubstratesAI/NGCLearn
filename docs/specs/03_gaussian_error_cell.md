# Port spec: `GaussianErrorCell`

Upstream: `ngclearn/components/neurons/graded/gaussianErrorCell.py`
Port: `src/components/neurons/graded/gaussian_error_cell.jl`

## Dynamics (fixed-point, non-spiking)

    e        = target - mu
    dmu      = e / Sigma                  # precision-scaled error → prediction
    dtarget  = -dmu
    dSigma   = 1                          # no Sigma gradient computed (upstream parity)
    L        = -sum(e^2) * (0.5 / Sigma)  # local Gaussian log density (squeezed to scalar)
    dmu      *= modulator * mask
    dtarget  *= modulator * mask
    mask     <- 1                         # one-shot mask is "eaten" each step

## Compartments

| kind | name | init |
|---|---|---|
| input | `mu`, `target` | zeros `(batch, n_units)` |
| input | `Sigma` | `sigma` over `(1,1)` block |
| input | `modulator`, `mask` | ones |
| output | `L` | `0.0` (scalar) |
| output | `dmu`, `dtarget` | zeros |
| output | `dSigma` | zeros `(1,1)` |
| (base) | `key` | PRNG seed |

## `advance_state!(c, dt)` — maps gaussianErrorCell.py:81-118

Computes the block above. `L` is reduced to a scalar via `_squeeze` (the analog
of `jnp.squeeze` on the `(1,1)` log density).

## `reset_state!(c)` — maps gaussianErrorCell.py:121-148

Reset compartments to init, recomputing `_shape` (2-D, or 4-D when a conv
`shape` was supplied).

## Divergences (see `docs/decisions.md`)

- **#5** — scalar-`sigma` path ported and tested. `shape` field plumbing for the
  4-D conv case is preserved, but full-covariance-matrix `L` is deferred until a
  downstream model exercises it.
- PRNG `key` compartment present for base-class parity even though
  `advance_state!` does not consume it (the cell is deterministic).

## Verification

`test/test_gaussian_error_cell.jl`, eager path, hand-computed: construction;
mismatch + precision-scaling + `L` at `sigma=1` (`dmu=[1,2]`, `L=-2.5`); scaling
at `sigma=2`; modulator + one-shot-mask gating; `reset_state!`.
