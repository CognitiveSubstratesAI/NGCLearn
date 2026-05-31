# Port spec: `HebbianSynapse`

Upstream: `ngclearn/components/synapses/hebbian/hebbianSynapse.py`
Port: `src/components/synapses/hebbian_synapse.jl`

A dense synaptic cable that adapts its weights via a two-factor Hebbian rule
plus an embedded optimizer (SGD or Adam). Forward pass is identical to
`DenseSynapse`; the addition is plasticity.

## Forward dynamics (same as DenseSynapse)

    outputs = (inputs * (weights ⊙ mask)) * resist_scale + biases

## Update rule — `_calc_update`, maps hebbianSynapse.py:17-69

    dW = (pre * pre_wght)' * (post * post_wght)     # two-factor Hebbian
    db = sum(post * post_wght, dims=1)
    if w_bound > 0:  dW *= (w_bound - |W|)          # soft bound
    dW += dW_reg                                    # optional prior
    return (dW, db) * sign_value

Prior (`dW_reg`) types: `l2`/`ridge` → `-W*lmbda`; `l1`/`lasso` →
`-sign(W)*lmbda`; `l1l2`/`elastic_net` → `lmbda = (scale, l1_ratio)`; any other
(incl. `constant`) → none. Upstream aliases `gaussian→ridge`, `laplacian→lasso`.

`_enforce_constraints` (hebbianSynapse.py:72-93): with `w_bound>0`, hard-clips W
to `[0, w_bound]` (`is_nonnegative`) or `[-w_bound, w_bound]`; else returns W.

## Compartments

| kind | name | init |
|---|---|---|
| input | `inputs`, `pre`, `post` | zeros |
| state | `weights` | dist `(n_in, n_out)` (default `uniform(-0.3, 0.3)`) |
| state | `biases` | dist `(1, n_out)` or zeros (not learned if `bias_init=nothing`) |
| state | `mask`, `key` | gate / `UInt64` seed |
| state | `opt_params` | optimizer state (NamedTuple, SGD/Adam) |
| output | `outputs`, `dWeights`, `dBiases` | zeros |

## Methods

- `advance_state!(c)` — forward pass (mirrors DenseSynapse.advance_state).
- `compute_update!(c)` — fill `dWeights`/`dBiases` from `pre`/`post`/`W` without
  stepping (maps `calc_update`, hebbianSynapse.py:243-259).
- `evolve!(c, dt)` — compute update AND step the optimizer to evolve
  `weights`/`biases`, then re-apply bound + mask (maps `evolve`,
  hebbianSynapse.py:262-301).
- `reset_state!(c)` — zero plasticity compartments; `inputs` only if untargeted.

## Divergences (see `docs/decisions.md`)

- **#1** Julia has no single-class inheritance for mutable structs, so the
  `DenseSynapse` fields are duplicated explicitly rather than inherited. **#2**
  `UInt64` PRNG seed. Default `weight_init=uniform(-0.3, 0.3)` (pc_discrim's
  Hebbian default) rather than DenseSynapse's `uniform(0.025, 0.8)`.

## Verification

`test/test_hebbian_synapse.jl`, eager path: construction shapes + Hebbian
compartments; forward pass; 2-factor `compute_update!`; `sign_value=-1` flips dW;
l2 prior; `evolve!` with SGD descends; `evolve!` with Adam direction; `w_bound` +
`is_nonnegative` clamp; `reset_state!`; prior-type aliases.
