# Port spec: `DenseSynapse`

Upstream: `ngclearn/components/synapses/denseSynapse.py`
Port: `src/components/synapses/dense_synapse.jl`

(Upstream `StaticSynapse` is an alias for this dense linear cable; the
pc_discrim feedback/inference synapses E2/E3/Q1-Q3 are `DenseSynapse`s.)

## Dynamics (linear forward pass, no in-built learning)

    outputs = (inputs * (weights ⊙ mask)) * resist_scale + biases

`mask` is a multiplicative gate over `weights` (zero entries implement
sparsity / connectivity constraints).

## Compartments

| kind | name | init |
|---|---|---|
| input | `inputs` | zeros `(batch, n_in)` |
| state | `weights` | dist `(n_in, n_out)` (default `uniform(0.025, 0.8)`) |
| state | `biases` | dist `(1, n_out)` or zeros if `bias_init=nothing` |
| state | `mask` | explicit override or `(1,1)` broadcast-passthrough |
| state | `key` | `UInt64` PRNG seed |
| output | `outputs` | zeros `(batch, n_out)` |

Weight init distributions (`_init_array`): `("uniform", amin, amax)`,
`("gaussian"/"normal", mu, sigma)`, `("constant", c)`. A `p_conn < 1` argument
sparsifies the weight matrix at construction via a Bernoulli mask.

## `advance_state!(c)` — maps denseSynapse.py:98-102

Compute the masked linear forward pass above.

## `reset_state!(c)` — maps denseSynapse.py:104-109

Zero `outputs`; reset `inputs` only when not externally `targeted` (matches
LIFCell.reset semantics).

## Divergences (see `docs/decisions.md`)

- **#1** explicit fields, no `@ngc_component`. **#2** `UInt64` PRNG seed via
  `Xoshiro` rather than a JAX key array (sub-key splitting → sequential RNG
  advance, preserving "fresh value per init call").

## Verification

`test/test_dense_synapse.jl`, eager path: construction defaults + shapes; linear
forward pass; `resist_scale`; `bias_init`; `mask`-zero suppression; `p_conn<1`
sparsity; `reset_state!`; explicit-key reproducibility; unsupported-dist error.
