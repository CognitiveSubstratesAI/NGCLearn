```@meta
CurrentModule = NGCLearn
```

# API Reference

Auto-generated from inline docstrings. To avoid duplicate entries, each symbol
is rendered on exactly one page:

- **Components** (the shared protocol, neurons, synapses, encoders, traces) are
  documented on the [Components](../components.md) page.
- **Models** ([`PCN`](@ref), [`DC_SNN`](@ref) and their drivers) are documented
  on the [Models](../models.md) page.

This page covers the remaining backend: the integration utilities, activation /
matrix helpers, and optimizers.

## Integration backend

```@docs
get_integrator_code
step_euler
step_rk2
```

## Activations & utilities

```@docs
create_function
relu
sigmoid
d_identity
d_tanh
d_relu
d_sigmoid
threshold_soft
threshold_cauchy
normalize_matrix
```

## Optimizers

```@docs
sgd_init
sgd_step
adam_init
adam_step
get_opt_init_fn
get_opt_step_fn
```

## Index

```@index
```
