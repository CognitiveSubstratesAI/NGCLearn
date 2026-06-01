# model_utils.jl — activation + threshold helpers used by the component zoo.
#
# Phase-A scope: just what the first round of cells (RateCell, future
# DenseSynapse update rules, pc_discrim exhibit) actually need. Upstream
# `model_utils.py` exports a much larger catalogue (lrelu, elu, silu, gelu,
# softmax, softplus, bkwta, etc.); ports land as the consuming component is
# written, not pre-emptively.
#
# Convention:
#   - `fx(x)` = the activation
#   - `d_<name>(x)` = the elementwise derivative
#   - `create_function(name)` returns `(fx, dfx)` (mirrors upstream's
#     `create_function` API — RateCell stores the pair on construction)
#
# Names that already exist in `Base` (`identity`, `tanh`) are re-used directly;
# we only introduce new exports for the derivatives and the rectified families.
# See NGCSimLib `docs/decisions.md` §3 (don't shadow Base) — that rule applies
# verbatim here.

# ── Activations ──────────────────────────────────────────────────────────────

"""
    d_identity(x)

Derivative of the identity. Returns ones the same shape as `x`.
Mirrors `d_identity` (model_utils.py:290).
"""
@inline d_identity(x) = zero.(x) .+ 1.0

"""
    relu(x)

ReLU / linear rectifier: `max(0, x)` elementwise. Mirrors `relu`
(model_utils.py:303).
"""
@inline relu(x) = max.(zero.(x), x)

"""
    d_relu(x)

Derivative of ReLU: indicator `(x ≥ 0)`. Mirrors `d_relu`
(model_utils.py:316).
"""
@inline d_relu(x) = (x .>= 0.0) .* 1.0

"""
    d_tanh(x)

Derivative of `Base.tanh`: `1 - tanh(x)^2`. Mirrors `d_tanh`
(model_utils.py:411).
"""
@inline d_tanh(x) = (t=tanh.(x); 1.0 .- t .* t)

"""
    sigmoid(x)

Logistic-link function `1 / (1 + exp(-x))`. Mirrors `sigmoid`
(model_utils.py:589). Defined here because `Base` doesn't ship it.
"""
@inline sigmoid(x) = 1.0 ./ (1.0 .+ exp.(.-x))

"""
    d_sigmoid(x)

Derivative of the sigmoid: `σ(x) * (1 - σ(x))`. Mirrors `d_sigmoid`
(model_utils.py:605).
"""
@inline d_sigmoid(x) = (s=sigmoid(x); s .* (1.0 .- s))

# ── Thresholds (no derivatives — upstream model_utils.py:794, 812 notes the
#                companion derivative is intentionally omitted) ──────────────

"""
    threshold_soft(x, lmbda)

Soft-threshold (elementwise): `max(x - λ, 0) - max(-x - λ, 0)`. Mirrors
`threshold_soft` (model_utils.py:791).
"""
@inline threshold_soft(x, lmbda) = max.(x .- lmbda, 0.0) .- max.(.-x .- lmbda, 0.0)

"""
    threshold_cauchy(x, lmbda)

Cauchy distributional threshold (elementwise). Mirrors `threshold_cauchy`
(model_utils.py:809). The companion derivative is intentionally not provided
upstream (see source comment).
"""
@inline function threshold_cauchy(x, lmbda)
    inner = sqrt.(max.(x .* x .- lmbda, 0.0))
    f = (x .+ inner) .* 0.5
    g = (x .- inner) .* 0.5
    return f .* (x .> 0.0) .+ g .* (x .< 0.0)
end

# ── create_function: (fx, dfx) lookup keyed by upstream string names ─────────

"""
    create_function(fun_name::AbstractString) -> (fx::Function, dfx::Function)

Map an activation name to its `(fx, dfx)` pair. Mirrors `create_function`
(model_utils.py:69-138). Throws on unknown name (matches upstream).

Phase-A coverage: `"identity"`, `"tanh"`, `"relu"`, `"sigmoid"`. Names beyond
this set are added on demand as new components import them.
"""
function create_function(fun_name::AbstractString)
    if fun_name == "identity"
        return (Base.identity, d_identity)
    elseif fun_name == "tanh"
        return ((x) -> tanh.(x), d_tanh)
    elseif fun_name == "relu"
        return (relu, d_relu)
    elseif fun_name == "sigmoid"
        return (sigmoid, d_sigmoid)
    else
        error("Activation function ($fun_name) is not recognized/supported!")
    end
end

# ── Matrix normalization ─────────────────────────────────────────────────────

"""
    normalize_matrix(data, wnorm; order=1, axis=1) -> Matrix

Rescale each vector span of `data` to have target norm `wnorm`. Mirrors
`normalize_matrix` (model_utils.py:157-185).

  - `order=1` → L1 norm (default); `order=2` → L2 norm.
  - `axis=1` normalizes each COLUMN (upstream `axis=0`, NumPy row-axis); `axis=2`
    normalizes each ROW. (Julia's `dims` is 1-based and transposed relative to
    NumPy's `axis`, so the DC-SNN call `axis=0` ⇒ per-column ⇒ `axis=1` here.)

A small floor (1e-8) prevents division by zero for all-zero spans.
"""
function normalize_matrix(
    data::AbstractMatrix, wnorm::Real; order::Integer=1, axis::Integer=1
)
    dims = axis == 1 ? 1 : 2
    if order == 2
        denom = max.(sqrt.(sum(abs2, data; dims=dims)), 1e-8)
    else
        denom = max.(sum(abs, data; dims=dims), 1e-8)
    end
    return data .* (wnorm ./ denom)
end
