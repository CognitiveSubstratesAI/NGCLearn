# gaussian_error_cell.jl — fixed-point Gaussian error/mismatch cell.
#
# 1:1 port of ngclearn/components/neurons/graded/gaussianErrorCell.py.
#
# A non-spiking predictive-coding error unit. Given a prediction `mu`, a target
# `target`, and a (co)variance `Sigma`, it computes the precision-weighted
# mismatch and the local Gaussian log-likelihood:
#   e        = target - mu
#   dmu      = e / Sigma                 (precision-scaled error, sent to mu)
#   dtarget  = -dmu
#   L        = -sum(e^2) * (0.5 / Sigma) (local free energy / log density)
# `dmu`/`dtarget` are gated by `modulator` and a one-shot `mask`.
#
# Spec: docs/specs/03_gaussian_error_cell.md.
#
# Scope note (see docs/decisions.md #4): the scalar-`sigma` path is ported and
# tested. Upstream also supports a 4-D convolutional `shape` and a full
# covariance matrix; the field plumbing for `shape` is preserved, but the
# covariance-matrix reduction in `L` follows upstream's scalar-collapse
# behavior. Full-covariance `L` is deferred until a component needs it.

"""
    GaussianErrorCell <: JaxComponent

A fixed-point Gaussian error cell computing a precision-weighted mismatch
signal and its local log-likelihood.

| Input compartments:  `mu`, `Sigma`, `target`, `modulator`, `mask`
| Output compartments: `L` (log-likelihood), `dmu`, `dSigma`, `dtarget`

Construct with [`GaussianErrorCell(; ...)`](@ref). Required: `name`, `n_units`.
"""
mutable struct GaussianErrorCell <: JaxComponent
    # Standard component fields.
    name::String
    context_path::String
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}

    # Hyperparameters / shape config.
    n_units::Int
    batch_size::Int
    shape::Tuple
    sigma_shape::Tuple
    width::Int
    height::Int

    # Compartments.
    key::NGCSimLib.Compartment
    L::NGCSimLib.Compartment
    mu::NGCSimLib.Compartment
    dmu::NGCSimLib.Compartment
    Sigma::NGCSimLib.Compartment
    dSigma::NGCSimLib.Compartment
    target::NGCSimLib.Compartment
    dtarget::NGCSimLib.Compartment
    modulator::NGCSimLib.Compartment
    mask::NGCSimLib.Compartment
end

"""
    GaussianErrorCell(; name, n_units, batch_size=1, sigma=1.0, shape=nothing,
                      key=nothing, context_path="", args=Any[],
                      kwargs=Dict{Symbol,Any}())

Keyword constructor mirroring `GaussianErrorCell.__init__`
(gaussianErrorCell.py:36-67). With `shape === nothing` the cell is a
`(batch_size, n_units)` matrix; passing a 3-tuple `shape` makes it a 4-D
`(batch_size, shape...)` tensor.
"""
function GaussianErrorCell(;
    name::AbstractString,
    n_units::Integer,
    batch_size::Integer=1,
    sigma::Real=1.0,
    shape::Union{Tuple, Nothing}=nothing,
    key::Union{Integer, Nothing}=nothing,
    context_path::AbstractString="",
    args::Vector{Any}=Any[],
    kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    if shape === nothing
        _shape = (batch_size, n_units)
        eff_shape = (n_units,)
    else
        _shape = (batch_size, shape[1], shape[2], shape[3])
        eff_shape = shape
    end
    sigma_shape = (1, 1)
    rest = zeros(Float64, _shape)
    _sigma_block = zeros(Float64, sigma_shape)

    return GaussianErrorCell(
        String(name),
        String(context_path),
        args,
        kwargs,
        Int(n_units),
        Int(batch_size),
        eff_shape,
        sigma_shape,
        Int(n_units),   # width
        Int(n_units),   # height
        NGCSimLib.Compartment(make_prng_key(key)),
        NGCSimLib.Compartment(0.0; display_name="Gaussian Log likelihood", units="nats"),
        NGCSimLib.Compartment(copy(rest); display_name="Gaussian mean"),
        NGCSimLib.Compartment(copy(rest)),
        NGCSimLib.Compartment(
            _sigma_block .+ sigma; display_name="Gaussian variance/covariance"
        ),
        NGCSimLib.Compartment(copy(_sigma_block)),
        NGCSimLib.Compartment(copy(rest); display_name="Gaussian data/target variable"),
        NGCSimLib.Compartment(copy(rest)),
        NGCSimLib.Compartment(rest .+ 1.0),
        NGCSimLib.Compartment(rest .+ 1.0)
    )
end

# Squeeze a 1-element array to a scalar (the analog of `jnp.squeeze` on a
# (1,1) log-density); pass non-arrays / multi-element arrays through unchanged.
_squeeze(x::AbstractArray) = length(x) == 1 ? x[begin] : x
_squeeze(x) = x

# ── Dynamics ──────────────────────────────────────────────────────────────────

# Mirrors `advance_state` (gaussianErrorCell.py:81-118).
NGCSimLib.@compilable function advance_state!(c::GaussianErrorCell, dt)
    mu = NGCSimLib.get_value(c.mu)
    target = NGCSimLib.get_value(c.target)
    Sigma = NGCSimLib.get_value(c.Sigma)
    modulator = NGCSimLib.get_value(c.modulator)
    mask = NGCSimLib.get_value(c.mask)

    _dmu = target .- mu               # raw error e
    dmu = _dmu ./ Sigma               # precision-scaled error
    dtarget = -dmu
    dSigma = Sigma .* 0 .+ 1.0        # no Sigma derivative computed yet (upstream parity)
    L = -sum(abs2, _dmu) .* (0.5 ./ Sigma)

    dmu = dmu .* modulator .* mask
    dtarget = dtarget .* modulator .* mask
    mask = mask .* 0.0 .+ 1.0         # "eat" the one-shot mask

    NGCSimLib.set!(c.dmu, dmu)
    NGCSimLib.set!(c.dtarget, dtarget)
    NGCSimLib.set!(c.dSigma, dSigma)
    NGCSimLib.set!(c.L, _squeeze(L))
    NGCSimLib.set!(c.mask, mask)
    return c
end

# Mirrors `reset` (gaussianErrorCell.py:121-148).
NGCSimLib.@compilable function reset_state!(c::GaussianErrorCell)
    if length(c.shape) > 1
        _shape = (c.batch_size, c.shape[1], c.shape[2], c.shape[3])
    else
        _shape = (c.batch_size, c.shape[1])
    end
    rest = zeros(Float64, _shape)

    NGCSimLib.set!(c.dmu, copy(rest))
    NGCSimLib.set!(c.dtarget, copy(rest))
    NGCSimLib.set!(c.dSigma, zeros(Float64, c.sigma_shape))
    NGCSimLib.set!(c.target, copy(rest))
    NGCSimLib.set!(c.mu, copy(rest))
    NGCSimLib.set!(c.modulator, rest .+ 1.0)
    NGCSimLib.set!(c.L, 0.0)
    NGCSimLib.set!(c.mask, ones(Float64, _shape))
    return c
end
