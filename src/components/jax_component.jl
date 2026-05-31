# jax_component.jl — abstract base for the NGCLearn component zoo.
#
# Upstream `JaxComponent` (ngclearn/components/jaxComponent.py) is a thin
# `Component` subclass that:
#   1. owns a PRNG `key` Compartment (seeded from the wall clock if not given),
#   2. provides default `save`/`load` over non-wired compartments,
#   3. provides a `__repr__` that tensor-stats every compartment.
#
# In Julia we model the class as an abstract type so concrete cells dispatch
# `advance_state!` / `reset_state!` polymorphically. Every NGCLearn cell is a
# `mutable struct <: JaxComponent`; since `JaxComponent <: AbstractComponent`,
# all NGCSimLib substrate machinery (`compartments`, `post_init!`, the Parser)
# works on them unchanged.
#
# Design decisions (see docs/decisions.md):
#   - #1 abstract-type-not-metaclass: `JaxComponent` is the dispatch root, not a
#     macro-injected mixin. Concrete cells declare the 4 standard component
#     fields explicitly (`name`, `context_path`, `args`, `kwargs`) because
#     `@ngc_component` hardcodes the supertype to `AbstractComponent` and would
#     erase the `JaxComponent` layer.
#   - #2 PRNG keys are `UInt64` seeds, not JAX `PRNGKey` arrays. JAX threads a
#     splittable key array; Julia's RNGs (`Xoshiro`) seed from an integer, so a
#     `UInt64` seed carried in a Compartment is the faithful, idiomatic analog.

"""
    JaxComponent <: NGCSimLib.AbstractComponent

Abstract base for every NGCLearn cell/synapse. Concrete subtypes own a PRNG
`key` compartment plus their state compartments, and implement
[`advance_state!`](@ref) and [`reset_state!`](@ref).
"""
abstract type JaxComponent <: NGCSimLib.AbstractComponent end

# Shared verbs — declared once here so each component file adds a method
# (multiple dispatch) rather than shadowing a fresh generic.

"""
    advance_state!(c::JaxComponent, args...)

Advance the component's dynamics one simulation step, mutating its state
compartments in place. Each concrete cell defines a `@compilable` method.
(Named with `!` per Julia convention; upstream spelling is `advance_state`.)
"""
function advance_state! end

"""
    reset_state!(c::JaxComponent)

Reset the component's compartments to their initial values. Each concrete cell
defines a `@compilable` method. (Upstream spelling is `reset`; renamed to avoid
implying a relationship to any `Base` verb and to carry the mutating `!`.)
"""
function reset_state! end

"""
    make_prng_key(seed::Union{Integer,Nothing}=nothing) -> UInt64

Produce a PRNG seed for a component's `key` compartment. With `nothing`, draws a
nondeterministic seed from the OS RNG (the analog of upstream's
`random.PRNGKey(time.time_ns())`); pass an explicit integer for reproducibility.
"""
function make_prng_key(seed::Union{Integer, Nothing}=nothing)
    return seed === nothing ? rand(RandomDevice(), UInt64) : UInt64(seed)
end
