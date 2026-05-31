# NGCLearn — Julia port of NACLab's ngc-learn (https://github.com/NACLab/ngc-learn).
#
# This is Layer 1 of the NGC stack:
#   Layer 0    NGCSimLib      substrate — Component / Compartment / Context / Process
#   Layer 1 (this)  NGCLearn   biophysical component zoo (neurons + synapses)
#   Layer 2    FabricPC.jl    predictive-coding graph framework
#
# Layer 1 depends on Layer 0 via a `[sources]` path dev-link to ../NGCSimLib
# (see Project.toml). See `docs/decisions.md` for cross-cutting design
# decisions and `docs/specs/*.md` for the per-component ports from the
# upstream Python.

module NGCLearn

using NGCSimLib
using Random

const NGCLEARN_VERSION = v"0.1.0"

# ── Utilities ──────────────────────────────────────────────────────────────────
# Differential-equation integration backend (Euler/RK1, midpoint/RK2, ...).
include("utils/diffeq/ode_utils.jl")

# ── Component base ───────────────────────────────────────────────────────────
# Abstract base for every Jax-derived cell/synapse. Mirrors upstream
# JaxComponent (a Component subclass that owns a PRNG `key` compartment).
include("components/jax_component.jl")

# ── Components ───────────────────────────────────────────────────────────────
# Spiking neurons
include("components/neurons/spiking/lif_cell.jl")
# Graded (rate-coded / error) neurons
include("components/neurons/graded/gaussian_error_cell.jl")

# ── Exports ──────────────────────────────────────────────────────────────────
export NGCLEARN_VERSION

# Integration backend
export get_integrator_code, step_euler, step_rk2

# Component base + shared verbs
export JaxComponent, advance_state!, reset_state!, make_prng_key

# Components
export LIFCell, GaussianErrorCell

end # module NGCLearn
