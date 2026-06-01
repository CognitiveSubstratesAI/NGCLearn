```@meta
CurrentModule = NGCLearn
```

# Discriminative Predictive Coding (Whittington & Bogacz; 2017)

In this exhibit, we will see how a classifier can be created based on
predictive coding. This exhibit model effectively reproduces some of the results
reported in (Whittington & Bogacz, 2017) **[1]**. NGCLearn ships this as the
[`PCN`](@ref) model (`src/models/pcn.jl`), a faithful Julia port of the
ngc-museum `pc_discrim` exhibit.

## The Predictive Coding Network (PCN)

The discriminative predictive coding network (PCN) is a hierarchical neuronal
model that seeks to predict the label `y` ($\mathbf{y}$; typically a one-hot
encoded label) of a given sensory input data point `x` ($\mathbf{x}$). The PCN
model of **[1]** showed that constructing a multi-layer system composed of
layers of stateful neural processing units, where one layer of neurons would
locally predict the activities of the ones (situated) above them, resulted in an
effective yet more biologically-plausible classifier, as compared to deep neural
networks trained with backpropagation of errors (or backprop). Furthermore,
**[1]** showed that the underlying dynamics of the PCN could be shown to recover
the updates to synaptic weights produced by backprop under certain
assumptions/conditions.

In NGCLearn, building the PCN requires working with the library's `graded`
neurons, i.e., those that follow dynamics without spikes/discrete action
potentials, specifically the [`RateCell`](@ref) and the
[`GaussianErrorCell`](@ref) components. In effect, to construct a two-hidden-layer
model of the form of **[1]**, one wires together four layers of `RateCell`'s,
i.e., `z0` ($\mathbf{z}^0$), `z1` ($\mathbf{z}^1$), `z2` ($\mathbf{z}^2$), `z3`
($\mathbf{z}^3$), with three [`HebbianSynapse`](@ref)s, i.e., `W1`
($\mathbf{W}^1$), `W2` ($\mathbf{W}^2$), `W3` ($\mathbf{W}^3$). Inference
conducted in accordance with predictive coding mechanics generally follows a
message-passing scheme where local prediction errors -- the error/mismatch
activities associated with each layer's guess of the activities of the one above
it -- are passed up and down via local feedback synaptic connections in order to
produce updates to the neuronal activities themselves. Notice that this is much
akin to the E-step of expectation-maximization (E-M) **[2]**. This
message-passing is done typically for several steps in time and, after these
dynamics are iteratively run several times, the synaptic weight updates are
computed for each layer's associated predictive synapses (`W1`, `W2`, and `W3`)
using two-factor Hebbian rules.

Formally, the above means that each layer of neuronal dynamics can be
characterized by the following single ordinary differential equation (ODE):

```math
\tau_m \frac{\partial \mathbf{z}^\ell_t}{\partial t} = (-\gamma \mathbf{z}^\ell_t + \mathbf{h}^\ell_t), \; \text{where} \;
\mathbf{h}^\ell_t = -\mathbf{e}^\ell_t + (\mathbf{W}^{\ell+1})^T \cdot \mathbf{e}^{\ell+1}_t
```

where $()^T$ denotes the matrix transpose and $\cdot$ denotes matrix/vector
multiplication. Since there is no layer above it and it will only ever be clamped
to the label $\mathbf{y}$ (during training), $\mathbf{z}^3_t$'s dynamics
ultimately reduce to $\mathbf{z}^3_t = \mathbf{y}$, i.e., this means we set the
target output activity to be equal to the label. Furthermore, the bottom layer
$\mathbf{z}^0_t$ will always be clamped to the sensory input, e.g., a pixel
image, so it also reduces to $\mathbf{z}^0 = \mathbf{x}$ (this means that we set
the input layer to be equal to the data).[^1] These two simplifications mean we
only need to run the dynamics for neuronal layers $\mathbf{z}^1$ and
$\mathbf{z}^2$ via Euler integration (which is what the [`RateCell`](@ref) does
by default).

The only other key aspect to define in the above ODE is what $\mathbf{e}^\ell_t$
means. This is known as an "error cell", which is another type of graded neuron
component ([`GaussianErrorCell`](@ref)) and is, for the purposes of constructing
the model of **[1]**, defined as:

```math
\mathbf{e}^\ell_t = (\mathbf{z}^\ell_t - \mu^\ell_t), \; \text{where} \;
\mu^\ell_t = \mathbf{W}^\ell \cdot \phi^{\ell-1}(\mathbf{z}^{\ell-1})
```

given that it is the first derivative (with respect to the prediction mean
$\mu^\ell_t$) of a Gaussian log likelihood functional $\mathcal{F}^\ell$, with
fixed unit variance, applied locally to measure the discrepancy between the
actual neural activity at layer $\ell$ -- $\mathbf{z}^\ell_t$ -- and the
predicted neural activity $\mu^\ell_t$. You will notice that the equation
$\mu^\ell_t = \mathbf{W}^\ell \cdot \phi^{\ell-1}(\mathbf{z}^{\ell-1})$ depicts
how a local prediction is made from layer $\ell-1$ about the activity values in
layer $\ell$, where $\phi^{\ell-1}$ is an elementwise activation function applied
to the neuronal activity values.[^2]

The next important part in characterizing a PCN is its synaptic dynamics, which,
after running the above equations for several steps in time, amounts to a single
application of the following ODE (for each weight matrix `W1`, `W2`, and `W3`):

```math
\tau_w \frac{\partial \mathbf{W}^\ell_{t_j}}{\partial {t_j}} =
-\lambda \mathbf{W}^\ell_{t_j} + \mathbf{e}^\ell \cdot (\phi^{\ell-1}(\mathbf{z}^{\ell-1}))^T
```

where we use $t_j$ to indicate that the time-scale of the updates for any
synaptic weight matrix $\mathbf{W}^\ell$ is slower than those of the neuronal
activities described earlier.

The last part for constructing an effective PCN for classification is simply
determining how the initial conditions are set for the neurons in layers
$\ell = 1$ and $\ell = 2$. Much as in **[1]**, this is done by clamping the input
layer to sensory data, i.e., $\mathbf{z}^0 = \mathbf{x}$, clamping the output
layer to label data $\mathbf{z}^3 = \mathbf{y}$, and initializing $\mathbf{z}^1$
and $\mathbf{z}^2$ to values that are produced via a feedforward pass/sweep (or
"ancestral projection pass") from the input layer to output layer via:

```math
\begin{aligned}
\mu^1_t &= \mathbf{W}^1 \cdot \mathbf{z}^0, \; \mathbf{z}^1 = \phi^1(\mu^1_t) \\
\mu^2_t &= \mathbf{W}^2 \cdot \mathbf{z}^1, \; \mathbf{z}^2 = \phi^2(\mu^2_t) \\
\mu^3_t &= \mathbf{W}^3 \cdot \mathbf{z}^2
\end{aligned}
```

which also gets us our initial local predictions for free (same goes for the
error values, which are, at initialization, $\mathbf{e}^1 = 0$, $\mathbf{e}^2 =
0$, and $\mathbf{e}^3 = \mathbf{z}^3_t - \mu^3_t$). (Note: the above equations are
simply run at test-time when a label is not present to obtain a fast prediction
of input samples in the test-set and to measure model generalization ability.)

### How NGCLearn realizes the two sub-networks

In NGCLearn, [`PCN`](@ref) holds **two coupled sub-networks** built inside a
single `NGCSimLib.Context` (mirroring the upstream exhibit's `pcn_model.py`):

- **Generative / forward network** — the four `RateCell`s `z0..z3`, three
  `GaussianErrorCell`s `e1..e3`, the three Hebbian-adapted predictive synapses
  `W1..W3`, and two **static** feedback synapses `E2`, `E3`
  ([`DenseSynapse`](@ref)). The forward path computes the predictions
  $\mu^\ell$, and the feedback path carries the error signals $(\mathbf{W}^{\ell+1})^T
  \cdot \mathbf{e}^{\ell+1}$ back down to drive the latent ODEs.
- **Inference / projection network** — a parallel feed-forward stack `q0..q3`
  wired through three static [`DenseSynapse`](@ref)s `Q1..Q3` (plus an output
  error cell `eq3`). This implements the ancestral projection pass above; its
  outputs are used **only to initialise** the generative latents (`z1.z = q1.z`,
  `z2.z = q2.z`) at the start of each settling cycle, and to read out the fast
  prediction $\mu^3_t$ at test time.

Before each cycle the inference weights are tied to the forward weights,
$\mathbf{Q}^\ell = \mathbf{W}^\ell$, and the feedback weights are set to the
transpose, $\mathbf{E}^\ell = (\mathbf{W}^\ell)^T$, so that the projection graph
and feedback path always reflect the current generative model.

## The PEM cycle (Projection–Expectation–Maximization)

[`process!`](@ref) runs the full "PEM" cycle on a single `(obs, lab)` pair, which
realizes the message-passing E-M dynamics described above as three ordered
phases:

1. **Projection (P-step)** — reset all cells, tie $\mathbf{Q}=\mathbf{W}$ and
   $\mathbf{E}=\mathbf{W}^T$, clamp the input ($\mathbf{z}^0 = \mathbf{x}$),
   feed-forward project through the `q`-network, and seed the generative latents
   $\mathbf{z}^1, \mathbf{z}^2$ from the projected inference states (and the
   output error $\mathbf{e}^3$ from `eq3`).
2. **Expectation (E-step)** — run `T` rounds of error-driven latent settling
   (the neuronal ODE above), clamping $\mathbf{z}^0 = \mathbf{x}$ and
   $\mathbf{z}^3 = \mathbf{y}$ each round.
3. **Maximization (M-step)** — when adapting, apply one Hebbian update (with an
   Adam adaptive learning rate) to each predictive synapse `W1`, `W2`, `W3` using
   the synaptic ODE above.

This is much like the E-step / M-step alternation of expectation-maximization
**[2]**: the E-step infers the latent activities that best explain the data under
the current weights, and the M-step adjusts the weights to reduce the residual
prediction error.

### The free-energy objective

The entire PCN system can be shown to be optimizing a single global objective
function, the (expected) free energy

```math
\mathcal{F} = \sum_{\ell=1}^{L} \mathcal{F}^\ell
```

where the first derivative of $\mathcal{F}^\ell$ is exactly what gave us the error
cell equation for layer $\ell$. NGCLearn surfaces an approximation of this
quantity, `EFE`, as the third return value of [`process!`](@ref), computed as the
sum of the per-layer error-cell losses $\mathcal{F} \approx \mathcal{L}^1 +
\mathcal{L}^2 + \mathcal{L}^3$. As learning proceeds this quantity is driven
toward zero, confirming the PCN is minimizing the full free energy
$\mathcal{F}$.

## Running the PCN model

The most central configuration/hyper-parameter values are set through the model
constructor. The integration time constant for the Euler integration of the
neuronal dynamics is `dt` (in milliseconds), the number of E-steps is `T`, and
the Hebbian adjustments use an Adam adaptive learning rate with global learning
rate `eta`. The PRNG is seeded by a `UInt64`-style `key`:

```julia
using NGCLearn

m = PCN(; in_dim=4, out_dim=2, hid1_dim=8, hid2_dim=6,
        T=10, dt=1.0, eta=0.002, key=7)
```

To run one training step (a full PEM cycle), pass a single observation row and
its one-hot label row to [`process!`](@ref). It returns the projected prediction
`y_inf` (the fast $\mathbf{z}^3 = \mathbf{q}^3$ from the P-step), the settled
prediction `y_mu` (the post-settling $\mu^3_t = $ `e3.mu`), and the approximate
free energy `EFE`:

```julia
x = reshape(Float64[1, 1, 0, 0], 1, 4)   # one observation (1×in_dim)
y = reshape(Float64[1, 0], 1, 2)         # one one-hot label (1×out_dim)

y_inf, y_mu, EFE = process!(m, x, y)      # one PEM training step (adapt=true)
```

Setting `adapt=false` turns the learning dynamics off and only reads out the
output prediction (the label is not used). For pure test-time inference, use
[`project`](@ref), which runs only the ancestral projection pass (no settling, no
learning) and returns $\mathbf{q}^3$:

```julia
pred = project(m, x)                      # fast p(y|x) read-out
```

On a small deterministic classification task, training over many epochs drives
the mean output error from roughly `0.7` down to below `0.001` and classifies the
training set `4/4`. This corroborates the spirit of the much larger MNIST
experiment in (Whittington & Bogacz, 2017) **[1]**, where less than `2%`
validation error was reported. See `examples/02_pc_discrim_train.jl` for the full
training loop.

Note that the accuracy/read-out used at deployment comes from the ancestral
projection graph (the initialization step of the settling process), meaning the
projection graph itself can be deployed as a probabilistic neuronal model of
`p(y|x)` — which is exactly what [`project`](@ref) exposes.

## API

```@docs
PCN
process!
project
```

## References

**[1]** Whittington, James C.R., and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." *Neural Computation* 29.5 (2017): 1229-1262.

**[2]** Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum
likelihood from incomplete data via the EM algorithm." *Journal of the Royal
Statistical Society: Series B (Methodological)* 39.1 (1977): 1-22.

[^1]: The integration time constant `tau_m` for both `z0` and `z3` is set to zero
    -- a shortcut that tells those nodes to run "stateless dynamics" (a simple
    forwarding operation that does not require integrating an ODE), since `z0` is
    always clamped to the input and `z3` to the label.
[^2]: The bottom/input layer `z0`'s activation $\phi^0$ and the top layer `z3`'s
    activation $\phi^3$ are both set to the identity function, as in **[1]**.
