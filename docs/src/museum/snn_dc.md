```@meta
CurrentModule = NGCLearn
```

# The Diehl and Cook Spiking Neuronal Network

In this exhibit, we will see how a spiking neural network model that adapts its
synaptic efficacies via spike-timing-dependent plasticity can be created. This
exhibit model effectively reproduces some of the results reported in Diehl &
Cook (2015) **[1]**. It is a faithful Julia port of the `diehl_cook_snn`
exhibit from NACLab's `ngc-museum`, built on the NGCLearn components and driven
by the [`DC_SNN`](@ref) model.

## The Diehl and Cook Spiking Network (DC-SNN)

The Diehl and Cook spiking neural network **[1]** (which we abbreviate to
"DC-SNN") is an important model of spiking neuronal dynamics that crucially
demonstrated effective unsupervised learning on MNIST digit patterns.
Furthermore, it is a useful model for getting a handle on the interaction
between explicit excitatory neurons and inhibitory neurons (where cells in both
populations/groups are leaky integrators). In NGCLearn, constructing a DC-SNN is
straightforward using its in-built leaky integrator, the [`LIFCell`](@ref),
which is what the [`DC_SNN`](@ref) model constructor does for you. The DC-SNN
model in this exhibit also makes use of NGCLearn's in-built trace-based
spike-timing-dependent plasticity synapse, the [`TraceSTDPSynapse`](@ref), in
order to recover one of the ways that **[1]** adapted its synaptic strengths.

### Neuronal Dynamics

The DC-SNN is effectively made up of three layers:

1. a sensory input layer made up of [`PoissonCell`](@ref) encoding neuronal
   cells (which are configured in the exhibit model to fire spikes at a maximal
   frequency of `63.75` Hertz as in **[1]**);
2. one hidden layer of excitatory leaky integrate-and-fire (LIF) cells; and,
3. one (laterally-wired) hidden layer of inhibitory LIF cells.

The sensory input layer connects to the excitatory layer with a synaptic cable
`W1` ($\mathbf{W}^1_t$) — which will be adapted with (trace-based)
spike-timing-dependent plasticity as we will describe in the next section. The
excitatory layer of the model connects to the inhibitory layer with a fixed
identity ("eye") synaptic cable (ensuring that all excitatory neurons map
one-to-one to the inhibitory ones) while the inhibitory layer is connected back
to the excitatory layer via a fixed, negatively scaled hollow synaptic cable.
In the [`DC_SNN`](@ref) port these two fixed cables are [`DenseSynapse`](@ref)
components initialized with `weight_init=("constant", 22.5, :eye)` and
`weight_init=("constant", -120.0, :hollow)` respectively, and are never adapted.

The dynamics on the structure described above result in a form of sensory
input-driven excitatory LIF dynamics that are recurrently inhibited by the
laterally-wired inhibitory LIF cells. Formally, the excitatory and inhibitory
neurons in the DC-SNN exhibit both adhere to the following ordinary differential
equation (ODE):

```math
\tau_m \frac{\partial \mathbf{v}_t}{\partial t} = (v_{rest} - \mathbf{v}_t + R \mathbf{j}_t) \odot \mathbf{m}_{rfr}
```

where $v_{rest}$ is the resting potential (in milliVolts, mV) and
$\mathbf{m}_{rfr}$ is a binary mask where any element is zero if the neuronal
cell is in its refractory period (meaning that it will be clamped to its
resting/reset potential for so many milliseconds). Note that $\odot$ denotes an
elementwise multiplication (Hadamard product) and $\cdot$ denotes a
matrix-vector multiplication. Spikes are emitted from any cell within
$\mathbf{v}_t$ according to the following piecewise function:

```math
\mathbf{s}_{t, i} = \begin{cases}
                       1 & \mathbf{v}_{t, i} > \theta_{t, i} \\
                       0 & \mbox{otherwise.}
                    \end{cases}
```

and any neuron that breached its threshold value and emitted a (binary) spike is
immediately set to its $v_{reset}$ potential (mV); note that $v_{reset}$ might
not be the same as $v_{rest}$. $\theta_t$ contains all of the current voltage
threshold values (and is the same shape as $\mathbf{v}_t$). In NGCLearn, the
[`LIFCell`](@ref) component also evolves $\theta_t$ according to its own set of
dynamics as follows:

```math
\begin{aligned}
\tau_\delta \frac{\partial \delta}{\partial t} &= -\delta + \alpha \mathbf{s}_t \\
\theta_t &= \theta_{base} + \delta
\end{aligned}
```

where $\delta$ is a "homeostatic variable" that essentially increments (any
dimension $i$) by a small constant amount every time a particular cell $i$ emits
a spike (setting $\tau_\delta = 0$ turns off the threshold dynamics). Note that
the second equation above implies that $\theta_t$ does not evolve its base
threshold value ($\theta_{base}$); it is simply re-computed as a sum of its base
value and the current value of the evolved homeostatic variable. With the above
knowledge, we can now effectively recreate the setup of **[1]** by using a value
greater than zero for $\tau_\delta$ (the `tau_theta` keyword) for the excitatory
LIFs and a value of zero for the inhibitory ones. In the [`DC_SNN`](@ref) port
this is exactly the construction: the excitatory `z1e` cell uses
`tau_theta=1e7`, `theta_plus=0.05` (so its threshold slowly adapts), while the
inhibitory `z1i` cell uses `tau_theta=0.0` (threshold dynamics off).

Finally, the last key item to specify is how the electrical current is produced
along synapses for the excitatory and inhibitory neurons. For the excitatory
LIFs, the electrical currents can be produced either as their own evolving ODE
(as in **[1]**) or simplified to pointwise currents. The [`DC_SNN`](@ref) port
opts for the latter for simplicity and additional simulation speed, entailing the
following (synaptic) wiring scheme:

```math
\begin{aligned}
\mathbf{j}^e_t &= \mathbf{W}^1_t \cdot \mathbf{s}^{inp}_t - \big((1 - \mathbf{I}) \odot \mathbf{W}^{ie}\big) \cdot \mathbf{s}^i_t \\
\mathbf{j}^i_t &= \big(\mathbf{I} \odot \mathbf{W}^{ei}\big) \cdot \mathbf{s}^e_t
\end{aligned}
```

where we denote an excitatory spike vector as $\mathbf{s}^e_t$, an inhibitory
spike vector as $\mathbf{s}^i_t$, and an input (Poisson) spike vector as
$\mathbf{s}^{inp}_t$. As mentioned earlier, $\mathbf{W}^1_t$ is the
input-to-excitatory synaptic cable while $\mathbf{W}^{ei}$ is the
excitatory-to-inhibitory synaptic cable and $\mathbf{W}^{ie}$ is the
inhibitory-to-excitatory synaptic cable; the subscript $t$ has been dropped for
these last two cables because they are held fixed to constant values **[1]**.
Ultimately, the inhibitory neurons will emit spikes once enough voltage has been
built up as they receive enough electrical current driven by the excitatory
neurons — once the inhibitory neurons fire, they will laterally
inhibit/suppress the activities of other excitatory cells (besides the one they
receive electrical current directly from). This type/pattern of cross-layer
inhibition induces a form of "lateral competition" in the dynamics of the
excitatory neural units, where few units specialize to represent particular
types/kinds of input patterns (these competitive dynamics are important for
effective adaptation of synapses via STDP).

In the [`DC_SNN`](@ref) port, this wiring is expressed directly as compartment
connections inside the model's `NGCSimLib.Context`:

```julia
z0.outputs >> W1.inputs                            # Poisson input → feedforward W1
z1i.s      >> W1ie.inputs                           # inhibitory spikes → lateral cable
NGCSimLib.Summation(W1.outputs, W1ie.outputs) >> z1e.j  # feedforward + lateral inhibition
z1e.s      >> W1ei.inputs                           # excitatory spikes → eye cable
W1ei.outputs >> z1i.j                               # → inhibitory current
```

The spiking neural system that the above specifies will engage in a form of
unsupervised representation learning, simply resulting in sparse spike-train
patterns that correlate with different input digit patterns. The case worth
picturing is the one where only one out of the population of excitatory neuronal
cells gets triggered and drives its corresponding inhibitory cell, which
transmits back/laterally a suppression signal to all but the triggered
excitatory neuron.

All that remains is to specify the synaptic plasticity dynamics (learning) for
the DC-SNN, which we do next.

### Spike-Timing-Dependent Plasticity (STDP)

Adaptation/plasticity in our DC-SNN is rather simple — in this model exhibit,
a form of STDP is used to adapt the synaptic weight values (specifically, it is
trace-based STDP, provided by the [`TraceSTDPSynapse`](@ref)), which models a
form of long-term depression (LTD) and long-term potentiation (LTP) to adjust
synapses according to a Hebbian-like rule as follows (in matrix-vector format):

```math
\tau_{syn} \frac{\partial \mathbf{W}^1_t}{\partial t} =
A_{+} \Big(\mathbf{s}^e_t \cdot (\mathbf{z}^{inp}_t - z_{tar})^T\Big)
- A_{-} \Big(\mathbf{z}^e_t \cdot (\mathbf{s}^{inp}_t)^T\Big)
```

where we see that LTP is modeled by the first term (to the left of the minus
sign) — a product of the (excitatory) post-synaptic spike(s)/event(s) that
might have occurred at time $t$ and the value of the pre-synaptic trace at the
occurrence of spikes at $t$ — and that LTD is modeled by the second term (to
the right of the minus sign) — a product of the (excitatory) post-synaptic
trace at time $t$ and the pre-synaptic spike(s)/event(s) that might have happened
at $t$. $A_{+}$ is the scaling factor controlling how much LTP is applied to a
synaptic update and $A_{-}$ controls how LTD would be applied to the update.
$()^T$ denotes the matrix/vector transpose while $z_{tar}$ is a pre-synaptic
trace target value, set to $0$ in the default configuration of the DC-SNN (in
**[1]**, this value was used in one of the variants of STDP investigated; if it
is set to be $> 0$, then rarely-spiking neurons in the input layer will be
effectively disconnected from the excitatory layer, fading away to zero).

Roughly speaking, the above STDP rule effectively applies the idea that, for a
pre-synaptic neuron $i$ in the input layer and a post-synaptic neuron $j$ in the
excitatory layer (in our DC-SNN model), we increase the efficacy value of the
synapse $W^1_{ij}$ (that connects them) if neuron $j$ spikes after neuron $i$,
and we decrease the efficacy value $W^1_{ij}$ if neuron $i$ spikes before neuron
$j$.

The pre- and post-synaptic traces ($\mathbf{z}^{inp}_t$ and $\mathbf{z}^e_t$)
are produced by [`VarTrace`](@ref) components, which iteratively apply a low-pass
filter to the spike trains. In the [`DC_SNN`](@ref) port the relevant
hyperparameters match **[1]**: `A_plus=1e-2`, `A_minus=1e-4`,
`pretrace_target=0.0`, and trace time constant `tau_tr=20.0`. The traces and
spike events are wired into `W1`'s plasticity compartments as follows:

```julia
z0.outputs >> tr0.inputs        # pre-synaptic trace over Poisson spikes
z1e.s      >> tr1.inputs        # post-synaptic trace over excitatory spikes
tr0.trace  >> W1.preTrace
z0.outputs >> W1.preSpike
tr1.trace  >> W1.postTrace
z1e.s      >> W1.postSpike
```

#### Weight normalization (homeostasis)

Beyond the trace-based STDP rule, there is one further small mechanism applied to
this model — a synaptic rescaling step, applied as was done in **[1]**. After
each stimulus window, each column of the feedforward weight matrix `W1` is
rescaled to a fixed L1-norm ($78.4$ by default), keeping the total incoming
weight per excitatory unit bounded. This is the homeostatic constraint of Diehl
& Cook (2015) and is exposed in the port as [`norm!`](@ref), which
[`process!`](@ref) calls automatically at the end of an adapting window.

## Running the DC-SNN Model

The model is driven entirely through its Julia API. Build a DC-SNN with the
[`DC_SNN`](@ref) keyword constructor, then drive it one observation at a time
with [`process!`](@ref) (online, batch size 1):

```julia
using NGCLearn

m = DC_SNN(; in_dim=64, hid_dim=10, T=200, key=3)
obs = fill(0.9, 1, 64)                 # one stimulus pattern (1×in_dim row)

counts = process!(m, obs; adapt=true)  # 1×hid_dim excitatory spike counts
```

[`process!`](@ref) runs a `T`-step stimulus window: reset all components, clamp
`obs` into the Poisson encoder, advance every component each step (running the
STDP `evolve!` when `adapt=true`), and finally L1-normalize `W1` via
[`norm!`](@ref). It returns the excitatory spike counts (`1×hid_dim`)
accumulated over the window. To evaluate without learning, pass `adapt=false`
(no STDP `evolve!` and no normalization step).

Because STDP-adaptation of synapses is unsupervised, there is no objective
function or cost functional to track. In the NAC group's experience, observing
the mean and Frobenius norm of synaptic values (the entries of `W1`) is a useful
starting point for determining unhealthy or degenerate behavior in spiking
credit assignment.

### Verified winner-take-all behavior

We verified the defining behavior of this model directly: under drive, the fixed
lateral inhibition (`W1ei` eye, `W1ie` hollow-negative) produces the expected
**winner-take-all** competition in the excitatory layer — one excitatory unit
dominates the accumulated spike counts returned by [`process!`](@ref) while the
remaining units are suppressed. This sparse, competitive activity is precisely
what makes the trace-STDP rule specialize different excitatory units to
different input patterns. (The excitatory [`LIFCell`](@ref) is built with
`one_spike=true`, sharpening this competition.)

Note that the DC-SNN in this exhibit's default configuration is small and
intended as a faithful, runnable port rather than a full reproduction of the
MNIST results in **[1]**; larger hidden layers and more training patterns would
be needed to recover the full set of digit-template receptive fields reported
there.

## API

```@docs
DC_SNN
norm!
```

See also [`process!`](@ref) for the simulation driver, and the components it is
built from: [`PoissonCell`](@ref), [`LIFCell`](@ref), [`TraceSTDPSynapse`](@ref),
[`DenseSynapse`](@ref), and [`VarTrace`](@ref).

## References

**[1]** Diehl, Peter U., and Matthew Cook. "Unsupervised learning of digit
recognition using spike-timing-dependent plasticity." *Frontiers in
Computational Neuroscience* 9 (2015): 99.
