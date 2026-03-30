---
title: 'Physics Informed Neural Networks'
description: 'Learning the heat equation from sparse data'
tags: ["machine learning", "physics", "PINNs", "PyTorch"]
date: '2026-03-28T19:50:27-07:00'
draft: true
math: true
---

## Introduction
Neural networks have an extraordinary ability to learn highly non-linear
solutions from data. However, the amount of data needed in order for a neural
net to learn a good, generalizable solution can be vast. Modern LLMs, for
example, are pre-trained on over [15 million
tokens](https://huggingface.co/meta-llama/Llama-3.1-8B#training-data). When such
vast datasets are out of reach, there are still some options. In some
situations, synthetic data generation can fill the gap. Models can also use
architectures that bake in known properties of the problem - an image detection
system should identify a given object regardless of where on the image it
appears, a property known as translational invariance. Convolutional neural nets
have exactly this property. But when our problem doesn't lend itself to
synthetic data generation or constraints that can easily be reflected in the
model, all hope is not lost.

Physics-Informed Neural Networks (PINNs) encode known constraints about the 
system being modeled directly into the training objective as an additional 
penalty term. This penalty can be thought of as a form of regularization that 
forces the model to be consistent with properties we know the solution must 
satisfy — whether derived from first principles or empirical knowledge. In the 
context of physics problems, those constraints generally come in the form of 
partial differential equations (PDEs). In these cases, modern ML frameworks like
PyTorch and JAX handle the penalty term quite elegantly due to their
autodifferentiation engines. No need for finite difference approximations or
symbolic math! In this post, I'll focus on PDEs, following the [original
paper](https://arxiv.org/abs/1711.10561), but the same ideas apply anywhere you
have reliable prior knowledge about the system.

The 1D heat equation makes for a natural first example — it has a clean 
analytical solution we can use to verify our results, and its form is similar 
to the non-linear Burgers' equation used in the original Raissi et al. paper. Beyond just 
showing a working implementation, I'll also explore how each term in the loss
works together to generate a full solution through ablation studies. By the end,
you should have a pretty clear sense of exactly how PINNs work and what
classes of problems you might reach for a PINN to solve.


## The Heat Equation
The 1D heat equation describes how heat diffuses through a medium over time:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

where $u(x, t)$ is the temperature at position $x$ and time $t$, and $\alpha >
0$ is the thermal diffusivity of the medium.

The analytical solution can be found by assuming the solution separates into a
product of a function of $x$ alone and a function of $t$ alone — an approach
known as separation of variables. This yields a general solution

$$
\begin{equation}
\label{eq:general}
u(x, t) = A_0 e^{Ct}\left(B_0 e^{\beta x} + B_1 e^{-\beta x}\right)
\end{equation}
$$

where $\beta = \sqrt{C/\alpha}$ and $A_0$, $B_0$, $B_1$, and $C$ are free
parameters. 

We consider the problem on the domain $x \in [0, 1]$, $t \in [0, T]$, with the
following boundary and initial conditions:

$$
u(0, t) = 0, \quad u(1, t) = 0, \quad u(x, 0) = \sin(\pi x)
$$

The boundary conditions fix the temperature at both ends of the rod to zero for
all time, and the initial condition describes a sinusoidal temperature
distribution along the rod at $t = 0$. Enforcing our boundary and initial
conditions fixes the free parameters uniquely, giving the exact solution:

$$
\begin{equation}
\label{eq:specific}
u(x, t) = e^{-\alpha \pi^2 t} \sin(\pi x)
\end{equation}
$$

The spatial profile remains sinusoidal for all time, decaying exponentially as
heat dissipates. The full derivation is included below for the curious.

## The PINN Approach

### Network Architecture
The network itself is a straightforward fully connected MLP that takes the
spatial coordinate $x$ and time $t$ as inputs and outputs the predicted
temperature $u(x, t)$:

$$
\mathcal{N}: (t, x) \mapsto u(x, t)
$$

The default architecture uses 4 hidden layers with 32 neurons each and Tanh
activations throughout. Tanh is a natural choice here — unlike ReLU, it is
infinitely differentiable, which matters because we will need to compute
second-order derivatives of the network output during training.
```python
class PINN(nn.Module):
    def __init__(self, n_neurons, n_layers):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, x):
        input = torch.cat([t, x], dim=1)
        return self.mlp(input)
```

There is nothing unusual about this architecture — it is a standard MLP. The
physics enters not through the model itself, but through the loss function.

### Loss Function

The total loss has two components — a data loss and a physics loss:

<div>
\begin{equation}
\mathcal{L} = \lambda_u \mathcal{L}_{data} + \lambda_f \mathcal{L}_{physics}
\label{eq:loss}
\end{equation}
</div>

The weights $\lambda_u$ and $\lambda_f$ control the
relative contribution of each term and default to 1.0.

The data loss is a standard MSE over the boundary and initial condition points,
locations where we do know the value of $u$:

<div>
\begin{equation}
\mathcal{L}_{data} = \frac{1}{N_u}\sum_{i=1}^{N_u}\left(\mathcal{N}(t^i, x^i) - u^i\right)^2
\label{eq:loss_data}
\end{equation}
</div>


The physics loss penalizes the network for violating the heat equation at a set
of __collocation points__ — locations in the domain where we don't know $u$, but
we do know the governing equation must hold:

<div>
\begin{equation}
\mathcal{L}_{physics} = \frac{1}{N_f}\sum_{i=1}^{N_f}
\left(
    \frac{\partial \mathcal{N}}{\partial t}\bigg|_{(t^i, x^i)}
    - \alpha \frac{\partial^2 \mathcal{N}}{\partial x^2}\bigg|_{(t^i, x^i)}
\right)^2
\label{eq:loss_physics}
\end{equation}
</div>

In other words, $\mathcal{L}_{physics}$ is the mean squared PDE residual — it is
zero only when the network output exactly satisfies the heat equation at every
collocation point. What makes this particularly elegant in PyTorch is that the
derivatives in $\mathcal{L}_{physics}$ are computed via autodifferentiation
directly through the network:
```python
def physics_informed_nn(model, t, x, alpha=ALPHA):
    out = model.forward(t, x)
    u_x = torch.autograd.grad(
        out, x, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_t = torch.autograd.grad(
        out, t, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]

    return u_t - alpha * u_xx
```

`torch.autograd.grad` is less common than a simple `.backward()` call and the
[documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
can be a little opaque, so it's worth a closer look. A call
`torch.autograd.grad(output, input, grad_outputs=v)` computes the
vector-Jacobian product $v^T J$, where $J$ is the Jacobian of `output` with
respect to `input`. In our case the network output for a batch of size $n$ is
a tensor of shape `(n, 1)`, so the Jacobian of $\mathcal{N}$ with respect to
$x$ is a column vector whose $i$-th entry is
$\partial \mathcal{N}(t^i, x^i) / \partial x^i$. By passing
`grad_outputs=torch.ones_like(out)` we set $v = \mathbf{1}$, which simply
sums the Jacobian entries — giving us the elementwise gradients across the
batch.

The `create_graph=True` argument tells PyTorch to keep the computational graph
alive through the derivative computation so that gradients can be
backpropagated through $\mathcal{L}_{physics}$ during training. Without it,
PyTorch discards the graph after the computation for efficiency, and the second
call to `torch.autograd.grad` for $u_{xx}$ would fail.

### Training Setup

The model is trained using Adam with a StepLR learning rate scheduler that
decays the learning rate by a factor of 0.5 every 2000 epochs. The default
training run uses 5000 epochs with an initial learning rate of $10^{-3}$.

The boundary and initial condition points are sampled once before training and
held fixed throughout. The collocation points, however, are resampled randomly
at every epoch. This means the physics loss is evaluated at a different set of
locations each epoch, which acts as a form of regularization and helps the
network learn to satisfy the heat equation across the full domain rather than
just at a fixed set of points.

The default sample sizes are:
- $N_u = 200$ data points (100 boundary + 100 initial condition)
- $N_f = 5000$ collocation points

The relatively large number of collocation points compared to data points
reflects the nature of the problem — the physics loss needs to cover the full
two-dimensional domain $(x, t) \in [0,1] \times [0,1]$, while the data loss
only needs to cover the boundary and initial condition.

## Results

With both loss terms active, the PINN recovers the analytical solution well.
The figure below shows the predicted and exact solutions as heatmaps over the
full domain, along with the absolute error. The total L2 error is $8.46 \times
10^{-3}$.

![PINN results](images/full_results.png)

The bottom row shows five time snapshots with the predicted and exact solutions
overlaid. The network captures both the sinusoidal spatial profile and the
exponential decay in time accurately across the full domain, pretty cool!


## Ablation Studies

### Physics Loss Only

Setting $\lambda_u = 0$ removes the boundary and initial condition constraints
from the training objective entirely, leaving the network free to find any
solution that satisfies the heat equation. The loss curves tell the story
clearly — the physics loss drives down to nearly $10^{-10}$ over 5000 epochs,
while the data loss, which is not being optimized, flatlines around $0.15$
throughout.

![Physics only loss](images/loss_no_data.png)

The solution the network finds, however, bears no resemblance to the one we
are looking for.

![Physics only results](images/results_no_data.png)

Rather than the decaying sinusoidal profile of the exact solution, the network
has learned something nearly flat and slightly negative across the entire
domain. The time slices show a solution that grows slowly in time and increases
linearly with $x$ — the opposite behavior to what we expect physically.

To understand what the network actually learned, recall the general solution to
the heat equation before enforcing boundary conditions:

$$
u(x, t) = A_0 e^{Ct}\left(B_0 e^{\beta x} + B_1 e^{-\beta x}\right)
$$

Fitting this form to the network output at five time snapshots gives consistent
results across all slices, confirming that the network did indeed learn a valid
solution to the PDE. The fit parameters are summarized below:

| Parameter | Mean | Std |
|-----------|------|-----|
| $A_0$ | $0.2125$ | $0.0187$ |
| $B_0$ | $-0.2231$ | $0.0158$ |
| $B_1$ | $-0.2905$ | $0.0348$ |
| $C$ | $0.0026$ | $0.0014$ |

Note that some variation in the fit parameters is expected — it is difficult to
disentangle noise introduced by the curve fitting algorithm itself from genuine
variation in the learned solution across time slices.

With both $B_0$ and $B_1$ negative and of similar magnitude, the spatial part
is approximately proportional to $-\cosh(\beta x)$, giving the downward
solution visible in the heatmap. The value of $C$ is small and positive, meaning
the solution grows slowly in time rather than decaying — the opposite of the
physical behavior. With $\alpha = 0.5$, this gives $\beta = \sqrt{C / \alpha}
\approx 0.072$, which is small enough that $\cosh(\beta x)$ is nearly
indistinguishable from a linear function over $x \in [0, 1]$. The fact that
$|B_1| > |B_0|$ in all cases tips the balance slightly toward $e^{-\beta x}$,
producing the gentle linear increase with $x$ visible in the time slice plot.

This is a valid solution to the heat equation — the physics loss confirms it —
but it is completely unphysical. Without the data loss term to enforce the
boundary and initial conditions, the network found its own preferred solution
from the infinitely many that satisfy the PDE.

### Data Loss Only

Setting $\lambda_f = 0$ removes the physics constraint entirely, leaving the
network to learn solely from the boundary and initial condition data points.
The loss curves show the data loss driving down to around $10^{-5}$ over 5000
epochs, punctuated by a series of sharp spikes whose origin is not entirely
clear. The physics loss, which is not being optimized, shoots up early and
flatlines around $7$ — a reminder that satisfying the boundary conditions alone
says nothing about whether the heat equation is being satisfied in the interior.

![Data only loss](images/loss_no_physics.png)

The results plot tells a clear story. At $t = 0$ the network matches the exact
solution almost perfectly — it has direct training data for the initial
condition $u(0, x) = \sin(\pi x)$ and learns it well. But without any physics
to guide the interior, the network fails to learn that the solution should decay
exponentially in time. Instead of the amplitude collapsing toward zero, the
PINN prediction barely decays at all, with an L2 error of $2.01$.

![Data only results](images/results_no_physics.png)

To understand what the network learned in the interior, we can fit the exact
solution form $u(x, t) = a e^{-ct} \sin(\beta x)$ to the PINN output at five
time snapshots:

| Time slice | $a$ | $c$ |
|------------|-----|-----|
| $t = 0.00$ | $1.00$ | $4.94$ |
| $t = 0.25$ | $3.00$ | $4.90$ |
| $t = 0.50$ | $8.87$ | $4.89$ |
| $t = 0.75$ | $25.6$ | $4.86$ |
| $t = 1.00$ | $71.4$ | $4.83$ |

The decay constant $c$ is remarkably consistent across all time slices and
close to the true value of $\alpha \pi^2 = 0.5 \times \pi^2 \approx 4.93$,
confirming that the spatial structure of the solution is correct. The amplitude
$a$, however, grows roughly by a factor of three between each snapshot —
the opposite of the exponential decay we expect. The network has learned the
right shape but the wrong dynamics, compensating for the missing decay by
inflating the amplitude instead. Without the physics loss to enforce the heat
equation in the interior, there is nothing to prevent this.

## Summary

PINNs offer an elegant solution to a common problem in scientific machine
learning — what do you do when data is sparse but you know something about
the physics governing the system? By encoding the governing equations directly
into the loss function as a PDE residual, the network is constrained to learn
solutions that are physically consistent, even in regions where no data exists.

The ablation studies make the contribution of each loss term concrete. Without
the physics loss, the network finds a valid solution to the heat equation but
one that ignores the boundary and initial conditions entirely — a mathematically
correct but physically meaningless result. Without the data loss, the network
fits the boundary and initial conditions well but has no reason to obey the
heat equation in the interior, learning the right spatial structure but the
wrong dynamics.

One of the appealing things about PINNs is how little changes when you swap
out the underlying PDE. Extending this to Burgers' equation, or to any other
PDE, requires only updating the physics residual — the network architecture,
training loop, and everything else stays the same. This generality is part of
what makes PINNs a useful tool beyond just physics problems; any system where
you have reliable prior knowledge about the structure of the solution is a
candidate.

An interesting direction not explored here is the inverse problem — rather than
using a known PDE to constrain the solution, you instead learn the parameters
of the PDE itself from data. Raissi et al. explore this in the
[second part](https://arxiv.org/abs/1711.10566) of their original work, and
it is a natural next step.
