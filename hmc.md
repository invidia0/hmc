# Hamiltonian Monte Carlo (HMC)
> Main reference: [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434) by Michael Betancourt

The problem with MCMC is that it can get stuck in local modes, especially in high-dimensional spaces. Hamiltonian Monte Carlo (HMC) is a method that helps to overcome this problem by using concepts from physics to guide the sampling process.

## Typical set
In high-dimensional spaces, the volume of the space increases exponentially with the number of dimensions. As a result, most of the probability mass is concentrated in a thin shell known as the **typical set**. You can think of the typical set as the region where most of the samples from the distribution will lie.


>[!Note]
> Consider a probability distribution, the typical set is a fairly high-density area of the likelihood function, but not the highest-density area. For example, in a 2D Gaussian distribution, the typical set is an annulus around the mean, where most of the samples will fall, rather than at the peak of the distribution. Suppose we have a density on [0, 3], and the height on [1,2] is a quadrillion times higher than the rest. For all intents, we can ignore everything outside of [1,2]. With standard MCMC, we are really unlikely to sample points outside [1,2], and we will get stuck in the high-density region. But in 10 dimensions this is not true anymore. We are focusing on $(\frac{1}{3})^{10}$ of the sapce, which is about 0.000016%. The typical set is now a thin shell around the high-density region, and we are much more likely to sample points in this shell. This is why HMC is so effective in high-dimensional spaces.

So the question we are trying to answer is: **how can we distill the geometry of the typical set into information about how to move through it?**

The answer is a *vector field*. A vector field is the assignment of a direction at every point in the parameter space, and if those directions are aligned with the typical set, then they act as a guide through the typical set.

But how do we construct such a vector field? This is where Hamiltonian dynamics come in. The key is to twist the gradient vector field into a vector field aligned with the typical set, and hence once capable of generating efficient exploration, is to expand the original probabilistic system with the introduction of **auxiliary momentum parameters**.

>[!Tip]
> Instead of trying to reason about a mode (a peak in the density), a gradient (a slope), and a typical set (a thin shell around the peak), we can think about a planet, a gravitational field and an orbit. The planet is the mode, the gravitational field is the gradient, and the orbit is the typical set.
>
> The probabilistic endeavour of exploring the typical set is becomes a physical endeavor of placing a satellite in a stable orbit around the hypotetical planet. Just placing the satellite in the gravitational field will cause it to fall into the planet. Instead, we need to give it some initial momentum to counteract the gravitational pull and keep it in orbit.
>
> - Too much momentum and the satellite will escape the gravitational pull and fly off into space (exploration of low-density regions).
> - Too little momentum and the satellite will fall into the planet (getting stuck in high-density regions).
> - Just the right amount of momentum and the satellite will stay in a stable orbit around the planet (efficient exploration of the typical set).

## Phase Space and Hamilton's Equations
We can introduce auxiliary momentum parameters $p_n$ to complement each dimension of our target parameter space,

$$
q_n \rightarrow (q_n, p_n).
$$

We can now lift the target distribution $\pi(q)$ into a joint distribution over the augmented phase space $(q, p)$,
$$
\pi(q, p) = \pi(p\mid q) \pi(q),
$$
that ensure the marginal distribution over the original parameters is preserved and we immediately recover our target distribution by marginalizing out the auxiliary momentum parameters,
$$
\pi(q) = \int \pi(q, p) \, dp.
$$
Importantly, it guarantees that **any trajectory exploring the typical set of the phase space distribution will project to trajectories exploring the typical set of the target distribution**.

We can write the joint distribution in terms of a Hamiltonian function $H(q, p)$,

$$
\pi(q, p) = \exp(-H(q, p)).
$$

We must note that because the Hamiltonian is independent of the details of any parameterization of the system, we are free to choose any convenient form for the Hamiltonian. Moreover, the value of the Hamiltonian at any point in phase space is called the **energy** at that point.

The Hamiltonian function can be decomposed into two parts, the potential energy $V(q)$ and the kinetic energy $T(p, q)$,

$$
\begin{aligned}
\mathcal{H}(q, p) &= -\log \pi(q, p) \\
&= -\log \pi(p\mid q) - \log \pi(q) \\
&= K(p, q) + V(q).
\end{aligned}
$$

Here:
- $V(q) = -\log \pi(q)$ is the **potential energy**, which depends only on the position variables $q$. The potential energy is completely determined by the target distribution.
- $K(p, q) = -\log \pi(p\mid q)$ is the **kinetic energy**, which can depend on both the momentum variables $p$ and the position variables $q$. The kinetic energy is unconstrained and must be specified by the implementation.

Because the Hamiltonian captures the geometry of the typical set, we can use it to generate a vector field oriented with the typical set of the canonical distribution and hence the trajectories we are after.

The vector field is given by Hamilton's equations,

$$
\begin{aligned}
\dfrac{dq_n}{dt} &= +\dfrac{\partial \mathcal{H}}{\partial p_n} = +\dfrac{\partial K}{\partial p_n}, \\
\dfrac{dp_n}{dt} &= -\dfrac{\partial \mathcal{H}}{\partial q_n} = -\dfrac{\partial K}{\partial q_n} - \dfrac{\partial V}{\partial q_n}.
\end{aligned}
$$

Here we can note that $\partial V/\partial q_n$ is just the gradient of the log density of the target distribution, which we can compute. The other terms depend on our choice of kinetic energy.

Following the Hamiltonian vector field for some time $t$, generates trajectories that rapidly move through phase space while being constrained to the typical set. Projecting these trajectories back down onto the target parameter space finally yields the efficient exploration of the target typical set for which we are searching.

## How HMC works
>[!Note]
> For reference, I followed this [blog post](https://colindcarroll.com/blog/hmc_from_scratch.html).

The algorithm works as follows:
1. Concatenate all parameters into a single position variable $\mathbf{q}$. The probability density function we are trying to sample from is $\pi(\mathbf{q})$.
2. Add a *momentum* variable $\mathbf{p}$ of the same dimension as $\mathbf{q}$ and define the joint probability density function

    $$
    \pi(\mathbf{q}, \mathbf{p}) = \pi(\mathbf{q}) \pi(\mathbf{p}\mid \mathbf{q}),
    $$

    where $\pi(\mathbf{p}\mid \mathbf{q})$ is a conditional distribution over the momentum variable given the position variable that we get to choose. A common choice is to use a Gaussian distribution with zero mean and covariance matrix $M$,

    $$
    \pi(\mathbf{p}\mid \mathbf{q}) = \mathcal{N}(\mathbf{p}\mid 0, M),
    $$

    where $M$ is called the *mass matrix* and usually chosen to be $M = I$ (the identity matrix).
3. Define the Hamiltonian function

    $$
    \begin{aligned}
    \mathcal{H}(\mathbf{q}, \mathbf{p}) &= -\log \pi(\mathbf{q}, \mathbf{p}) \\
    &= -\log \pi(\mathbf{p}\mid \mathbf{q}) - \log \pi(\mathbf{q}) \\
    &= K(\mathbf{p}, \mathbf{q}) + V(\mathbf{q}).
    \end{aligned}
    $$

    Here, $V(\mathbf{q}) = -\log \pi(\mathbf{q})$ is the *potential energy* and $K(\mathbf{p}, \mathbf{q}) = -\log \pi(\mathbf{p}\mid \mathbf{q})$ is the *kinetic energy*.
4. Given the current state $(\mathbf{q}, \mathbf{p})$, evolve the system $(\mathbf{q}, \mathbf{p})$ according to Hamilton's equations for some time $t$ to get a new state $(\mathbf{q}^*, \mathbf{p}^*)$.
   
    $$
    \begin{aligned}
    \dfrac{d\mathbf{q}}{dt} &= +\dfrac{\partial \mathcal{H}}{\partial \mathbf{p}} = +\dfrac{\partial K}{\partial \mathbf{p}}, \\
    \dfrac{d\mathbf{p}}{dt} &= -\dfrac{\partial \mathcal{H}}{\partial \mathbf{q}} = -\dfrac{\partial K}{\partial \mathbf{q}} - \dfrac{\partial V}{\partial \mathbf{q}}.
    \end{aligned}
    $$

    Here, $\partial V/\partial \mathbf{p} = 0$ because $V$ does not depend on $\mathbf{p}$.

We can choose the **kinetic energy** to be a Gaussian, obtaining

$$
K(\mathbf{p}, \mathbf{q}) = \dfrac{1}{2} \mathbf{p}^T M^{-1} \mathbf{p} + \log |M| + \text{constant},
$$

and with $M = I$ (the identity matrix), we have

$$
K(\mathbf{p}, \mathbf{q}) = \dfrac{1}{2} \mathbf{p}^T \mathbf{p} + \text{constant}.
$$

This gives us the following equations:

$$
\begin{aligned}
\dfrac{\partial K}{\partial \mathbf{p}} &= \mathbf{p}, \\
\dfrac{\partial K}{\partial \mathbf{q}} &= 0, \\
\end{aligned}
$$

And the Hamilton's equations simplify to

$$
\begin{aligned}
\dfrac{d\mathbf{q}}{dt} &= \mathbf{p}, \\
\dfrac{d\mathbf{p}}{dt} &= -\dfrac{\partial V}{\partial \mathbf{q}}.
\end{aligned}
$$

Then:
- Sample a $\mathbf{p} \sim \mathcal{N}(0, I)$.
- Simulate $(\mathbf{q}(t), \mathbf{p}(t))$ for some amount of time $t$ using the simplified equations above.
- We get a new sample $\mathbf{q}(T)$.

## Numerical Implementation
In practice, we cannot solve Hamilton's equations analytically, so we need to use numerical integration. A common choice is the **leapfrog integrator**, which is a symplectic integrator that preserves the volume in phase space and is time-reversible. Moreover, there is a **Metropolis acceptance** step to correct for the numerical errors introduced by the integrator.

### Leapfrog Integrator
The leapfrog integrator works as follows:
1. Half-step update of momentum:
   
$$
\mathbf{p}\left(t + \dfrac{\epsilon}{2}\right) = \mathbf{p}(t) - \dfrac{\epsilon}{2} \dfrac{\partial V}{\partial \mathbf{q}}\bigg|_{\mathbf{q}(t)}
$$

2. Full-step update of position:
   
$$
\mathbf{q}(t + \epsilon) = \mathbf{q}(t) + \epsilon \mathbf{p}\left(t + \dfrac{\epsilon}{2}\right)
$$

3. Half-step update of momentum:
   
$$
\mathbf{p}(t + \epsilon) = \mathbf{p}\left(t + \dfrac{\epsilon}{2}\right) - \dfrac{\epsilon}{2} \dfrac{\partial V}{\partial \mathbf{q}}\bigg|_{\mathbf{q}(t + \epsilon)}
$$

Here, $\epsilon$ is the step size.
We repeat these steps $L$ times to simulate the trajectory for a total time of $T = L \epsilon$.

### Metropolis Acceptance Step
After simulating the trajectory, we get a new state $(\mathbf{q}^*, \mathbf{p}^*)$. We then compute the acceptance probability

$$
\alpha = \min\left(1, \exp\left(-\mathcal{H}(\mathbf{q}^*, \mathbf{p}^*) + \mathcal{H}(\mathbf{q}, \mathbf{p})\right)\right).
$$

We accept the new state with probability $\alpha$. If we reject, we stay at the current state. This is necessary because the leapfrog integrator is not exact and introduces some error in the Hamiltonian. The Metropolis step ensures that the Markov chain has the correct stationary distribution.

## Symplectic Integrator Error Correction

The leapfrog integrator is symplectic—preserving phase-space volume and reversibility—but it introduces Hamiltonian error, leading to energy drift over time. A natural way to correct this bias is to embed the Hamiltonian transition in a Metropolis–Hastings (MH) framework.

Directly using leapfrog proposals fails, since integration is one-directional: from \$(q,p)\$ we can reach \$(q^*,p^*)\$, but not the reverse. This breaks detailed balance, giving an acceptance probability of

$$
\dfrac{\mathbb{Q}(q_0,p_0\mid q_L,p_L)}{\mathbb{Q}(q_L,p_L\mid q_0,p_0)} = 0.
$$

To restore reversibility, we augment each step with a **momentum flip**:

$$
(q,p) \mapsto (q,-p),
$$

yielding the MH acceptance rule

$$
\alpha = \min\big(1, e^{-\mathcal{H}(q^*,p^*) + \mathcal{H}(q,p)}\big).
$$

However, this naïve scheme often proposes states with very low acceptance, producing high rejection rates. A better approach is to sample along an entire trajectory \$\mathfrak{t} = {(q\_0,p\_0), \ldots, (q\_L,p\_L)}\$ and then select a point with probability proportional to its Boltzmann weight:

$$
\pi(q,p) = \frac{e^{-\mathcal{H}(q,p)}}{\sum_{(q_i,p_i)\in \mathfrak{t}} e^{-\mathcal{H}(q_i,p_i)}}.
$$

Storing full trajectories is memory-intensive. Instead, we **interleave trajectory construction with sampling**:

* Begin with \$\mathfrak{t}\_\text{old}\$ containing only \$(q\_0,p\_0)\$.
* Extend it by integrating either forward or backward to form \$\mathfrak{t}\_\text{new}\$.
* Merge into \$\mathfrak{t} = \mathfrak{t}*\text{old} \cup \mathfrak{t}*\text{new}\$, and update the active sample using

$$
\mathbb{T}(z'\mid \mathfrak{t}) = \frac{w_\text{old}}{w_\text{old}+w_\text{new}} \, \mathbb{T}(z'\mid \mathfrak{t}_\text{old}) + \frac{w_\text{new}}{w_\text{old}+w_\text{new}} \, \mathbb{T}(z'\mid \mathfrak{t}_\text{new}),
$$

with weights

$$
w_\text{old} = \sum_{(q_i,p_i)\in \mathfrak{t}_\text{old}} e^{-\mathcal{H}(q_i,p_i)}, 
\quad
w_\text{new} = \sum_{(q_i,p_i)\in \mathfrak{t}_\text{new}} e^{-\mathcal{H}(q_i,p_i)}.
$$

Expanding trajectories by doubling their length each iteration makes them equivalent to the leaves of a perfect binary tree (depth \$D\$, length \$L=2^D\$). For example, \$L=8\$ corresponds to depth \$D=3\$:

```
          (q0, p0)
         /        \
    (q1, p1)   (q2, p2)
     /   \       /    \
(q3,p3)(q4,p4)(q5,p5)(q6,p6)
```

At each expansion we sample both the trajectory \$\mathfrak{t}\$ and a state \$z'=(q,p)\$ from it, effectively drawing from the joint distribution

$$
\mathbb{T}(z',\mathfrak{t}\mid z_0) = \mathbb{T}(z'\mid \mathfrak{t}) \, \mathbb{T}(\mathfrak{t}\mid z_0).
$$

In practice, this means generating a random binary tree of leapfrog steps (starting from a momentum draw), and progressively sampling from it with probabilities proportional to \$e^{-\mathcal{H}(q,p)}\$. This avoids storing full trajectories while maintaining detailed balance.

Pseudo-code for the HMC algorithm with the above improvements:

```python
def hmc(n_samples, negative_log_prob, initial_params, path_len, step_size):
    samples = []
    current_params = initial_params
    for _ in range(n_samples):
        # Step 1: Sample initial momentum
        current_momentum = jax.random.normal(jax.random.PRNGKey(_), shape=current_params.shape)

        # Step 2: Initialize trajectory
        trajectory = [(current_params, current_momentum)]
        total_length = 0

        while total_length < path_len:
            # Step 3: Randomly choose direction
            direction = jax.random.choice(jax.random.PRNGKey(_), jnp.array([-1, 1]))

            # Step 4: Double the trajectory length
            new_trajectory = []
            for _ in range(2 ** len(trajectory)):
                last_params, last_momentum = trajectory[-1]
                if direction == 1:
                    new_params, new_momentum = leapfrog_step(last_params, last_momentum, step_size, negative_log_prob)
                else:
                    new_params, new_momentum = leapfrog_step(last_params, -last_momentum, step_size, negative_log_prob)
                new_trajectory.append((new_params, new_momentum))
                trajectory.append((new_params, new_momentum))

            # Step 5: Sample from the combined trajectory
            weights = jnp.array([jnp.exp(-negative_log_prob(q) - 0.5 * jnp.sum(p**2)) for q, p in trajectory])
            weights /= jnp.sum(weights)
            idx = jax.random.choice(jax.random.PRNGKey(_), jnp.arange(len(trajectory)), p=weights)
            current_params, current_momentum = trajectory[idx]

            total_length += len(new_trajectory) * step_size

        samples.append(current_params)
    return jnp.array(samples)
```
