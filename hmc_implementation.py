#!/usr/bin/env python3
# coding: utf-8
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
import time

@jax.jit
def multivariate_gaussian(x, mean, cov):
    '''
    Computes the probability density function (PDF) of a multivariate
    Gaussian distribution for multiple points.

    Args:
        x (np.ndarray): The data points, with shape (N, D).
        mean (np.ndarray): The mean vector, with shape (D,).
        cov (np.ndarray): The covariance matrix, with shape (D, D).

    Returns:
        np.ndarray: The PDF values for each point, with shape (N,).
    '''
    # Ensure x is a 2D array (N, D)
    x = jnp.atleast_2d(x)
    assert x.ndim == 2, "Input x must be a 2D array"
    assert mean.ndim == 1, "Mean must be a 1D array"
    # mean = np.atleast_1d(mean)
    D = mean.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    det = jnp.linalg.det(cov)
    d = x - mean
    # Squared Mahalanobis distance
    mahalanobis_sq = jnp.einsum('ij,jk,ik->i', d, cov_inv, d)
    return (1.0 / jnp.sqrt((2 * jnp.pi) ** D * det)) * jnp.exp(-0.5 * mahalanobis_sq)

@jax.jit
def target(x):
    """ This is the target distribution we want to sample from. """
    mean1 = jnp.array([0.5, 0.5])
    cov1 = jnp.array([[0.1, 0.05], [0.05, 0.1]])
    return multivariate_gaussian(x, mean1, cov1)


@partial(jax.jit, static_argnames=['dVdq'])
def leapfrog_step(q, p, dVdq, step_size):
    """Perform a single leapfrog step.

    Parameters
    ----------
    q : jnp.floatX
        Current position
    p : jnp.floatX
        Current momentum
    dVdq : callable
        Gradient of the velocity
    step_size : float
        Step size for the leapfrog integration

    Returns
    -------
    q, p : jnp.floatX, jnp.floatX
        Updated position and momentum after one leapfrog step
    """
    p_half = p - 0.5 * step_size * dVdq(q)  # Half step for momentum
    q_new = q + step_size * p_half  # Full step for position
    p_new = p_half - 0.5 * step_size * dVdq(q_new)  # Half step for momentum
    return q_new, p_new


@jax.jit
def negative_log_prob(x):
    res = -jnp.log(target(x) + 1e-12)  # Add a small constant to avoid log(0)
    return res.sum()


@partial(jax.jit, static_argnames=['dVdq'])
def integrate_block(key, q0, p0, dVdq, step_size, block_len, direction, collection):
    """
    Integrate for Hamiltonian Monte Carlo using the tree expansion method.
    """
    s = direction * step_size
    
    q, p = jnp.copy(q0), jnp.copy(p0)

    """
    Collection is an array with size max_block_len where we will store the positions,
    momentums and weights at each step of the integration.
    """
    def body(i, val):
        q, p = val[0], val[1]
        q, p = leapfrog_step(q, p, dVdq, s)
        H = hamiltonian_energy(q, p, negative_log_prob)
        w = jnp.exp(-H)
        collected = (val[2][0].at[i].set(q), val[2][1].at[i].set(p), val[2][2].at[i].set(w))
        return (q, p, collected)

    collected = jax.lax.fori_loop(0, block_len, body, (q, p, collection))

    q_traj, p_traj, w_traj = collected[2]
    
    """
    w_traj has 0s in the unused part of the array, so we can sum without problems.
    """
    total_w = jnp.sum(w_traj)

    # Normalize weights
    normalized_w = w_traj / total_w

    # If we will pick a sample from this trajectory, let's pick it now
    """
    In the unused part of the array we have 0 weights, so we will never pick those samples.
    """
    idx = jax.random.categorical(key, jnp.log(normalized_w))
    q_selected = q_traj[idx]
    p_selected = p_traj[idx]

    return (q_selected, p_selected), total_w


def kinetic_energy(p):
    """Compute the kinetic energy.
    
    :param jnp.array p:
        Momentum
    :return float:
        The kinetic energy
    """
    return 0.5 * jnp.sum(p ** 2)


def potential_energy(q, negative_log_prob):
    """Compute the potential energy.

    :param jnp.array q:
        Position
    :param callable negative_log_prob:
        The negative log probability to sample from
    :return float:
        The potential energy
    """
    return negative_log_prob(q)


def hamiltonian_energy(q, p, negative_log_prob):
    """Compute the Hamiltonian energy.

    :param jnp.array q:
        Position
    :param jnp.array p:
        Momentum
    :param callable negative_log_prob:
        The negative log probability to sample from
    :return float:
        The Hamiltonian energy
    """
    return potential_energy(q, negative_log_prob) + kinetic_energy(p)

def trajectory_sampling(traj_len, block_len, collection, negative_log_prob, initial_params, step_size, key):
    """
    This is the implementation leveraging the perfect binary tree sampling structure.

    :param int traj_len:
        Length of the trajectory (in number of blocks)
    :param jnp.array block_len:
        Array of block lengths (powers of 2)
    :param tuple of jnp.array collection:
        Preallocated arrays to collect positions, momentums and weights
    :param callable negative_log_prob:
        The negative log probability to sample from
    :param jnp.array initial_params:
        A place to start sampling from.
    :param float step_size:
        How long each integration step is. Smaller is slower and more accurate.
    :return jnp.array:
        A new sample from the target distribution
    """
    dVdq = jax.grad(negative_log_prob)
    rep_q, rep_p = initial_params
    H0 = hamiltonian_energy(rep_q, rep_p, negative_log_prob)
    rep_weight = jnp.exp(-H0)

    def body_fn(carry, iter):
        key, q_old, p_old, rep_weight = carry
        key, key_direction, key_uniform, key_leapfrog = jax.random.split(key, 4)

        direction = jnp.where(jax.random.bernoulli(key_direction, 0.5), 1, -1)

        (q_new, p_new), new_weight = integrate_block(
            key=key_leapfrog,
            q0=q_old,
            p0=p_old,
            dVdq=dVdq,
            step_size=step_size,
            block_len=iter,
            direction=direction,
            collection=collection
        )

        """
        Here we pick either the new sample or the old one based on their weights:

        w_old / (w_old + w_new) T(z'|t_old) + w_new / (w_old + w_new) T(z'|t_new)
        
        Basically here we are saying, with a probability of w_old / (w_old + w_new) we pick a sample
        from the old trajectory, otherwise we pick a sample from the new trajectory.
        If we were to pick a sample from the new trajectory T(z'|t_new), we already did that in the integrate_block
        function. While picking a sample from the old trajectory T(z'|t_old) means we just keep the old sample.

        To implement this we compute the probability of picking the new sample as:
        - p_new = w_new / (w_old + w_new)
        - Avoid division by zero: if w_old + w_new == 0, we set p_new = 0.5 (equal probability)
        - We then draw a uniform random number u ~ U(0, 1). This because p_old + p_new = 1 and p_new = 1 - p_old!

        At every step our representative in not just one state but it's a proxy for a distribution of states, so
        we need to carry the total weight to avoid bias.
        """
        total_w = rep_weight + new_weight
        p_new = jnp.where(total_w > 0, new_weight / total_w, 0.5)
        take_new = jax.random.uniform(key_uniform) < p_new

        q = jnp.where(take_new, q_new, q_old)
        p = jnp.where(take_new, p_new, p_old)
        rep_weight = total_w

        return (key, q, p, rep_weight), (q, p, rep_weight)

    (_, q, p, rep_weight), traj_chain = jax.lax.scan(body_fn, (key, rep_q, rep_p, rep_weight), block_len, length=traj_len+1)
    
    # Metropolis-Hastings correction (see pag. 39 of Betancourt's book)
    H_init = H0
    H_end = hamiltonian_energy(q, p, negative_log_prob)
    log_accept_ratio = H_init - H_end
    log_accept_ratio = jnp.nan_to_num(log_accept_ratio, neginf=-jnp.inf)
    _ , key_accept = jax.random.split(key)
    accept_ratio = jax.random.uniform(key_accept) < jnp.minimum(1.0, jnp.exp(log_accept_ratio))
    q = jnp.where(accept_ratio, q, rep_q)
    return q


def hmc_sample(n_samples, TODO_block_len, negative_log_prob, initial_params, traj_len, step_size):
    """
    This is the main function to sample from the target distribution using HMC.

    :param int n_samples:
        Number of samples to draw
    :param int TODO_block_len:
        --- IGNORE ---
    :param callable negative_log_prob:
        The negative log probability to sample from
    :param jnp.array initial_params:
        A place to start sampling from.
    :param int traj_len:
        Length of the trajectory (in number of blocks)
    :param float step_size:
        How long each integration step is. Smaller is slower and more accurate.
    :return jnp.array:
        An array of samples from the target distribution
    """
    # @TODO: block_len to param
    key = jax.random.PRNGKey(42)
    key, _ = jax.random.split(key)
    momentum = jax.random.normal(key, shape=(n_samples, initial_params.shape[0]))
    """
    Here collection is a matrix of size (max_block_len, dim) where dim is the dimensionality of the problem.
    We will use it to collect the positions and momentums at each step of the integration. With Jax we cannot
    dynamically resize arrays, so we need to preallocate the maximum size we will need.
    We can then access only the relevant part of the array later.
    """
    collection = (jnp.zeros((2**traj_len, initial_params.shape[0])),
                  jnp.zeros((2**traj_len, initial_params.shape[0])),
                  jnp.zeros((2**traj_len,)))

    block_len = jnp.array([2**x for x in range(traj_len+1)])

    # Scan over the number of samples
    def body_fn(carry, x):
        key, q = carry
        key, subkey = jax.random.split(key)
        sample = trajectory_sampling(
            traj_len=traj_len,
            block_len=block_len,
            collection=collection,
            negative_log_prob=negative_log_prob,
            initial_params=(q, x),
            step_size=step_size,
            key=subkey
        )
        return (key, sample), sample

    (_, _), samples = jax.lax.scan(body_fn, (key, initial_params), momentum, length=n_samples)    

    return samples


if __name__ == "__main__":
    """
    In this HMC implementation we will consider the problem of sampling from a multivariate
    Gaussian distribution using Hamiltonian Monte Carlo (HMC). We will use JAX to compute
    gradients and perform automatic differentiation.
    """

    x_axis = jnp.linspace(-2, 2, 100)
    y_axis = jnp.linspace(-2, 2, 100)
    X, Y = jnp.meshgrid(x_axis, y_axis)
    XY = jnp.array([X.flatten(), Y.flatten()]).T

    Z = target(XY)

    n_samples = 1000
    initial = jnp.array([0.0, 0.0])
    step_size = 0.1
    path_len = 3.0

    start_time = time.time()
    samples = hmc_sample(
        n_samples=n_samples,
        TODO_block_len=None,
        negative_log_prob=negative_log_prob,
        initial_params=initial,
        traj_len=3,
        step_size=step_size
    ).block_until_ready()
    end_time = time.time()
    print(f"Simple HMC took {end_time - start_time:.2f} seconds for {n_samples} samples.")

    plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contourf(X, Y, Z.reshape(X.shape), cmap='viridis')
    plt.colorbar()
    plt.scatter(samples[:, 0], samples[:, 1], s=1, color='red', alpha=0.5)
    plt.show()