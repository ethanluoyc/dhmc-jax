import functools

import jax
import jax.numpy as np
import jax.random as random
from jax import tree_util

from symppl.infer.hmc import IntegratorState, hamiltonian_monte_carlo_step
from symppl.infer import utils


def gauss_laplace_leapfrog(current_state,
                           target_log_prob_fn,
                           kinetic_energy_fn,
                           step_size,
                           n_disc,
                           rng=None):
    """
    One numerical integration step of the DHMC integrator for a mixed
    Gaussian and Laplace momentum.

    Params
    ------
    f: function(theta, req_grad)
      Returns the log probability and, if req_grad is True, its gradient.
      The gradient for discrete parameters should be zero.
    f_update: function(theta, step_sizeheta, index, aux)
      Computes the difference in the log probability when theta[index] is
      modified by step_sizeheta. The input 'aux' is whatever the quantity saved from
      the previous call to 'f' or 'f_update' that can be recycled.
    M: column vector
      Represents the diagonal mass matrix
    n_disc: int
      Number of discrete parameters. The parameters theta[:-n_disc] are
      assumed continuous.
    """
    del kinetic_energy_fn
    assert isinstance(current_state.state, list)
    assert isinstance(current_state.state_grads, list)

    M = tree_util.tree_map(np.ones_like, current_state.state)
    state, state_grads = current_state.state, current_state.state_grads
    momentum = current_state.momentum

    n_param = len(state)
    state = list(state)
    # Update the continuous parameters
    momentum[:-n_disc] = tree_util.tree_multimap(
        lambda p, g: p + 0.5 * step_size * g, momentum[:-n_disc],
        state_grads[:-n_disc])

    state[:-n_disc] = tree_util.tree_multimap(
        lambda t, p: t + 0.5 * step_size * p, state[:-n_disc],
        momentum[:-n_disc])
    logp = utils.call_fn(target_log_prob_fn, state)
    if np.isinf(logp):
        return current_state
    # Update discrete
    coord_order = n_param - n_disc + np.arange(n_disc)
    coord_order = random.shuffle(rng, coord_order)
    for index in coord_order:
        state, momentum, logp = _update_coordwise(target_log_prob_fn, index,
                                                  state, momentum, M,
                                                  step_size, logp)
    # Another half step of discrete
    state[:-n_disc] = tree_util.tree_multimap(
        lambda t, p: t + 0.5 * step_size * p, state[:-n_disc],
        momentum[:-n_disc])
    new_target_logp, new_state_grads = utils.call_fn_value_and_grad(
        target_log_prob_fn, state)
    momentum[:-n_disc] = tree_util.tree_multimap(
        lambda p, g: p + 0.5 * step_size * g, momentum[:-n_disc],
        new_state_grads[:-n_disc])
    return IntegratorState(state=state,
                           state_grads=new_state_grads,
                           target_log_prob=new_target_logp,
                           momentum=momentum)


def _update_coordwise(target_log_prob_fn, index, theta, p, M, step_size,
                      target_logp):
    """Update the parameter at index"""
    theta = theta[:]
    p = p[:]

    def f_update(theta, dtheta, index):
        theta_copied = list(theta[:])
        theta_copied[index] = theta_copied[index] + dtheta
        return utils.call_fn(target_log_prob_fn, theta_copied)

    p_sign = np.sign(p[index])
    dtheta = p_sign / M[index] * step_size
    new_target_logp = f_update(theta, dtheta, index)

    potential_diff = target_logp - new_target_logp
    if np.abs(p[index]) / M[index] > potential_diff:
        p[index] = p[index] - p_sign * M[index] * potential_diff
        theta[index] += dtheta
        target_logp = new_target_logp
    else:
        p[index] = -p[index]
    return theta, p, target_logp


def sample_momentum(rng, state, n_disc):
    state = tree_util.tree_map(np.asarray, state)

    rngn, rngl = random.split(rng)
    s_cont, s_disc = state[:-n_disc], state[-n_disc:]
    rngn = utils.split_rng_as(rngn, s_cont)
    rngl = utils.split_rng_as(rngl, s_disc)

    p_cont = tree_util.tree_multimap(
        lambda s, r: random.normal(r, shape=s.shape), s_cont, rngn)
    p_disc = tree_util.tree_multimap(
        lambda s, r: random.laplace(r, shape=s.shape), s_disc, rngl)

    return p_cont + p_disc


def gauss_laplace_kinetic_energy_fn(*state, n_disc=0):
    state = list(state)
    p_cont, p_disc = state[:-n_disc], state[-n_disc:]
    ke_cont = sum(
        tree_util.tree_map(lambda x: np.sum(np.square(x) / 2), p_cont))
    ke_disc = sum(tree_util.tree_map(lambda x: np.sum(np.abs(x)), p_disc))
    return ke_cont + ke_disc


def dhmc_step(rng,
              target_log_prob_fn,
              current_state,
              n_disc,
              path_length=1.0,
              step_size=1.):
    """ rng: random key
    state: initial state for parameters
    """

    kinetic_energy_fn = functools.partial(gauss_laplace_kinetic_energy_fn,
                                          n_disc=n_disc)

    integrator_step_fn = functools.partial(gauss_laplace_leapfrog,
                                           n_disc=n_disc)
    sample_momentum_fn = functools.partial(sample_momentum, n_disc=n_disc)
    return hamiltonian_monte_carlo_step(rng,
                                        target_log_prob_fn,
                                        current_state,
                                        kinetic_energy_fn,
                                        integrator_step_fn,
                                        sample_momentum_fn,
                                        path_length=path_length,
                                        step_size=step_size)
