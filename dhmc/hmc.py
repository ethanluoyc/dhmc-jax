import collections

import jax.numpy as np
from jax import random, tree_util
from numpyro import distributions
from symppl.infer import utils, mcmc

IntegratorState = collections.namedtuple(
    "IntegratorState", "state, state_grads, target_log_prob, momentum")


def leapfrog_step(state,
                  target_log_prob_fn,
                  kinetic_energy_fn,
                  step_size,
                  rng=None):
    """Single step of leapfrog.

    Notes
    =====

    The canonical distribution is related to the energy of the system 
    by 

    p(p, \theta) = 1/Zexp(-H(\theta, p)/T)

    For now, we assume that the kinetic energy takes
    the form
    K(p) = sum_i(p_i^2/(2m_i))
    """
    del rng
    p, q, q_grad = state.momentum, state.state, state.state_grads
    p_half = tree_util.tree_multimap(lambda p, qg: p + 0.5 * step_size * qg, p,
                                     q_grad)
    _, grad_p_half = utils.call_fn_value_and_grad(kinetic_energy_fn, p_half)
    q_full = tree_util.tree_multimap(lambda q, ph: q + step_size * ph, q,
                                     grad_p_half)
    logprob, q_full_grad = utils.call_fn_value_and_grad(
        target_log_prob_fn, q_full)
    p_full = tree_util.tree_multimap(lambda ph, qg: ph + 0.5 * step_size * qg,
                                     p_half, q_full_grad)
    return IntegratorState(q_full, q_full_grad, logprob, p_full)


def integrator_step(state, target_log_prob_fn, integrator_step_fn,
                    kinetic_energy_fn, num_steps, step_size, rng):
    for _ in range(num_steps):
        rng, rng_step = random.split(rng)
        state = integrator_step_fn(state,
                                   target_log_prob_fn,
                                   kinetic_energy_fn,
                                   step_size=step_size,
                                   rng=rng_step)
    return state._replace(
        momentum=tree_util.tree_map(lambda x: -x, state.momentum))


_momentum_dist = distributions.Normal()


def sample_momentum(rng, state):
    """Sample momentum p for the system"""
    rngs = utils.split_rng_as(rng, state)
    return tree_util.tree_multimap(
        lambda s, r: _momentum_dist.sample(r, sample_shape=s.shape), state,
        rngs)


def gaussian_kinetic_energy_fn(*state):
    # TODO: customize this when sampling momentum is also custimizable
    # ke = tree_util.tree_map(lambda s: -np.sum(_momentum_dist.log_prob(s)),
    #                         state)
    ke = tree_util.tree_map(lambda s: np.sum(np.square(s)) / 2, state)
    return tree_util.tree_reduce(lambda a, b: a + b, ke)


def hamiltonian_monte_carlo_step(rng,
                                 target_log_prob_fn,
                                 current_state,
                                 kinetic_energy_fn=gaussian_kinetic_energy_fn,
                                 integrator_step_fn=leapfrog_step,
                                 sample_momentum_fn=sample_momentum,
                                 path_length=1.0,
                                 step_size=1.):
    """ rng: random key
    state: initial state for parameters
    """
    rng, rng_momentum, rng_loguniform, rng_integrate = random.split(rng, 4)

    start_momentum = sample_momentum_fn(rng=rng_momentum, state=current_state)
    start_kinetic_energy = utils.call_fn(kinetic_energy_fn, start_momentum)
    num_steps = int(path_length / step_size)

    logprob, state_grads = utils.call_fn_value_and_grad(
        target_log_prob_fn, current_state)
    integrator_state = IntegratorState(current_state, state_grads, logprob,
                                       start_momentum)

    final_integrator_state = integrator_step(integrator_state,
                                             target_log_prob_fn,
                                             integrator_step_fn,
                                             kinetic_energy_fn,
                                             num_steps,
                                             step_size,
                                             rng=rng_integrate)
    proposed_state = final_integrator_state.state
    proposed_kinetic_energy = utils.call_fn(kinetic_energy_fn,
                                            final_integrator_state.momentum)

    # HMC accepts with probability
    # alpha = min(1, exp(-U(q*)+U(q)-K(q*)+K(p)))
    # this is equivalent to
    #   u ~ U[0, 1]
    #   log(u) < exp(-U(q*)+U(q)-K(p*)+K(p))
    #   log(u) < exp(-U(q*)+U(q)-K(p*)+K(p))
    #   log(u) < (pi(q*)-pi(q) + (K(p) - K(p*)))
    #   <=> log(u) < (pi(q*)-pi(q) + (K(p) - K(p*)))
    # Check Metropolis acceptance criterion
    start_energy = -utils.call_fn(target_log_prob_fn,
                                  current_state) + start_kinetic_energy
    new_energy = -utils.call_fn(target_log_prob_fn,
                                proposed_state) + proposed_kinetic_energy
    log_acceptance_ratio = start_energy - new_energy
    return mcmc.metropolis_hasting_step(rng_loguniform, proposed_state,
                                        current_state, log_acceptance_ratio)
