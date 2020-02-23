import jax
import jax.numpy as np
from numpyro import distributions
from symppl.infer import utils


def random_walk_proposal_fn(rng, state, scale=1.0):

    if not utils.is_list_like(state):
        state_ = (state, )
    else:
        state_ = state

    def maybe_flatten(x):
        return x if utils.is_list_like(state) else x[0]

    rngs = jax.random.split(rng, len(state_))
    proposed_state = [
        distributions.Normal(loc=s, scale=scale).sample(r)
        for s, r in zip(state_, rngs)
    ]

    return maybe_flatten(proposed_state), 0.


def metropolis_hasting_step(rng, proposed_state, state, log_acceptance_ratio):
    logu = np.log(jax.random.uniform(rng, shape=log_acceptance_ratio.shape))

    is_accepted = logu < log_acceptance_ratio
    return utils.choose(is_accepted, proposed_state, state)


def random_walk_metropolis_hasting_step(rng, state, target_log_prob_fn,
                                        proposal_fn):

    rng, rng_logu = jax.random.split(rng)
    proposed_state, log_proposed_bias = proposal_fn(rng, state)

    old_state_log_prob = utils.call_fn(target_log_prob_fn, state)

    proposed_target_log_prob = utils.call_fn(target_log_prob_fn,
                                             proposed_state)
    assert old_state_log_prob.shape == proposed_target_log_prob.shape
    log_acceptance_ratio = (proposed_target_log_prob - old_state_log_prob -
                            log_proposed_bias)

    return metropolis_hasting_step(rng_logu, proposed_state, state,
                                   log_acceptance_ratio)
