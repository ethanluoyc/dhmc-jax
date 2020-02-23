import jax
import jax.numpy as np
from jax import lax
from jax import tree_util


def is_list_like(x):
    if isinstance(x, (list, tuple)):
        return True
    else:
        return False


def call_fn(fn, args):
    if is_list_like(args):
        return fn(*args)
    return fn(args)


def call_fn_value_and_grad(fn, args):
    def _fn(args):
        if is_list_like(args):
            return fn(*args)
        return fn(args)

    return jax.value_and_grad(_fn)(args)


def choose(is_accepted, proposed_state, state):
    def _expand_is_accepted_like(x):
        if x.shape is not None and is_accepted.shape is not None:
            expand_shape = list(is_accepted.shape) + [1] * (
                len(x.shape) - len(is_accepted.shape))
        else:
            expand_shape = (is_accepted.shape + (1, ) *
                            (x.ndim - is_accepted.ndim))

        return np.reshape(is_accepted, expand_shape)

    if is_list_like(proposed_state):
        assert is_list_like(state)
        return [
            np.where(_expand_is_accepted_like(p), p, s)
            for p, s in zip(proposed_state, state)
        ]
    else:
        return np.where(_expand_is_accepted_like(proposed_state),
                        proposed_state, state)


def trace(state, fn, num_steps, trace_fn=None):

    if trace_fn is None:
        trace_fn = lambda state, extra: extra

    def wrapped_fn(state, _unused):
        next_state, aux = fn(state)
        return next_state, trace_fn(next_state, aux)

    (final_state, out) = lax.scan(wrapped_fn, state, xs=None, length=num_steps)
    return final_state, out

def split_rng_as(rng, structure):
    struct_flat, tree = tree_util.tree_flatten(structure)
    if len(struct_flat) == 1:
        rngs = (rng, )
    else:
        rngs = jax.random.split(rng, len(struct_flat))
    return tree_util.tree_unflatten(tree, rngs)