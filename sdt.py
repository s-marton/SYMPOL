import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import normal
import jax

def temperature_sigmoid(x, temperature=1.0):
    """ Custom sigmoid function with temperature. """
    return 1 / (1 + jnp.exp(-x / temperature))

def entmoid15(x, temperature=1.0):
    return entmax15JAX(jnp.stack([x / temperature, jnp.zeros(x.shape)], axis=-1), axis=-1)[..., 0]

class SubtractiveEntmaxDense(nn.Module):
    features: int  # Number of output features
    temperature: float
    
    @nn.compact
    def __call__(self, inputs, max_path=False):
        # Create weight and bias variables
        weight = self.param('kernel', normal(0.1), (inputs.shape[-1], self.features))
        bias = self.param('bias', normal(1.0), (self.features,))
        
        # Compute the dense layer output with bias subtraction
        weight = jax.lax.cond(
            max_path,
            lambda x: weight,
            lambda x: entmax15JAX(weight.T / self.temperature).T,
            weight
        )
            
        output = jnp.dot(inputs, weight) - bias
        output = jax.lax.cond(
            max_path,
            #lambda x: jnp.round(entmoid15(output, self.temperature)),
            lambda x: entmoid15(output, 0.0001),
            lambda x: entmoid15(output, self.temperature),
            output
        )
      
        return output
    
class SDT(nn.Module):
    input_dim: int
    output_dim: int
    depth: int = 3
    temperature: float = 1.0
    action_type: str = "discrete" #"continuous"

    def setup(self):
        self.internal_node_num = 2 ** self.depth - 1
        self.leaf_node_num = 2 ** self.depth

        self.inner_nodes = nn.Sequential([
            SubtractiveEntmaxDense(self.internal_node_num, self.temperature),#, use_bias=True, kernel_init=normal(0.1), bias_init=normal(1.0)),
            #nn.sigmoid
            #lambda x: temperature_sigmoid(x, self.temperature)
            #lambda x: entmoid15(x, self.temperature)
            #entmoid15
        ])
        
        if self.action_type == "discrete":
            self.leaf_nodes = nn.Dense(self.output_dim, use_bias=False, kernel_init=normal(0.1))
            self.stds = None#nn.Dense(self.output_dim, use_bias=False, kernel_init=normal(0.1))
        else:
            self.leaf_nodes = nn.Dense(self.output_dim, use_bias=False, kernel_init=normal(0.1))
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.output_dim,))      
    def __call__(self, x, max_path):
        batch_size = x.shape[0]
        #x = self._data_augment(x)
        #inner_nodes = self.inner_nodes
        #inner_nodes = entmax15JAX(inner_nodes)

        path_prob = self.inner_nodes(x, max_path=max_path)
        #entmax15JAX(self.inner_nodes(
        
        path_prob = jnp.expand_dims(path_prob, axis=2)
        path_prob = jnp.concatenate((path_prob, 1 - path_prob), axis=2)
        
        mu = jnp.ones((batch_size, 1, 1))

        begin_idx = 0
        end_idx = 1

        for layer_idx in range(self.depth):
            path_prob_layer = path_prob[:, begin_idx:end_idx, :]
            
            mu = jnp.reshape(mu, (batch_size, -1, 1))
            mu = jnp.tile(mu, (1, 1, 2))

            mu = mu * path_prob_layer

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = mu.reshape(batch_size, self.leaf_node_num)

        if self.action_type == "discrete":
            y_pred = self.leaf_nodes(mu)
        else:
            mean = self.leaf_nodes(mu)
            y_pred = [mean, self.log_std]
            
        return y_pred

class Critic_SDT(nn.Module):
    depth: int = 5
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, max_path=False, **kwargs):
        sdt = SDT(input_dim=x.shape[-1], output_dim=1, depth=self.depth, temperature=self.temperature)
        return sdt(x, False)#, **kwargs)

class Actor_SDT(nn.Module):
    action_dim: int
    depth: int = 5
    temperature: float = 1.0
    action_type: str = "discrete" #"continuous"
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, max_path=False, **kwargs):
        sdt = SDT(input_dim=obs.shape[-1], output_dim=self.action_dim, depth=self.depth, temperature=self.temperature, action_type=self.action_type)
        return sdt(obs, max_path)#, **kwargs)



"""
Taken from: https://github.com/deep-spin/entmax/blob/master/entmax/activations.py

An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.
"""


# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>
# License: MIT


def top_k_over_axisJAX(inputs, k, axis=-1, **kwargs):
    with jax.named_scope("top_k_along_axis"):
        if axis == -1:
            return jax.lax.top_k(inputs, k)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = jnp.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = jax.lax.top_k(input_perm, k=k, **kwargs)

        input_sorted = jnp.transpose(input_perm_sorted, inv_order)
        sort_indices = jnp.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """creates indices 0, ... , input[axis] unsqueezed to input dimensios"""
    assert jnp.ndim(inputs) is not None
    rho = jnp.arange(1, inputs.shape[axis] + 1, dtype=jnp.float32)
    view = [1] * jnp.ndim(inputs)
    view[axis] = -1
    return jnp.reshape(rho, view)


def jax_gather_nd(params, indices):
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]


def gather_over_axisJAX(values, indices, gather_axis):
    assert jnp.ndim(indices) is not None
    assert jnp.ndim(indices) == jnp.ndim(values)

    ndims = jnp.ndim(indices)
    gather_axis = gather_axis % ndims
    shape = jnp.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = jnp.arange(shape[axis_i])
            index_i = jnp.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = jnp.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)
    return jax_gather_nd(values, jnp.stack(selectors, axis=-1))


def entmax_threshold_and_supportJAX(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with jax.named_scope("entmax_threshold_and_supportJAX"):
        num_outcomes = inputs.shape[axis]

        inputs_sorted, _ = top_k_over_axisJAX(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = jnp.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = jnp.cumsum(jnp.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - jnp.square(mean))) / rho

        delta_nz = jax.nn.relu(delta)
        tau = mean - jnp.sqrt(delta_nz)

        support_size = jnp.sum(jnp.less_equal(tau, inputs_sorted), axis=axis, keepdims=True)

        tau_star = gather_over_axisJAX(tau, support_size - 1, axis)
    return tau_star, support_size


def entmax15JAX(inputs, axis=-1):

    # Implementation taken from: https://github.com/deep-spin/entmax/tree/master/entmax

    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """

    @jax.custom_gradient
    def _entmax_inner(inputs):
        with jax.named_scope("entmax"):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= jnp.max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_supportJAX(inputs, axis)
            outputs_sqrt = jax.nn.relu(inputs - threshold)
            outputs = jnp.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with jax.named_scope("entmax_grad"):
                d_inputs = d_outputs * outputs_sqrt
                q = jnp.sum(d_inputs, axis=axis, keepdims=True)
                q = q / jnp.sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)
