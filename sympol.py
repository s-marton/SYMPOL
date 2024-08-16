from functools import partial
import jax
import jax.numpy as jnp
#from torch.autograd import Function
import distrax

from flax import linen as nn
from flax import struct


@struct.dataclass
class SYMPOL_RL:
    obs_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    depth: int = struct.field(pytree_node=False)
    n_estimators: int = struct.field(pytree_node=False)
    action_type: str = struct.field(pytree_node=False)

    subset_fraction: float = 0.8

    # **kwargs is only used here to be compatible with the flax init procedure, i.e.
    # params = model.init(key, sample_input)
    def init(self, random_key, *args):
        estimator_weights_key, split_values_key, split_index_array_key, leaf_classes_array_key, logstd_key = jax.random.split(
            random_key, 5
        )

        internal_node_num = 2**self.depth - 1
        leaf_node_num = 2**self.depth

        leaf_classes_array_shape = (
            [self.n_estimators, leaf_node_num, self.action_dim]
            if self.action_type == "continuous"
            else [self.n_estimators, leaf_node_num, self.action_dim]
        )

        if self.n_estimators > 1:
            selected_variables = int(self.obs_dim * self.subset_fraction)
            selected_variables = min(selected_variables, 50)
            selected_variables = max(selected_variables, 10)
            selected_variables = min(selected_variables, self.obs_dim)  
            if not selected_variables * self.n_estimators > 3 * self.obs_dim:
                selected_variables = self.obs_dim
        else:
            selected_variables = self.obs_dim
        mu, std = 0.0, 0.05
        estimator_weights = mu + std * jax.random.normal(
            key=estimator_weights_key,
            shape=[self.n_estimators, leaf_node_num],
            dtype=jnp.float32,
        )
        split_values = mu + std * jax.random.normal(
            key=split_values_key,
            shape=[self.n_estimators, internal_node_num, selected_variables],
            dtype=jnp.float32,
        )
        split_index_array = mu + std * jax.random.normal(
            key=split_index_array_key,
            shape=[self.n_estimators, internal_node_num, selected_variables],
            dtype=jnp.float32,
        )
        leaf_classes_array = mu + std * jax.random.normal(
            key=leaf_classes_array_key, shape=leaf_classes_array_shape, dtype=jnp.float32
        )

        log_std = mu + std * jax.random.normal(
            key=logstd_key, shape=[self.action_dim,], dtype=jnp.float32
        )        

        
        params = {
            "estimator_weights": estimator_weights,
            "split_values": split_values,
            "split_idx_array": split_index_array,
            "leaf_array": leaf_classes_array,
            "log_std": log_std,
        }

        return params

    def init_indices(self, random_key):
        leaf_node_num = 2**self.depth
        if self.n_estimators > 1:
            selected_variables = int(self.obs_dim * self.subset_fraction)
            selected_variables = min(selected_variables, 50)
            selected_variables = max(selected_variables, 10)
            selected_variables = min(selected_variables, self.obs_dim)  
            if not selected_variables * self.n_estimators > 3 * self.obs_dim:
                selected_variables = self.obs_dim
        else:
            selected_variables = self.obs_dim

        
        features_by_estimator = jnp.stack(
            [
                jax.random.choice(random_key+i, self.obs_dim, shape=(selected_variables,), replace=False, p=None)
                for i in range(self.n_estimators)
            ]
        )

        path_identifier_list = []
        internal_node_index_list = []
        for leaf_index in range(leaf_node_num):
            for current_depth in range(1, self.depth + 1):
                path_identifier = jnp.floor(leaf_index / (2 ** (self.depth - current_depth))) % 2
                internal_node_index = (2 ** (current_depth - 1) + jnp.floor(leaf_index / (2 ** (self.depth - (current_depth - 1)))) - 1).astype(jnp.int32)
                path_identifier_list.append(path_identifier)
                internal_node_index_list.append(internal_node_index)
        
        path_identifier_list = jnp.reshape(jnp.array(path_identifier_list, dtype=jnp.float32), (-1, self.depth))
        internal_node_index_list = jnp.reshape(jnp.array(internal_node_index_list, dtype=jnp.int32), (-1, self.depth))


        #jax.debug.print("path_identifier_list: {}", path_identifier_list)
        #jax.debug.print("internal_node_index_list: {}", internal_node_index_list)

        indices = {
            "features_by_estimator": features_by_estimator,
            "path_identifier_list": path_identifier_list,
            "internal_node_index_list": internal_node_index_list,
        }

        return indices

    @jax.jit
    def apply(self, params, inputs, indices):
        split_values = params["split_values"]
        estimator_weights = params["estimator_weights"]
        split_index_array = params["split_idx_array"]
        leaf_classes_array = params["leaf_array"]
        log_std = params["log_std"]

        features_by_estimator = indices["features_by_estimator"]
        path_identifier_list = indices["path_identifier_list"]
        internal_node_index_list = indices["internal_node_index_list"]

        # einsum syntax:
        #       - b is the batch size
        #       - e is the number of estimators
        #       - l the number of leaf nodes  (i.e. the number of paths)
        #       - i is the number of internal nodes
        #       - d is the depth (i.e. the length of each path)
        #       - n is the number of variables (one value is stored for each variable)

        X_estimator = inputs[:, features_by_estimator]

        # entmax transformaton
        split_index_array = entmax15JAX(split_index_array)

        # use ST-Operator to get one-hot encoded vector for feature index
        adjust_constant = split_index_array - jax.nn.one_hot(
            jnp.argmax(split_index_array, axis=-1), num_classes=split_index_array.shape[-1]
        )
        split_index_array = split_index_array - jax.lax.stop_gradient(adjust_constant)
        #jax.debug.print("split_index_array: {}", split_index_array)
        # as split_index_array_selected is one-hot-encoded, taking the sum over the last axis after multiplication results in selecting the desired value at the index
        s1_sum = jnp.einsum("ein,ein->ei", split_values, split_index_array)
        s2_sum = jnp.einsum("ben,ein->bei", X_estimator, split_index_array)
        #s2_sum = jnp.einsum("bn,ein->bei", inputs, split_index_array)
        
        # calculate the split (output shape: (b, e, i))
        node_result = (jax.nn.soft_sign(s1_sum - s2_sum) + 1) / 2
        adjust_constant = node_result - jnp.round(node_result)

        # use round operation with ST operator to get hard decision for each node
        node_result_corrected = node_result - jax.lax.stop_gradient(adjust_constant)

        # the resulting shape of the tensors is (b, e, l, d):
        node_result_extended = node_result_corrected[:, :, internal_node_index_list]
        #jax.debug.print("node_result_extended {}: {}", node_result_extended.shape, node_result_extended)
        # reduce the path via multiplication to get result for each path (in each estimator) based on the results of the corresponding internal nodes (output shape: (b, e, l))
        p = jnp.prod(
            ((1 - path_identifier_list) * node_result_extended + path_identifier_list * (1 - node_result_extended)),
            axis=3,
        )
        #jax.debug.print("p {}: {}", p.shape, p)
        # calculate instance-wise leaf weights for each estimator by selecting the weight of the selected path for each estimator
        estimator_weights_leaf = jnp.einsum("el,bel->be", estimator_weights, p)

        # use softmax over weights for each instance
        estimator_weights_leaf_softmax = jax.nn.softmax(estimator_weights_leaf)
        #jax.debug.print("estimator_weights_leaf_softmax {}: {}", estimator_weights_leaf_softmax.shape, estimator_weights_leaf_softmax)
        # get raw prediction for each estimator
        if self.action_type == "continuous":
            layer_output = jnp.einsum("elc,bel->bec", leaf_classes_array, p)
            layer_output = jnp.einsum("be,bec->bc", estimator_weights_leaf_softmax, layer_output)
            result = [layer_output, log_std]
        elif self.action_type == "discrete":
            layer_output = jnp.einsum("elc,bel->bec", leaf_classes_array, p)
            result = jnp.einsum("be,bec->bc", estimator_weights_leaf_softmax, layer_output)

        return result


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
