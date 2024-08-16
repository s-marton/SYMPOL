import gymnasium as gym
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from sdt import entmax15JAX
from gymnasium.spaces import Box
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import OneHotPartialObsWrapper, ViewSizeWrapper, ObservationWrapper, ActionBonus, PositionBonus

from functools import reduce
import operator
from gymnasium import spaces

import random
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.envs.distshift import DistShiftEnv
from gymnasium.envs.registration import register
import graphviz
import numpy as np
from gymnax.wrappers.purerl import GymnaxWrapper

import functools
from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment
from gymnax.environments import spaces as spaces_gymnax

import copy

OBSERVATION_LABELS = {
    'LunarLander-v2': ['x', 'y', 'velocity_x', 'velocity_y', 'angle', 'angular_velocity',
                       'leg_1_ground_contact', 'leg_2_ground_contact']
}


class NormalizeObservationWrapper(GymnaxWrapper):
    """Normalize the observations of the environment."""


    def __init__(self, env, params):
        super().__init__(env)

        self.original_low_no_clip = self._env.observation_space(params).low
        self.original_high_no_clip = self._env.observation_space(params).high
        self.original_low = jnp.clip(self._env.observation_space(params).low, -10, 10)
        self.original_high = jnp.clip(self._env.observation_space(params).high, -10, 10)
    
    def observation_space(self, params) -> spaces_gymnax.Box:
        assert isinstance(
            self._env.observation_space(params), spaces_gymnax.Box
        ), "Only Box spaces are supported for now."

        space = spaces_gymnax.Box(
            low=-0.5 + (self.original_low_no_clip - self.original_low) / (self.original_high - self.original_low),
            high=-0.5 + (self.original_high_no_clip - self.original_low) / (self.original_high - self.original_low),
            shape=self._env.observation_space(params).shape,
            dtype=self._env.observation_space(params).dtype,
        )
        print(-0.5 + (self.original_low_no_clip - self.original_low) / (self.original_high - self.original_low), self.original_high, self.original_low, self.original_low_no_clip)
        print(space.low, space.high)
        return space
        
    def normalize_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        return -0.5 + (obs - self.original_low) / (self.original_high - self.original_low)        

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = self.normalize_obs(obs)#jnp.reshape(obs, (-1,))
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, Any]:  # dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = self.normalize_obs(obs)#jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class RandomGoalDistShiftEnv(DistShiftEnv):
    def __init__(self, strip2_row=2, **kwargs):
        super().__init__(strip2_row=strip2_row, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        # Remove the old goal
        self.grid.set(width-2, 1, None)
        
        # Randomly place the goal somewhere in the grid
        while True:
            goal_x = self._rand_int(0, width)
            goal_y = self._rand_int(0, height)
            if self.grid.get(goal_x, goal_y) is None:
                self.grid.set(goal_x, goal_y, Goal())
                break

        self.mission = "Get to the green goal square"

# Register the environment with Gymnasium
register(
    id='MiniGrid-DistShift3-v0',
    entry_point='utils:RandomGoalDistShiftEnv',
)

class RandomGoalDistShiftEnv2(DistShiftEnv):
    def __init__(self, strip2_row=5, **kwargs):
        super().__init__(strip2_row=strip2_row, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        # Remove the old goal
        self.grid.set(width-2, 1, None)
        
        # Randomly place the goal somewhere in the grid
        while True:
            goal_x = self._rand_int(0, width)
            goal_y = self._rand_int(0, height)
            if self.grid.get(goal_x, goal_y) is None:
                self.grid.set(goal_x, goal_y, Goal())
                break

        self.mission = "Get to the green goal square"

# Register the environment with Gymnasium
register(
    id='MiniGrid-DistShift4-v0',
    entry_point='utils:RandomGoalDistShiftEnv2',
)

class FlatCurrentWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="float32",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]

        obs = image.flatten().astype(np.float32)
        obs = obs * 2 - 1 # convert to range -1,1 instead of 0,1
        return obs

"""
    Object Types
    Each object type in the MiniGrid environment is assigned a specific integer value. Here are the common object types:

    [0,1,2,8,9]
    
    
    
    unseen	    0 |  0
    empty	    1 |  1
    wall	    2 |  2
    floor	    3 |  3
    door	    4 |  4
    key	        5 |  5
    ball	    6 |  6
    box	        7 |  7
    goal	    8 |  8
    lava	    9 |  9
    agent      10 | 10
    -----------------------

    
    Colors
    Colors are also encoded as integer values. Here are the typical colors used in the MiniGrid environment:
    
    Color	Value
    Red  	0 | 11
    Green	1 | 12
    Blue	2 | 13
    Purple	3 | 14
    Yellow	4 | 15
    Grey	5 | 16
    -----------------------
    States
    The state value provides additional context about the object. For some objects, this might indicate whether they are open or closed, picked up, etc. Here are some common state values:

    State	Value
    Open	0 | 17
    Closed	1 | 18
    Locked	2 | 19

    
    | **Num** | **Name**   | 
    |:-------:|:----------:|
    | **0**   | **left**   |
    | **1**   | **right**  |
    | **2**   | **forward**|
    | **3**   | **pickup** |
    | **4**   | **drop**   |
    | **5**   | **toggle** |
    | **6**   | **done**   |
    
"""

class FlatCurrentReducedWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        
        self.select_indices = [0,1,2,8,9]
        # Define a mapping from environment names to select indices
        env_select_indices = {
            "DistShift": [0, 1, 2, 8, 9], # left, right, forward
            "LavaGap": [0, 1, 2, 8, 9], # left, right, forward
            "LavaCrossing": [0, 1, 2, 8, 9], # left, right, forward
            
            "SimpleCrossing": [0, 1, 2, 8], # left, right, forward
            "FourRooms": [0, 1, 2, 8], # left, right, forward
            "Empty": [0, 1, 2, 8], # left, right, forward
            
            "MultiRoom": [0, 1, 2, 4, 8, 17, 18], # left, right, forward, toggle

            "Dynamic-Obstacles": [0, 1, 2, 4, 6, 8], # left, right, forward

            "Unlock": [0, 1, 2, 4, 5, 8, 17, 18, 19], # left, right, forward, toggle #No pickup key
            "UnlockPickup": [0, 1, 2, 4, 5, 7, 8, 17, 18, 19], # left, right, forward, pickup, toggle #No pickup key

            "DoorKey": [0, 1, 2, 4, 5, 8, 17, 18, 19], # left, right, forward, pickup, toggle #Pickup key

            "GoToDoor": [0, 1, 2, 4, 8, 11, 12, 13, 14, 15, 16], # left, right, forward, done

            "RedBlueDoors": [0, 1, 2, 4, 8, 11, 13, 17, 18], # left, right, forward, toggle
            "PutNear": [0, 1, 2, 4, 8, 17, 18], # left, right, forward, pickup, drop 
        }

        # Get the environment name
        env_name = env.spec.id
        
        env_identifier = env_name
        for key in env_select_indices.keys():
            if key in env_name:
                env_identifier = key     

        # Set select_indices based on the environment name
        if env_identifier in env_select_indices:
            self.select_indices = env_select_indices[env_identifier]
            print(f"Environment {env_identifier} with Observations {self.select_indices}")
        else:
            raise ValueError(f"Environment {env_identifier} is not supported by this wrapper.")

        
        imgSize = imgSpace.shape[0] * imgSpace.shape[1] * len(self.select_indices) #reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="float32",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]
        #print('image.shape', image.shape)
        #print('image.flatten().shape', image.flatten().shape)
        obs = image[:,:,self.select_indices].flatten().astype(np.float32)
        obs = obs * 2 - 1 # convert to range -1,1 instead of 0,1

        #obs =
        #print('obs.shape', obs.shape)
        return obs

class NormalizeWrapperLunarLander(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)  
    
    def observation(self, obs):
        obs[0] = (obs[0] - 0) / 1.5
        obs[1] = (obs[1] - 0) / 1.5
        obs[2] = (obs[2] - 0) / 5.0
        obs[3] = (obs[3] - 0) / 5.0
        obs[4] = (obs[4] - 0) / 3.14
        obs[5] = (obs[5] - 0) / 5.0
        obs[6] = (obs[6] - 1) / 0.5
        obs[7] = (obs[7] - 1) / 0.5
        
        return obs

class AutoResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AutoResetWrapper, self).__init__(env)
        self.reset_on_step = False
        self.reset_env()
        
    def reset_env(self):
        # Generate a new random seed
        seed = random.randint(0, 1000000)
        self.observation, self.info = self.env.reset(seed=seed)

    def step(self, action):
        if self.reset_on_step:
            self.reset_env()
            self.reset_on_step = False

        observation, reward, done, truncated, info = self.env.step(action)
        if done or truncated:
            self.reset_on_step = True
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self.reset_env()
        return self.observation, self.info



def build_env(env_id, n_env, view_size=3):
    if n_env > 1: 
        env = gym.make(id=env_id) #, render_mode="rgb_array")

    else:
        env = gym.make(id=env_id, render_mode="rgb_array") #, render_mode="rgb_array")

    if 'MiniGrid' in env_id:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
        env = OneHotPartialObsWrapper(env)
        env = FlatCurrentReducedWrapper(env) 
    elif 'LunarLander' in env_id:
        env = NormalizeWrapperLunarLander(env)
        
    if n_env > 1:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.vector.AsyncVectorEnv([lambda: env for _ in range(n_env)])
    return env




class ActorTrainState(TrainState):
    grad_accum: jnp.ndarray
    indices: dict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

@flax.struct.dataclass
class ObservationActionBuffer:
    obs: jnp.array
    actions: jnp.array

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

def convert_to_discrete_tree(params, action_type, temperature=1.0):
    """
    Convert a trained soft decision tree (SDT) into a discrete decision tree.

    Args:
        params (dict): The parameters of the trained SDT.

    Returns:
        dict: The parameters of the discrete decision tree.
    """
    # Create a deep copy of the parameters to avoid modifying the original parameters
    new_params = unfreeze(copy.deepcopy(params))

    beta = new_params['params']['SDT_0']['inner_nodes']['layers_0']['kernel']
    beta = entmax15JAX(beta.T / temperature).T
    
    #print('beta', beta)
    phi = new_params['params']['SDT_0']['inner_nodes']['layers_0']['bias']

    # Obtain the index of the feature to use
    j = jnp.argmax(beta, axis=0)

    one_hot_beta = jax.nn.one_hot(
                j, num_classes=beta.shape[0]
            ).T

    # Normalize phi
    #print('beta', beta)
    #print('jnp.sum(beta * one_hot_beta, axis=0)', jnp.sum(beta * one_hot_beta, axis=0))
    normalized_phi = phi / jnp.sum(beta * one_hot_beta, axis=0)
    #print('one_hot_beta', one_hot_beta)
    #print('jnp.sum(beta * one_hot_beta, axis=-1)', jnp.sum(beta * one_hot_beta, axis=0))
    
    # Update params
    new_params['params']['SDT_0']['inner_nodes']['layers_0']['kernel'] = one_hot_beta
    new_params['params']['SDT_0']['inner_nodes']['layers_0']['bias'] = normalized_phi

    if action_type == 'discrete':
        beta_leaf = new_params['params']['SDT_0']['leaf_nodes']['kernel']
        
        # Obtain the index of the feature to use
        j = jnp.argmax(beta_leaf, axis=1)
        
        # Create one-hot vector for beta
        one_hot_beta_leaf = jax.nn.one_hot(
                    j, num_classes=beta_leaf.shape[1]
                )
        
        # Update params
        new_params['params']['SDT_0']['leaf_nodes']['kernel'] = one_hot_beta_leaf
    else:
        log_std = new_params['params']['SDT_0']['log_std']
        
        new_params['params']['SDT_0']['log_std'] = jnp.zeros_like(log_std)
    return freeze(new_params)

def prune_and_merge_tree(node, split_ranges, constraints=None, continuous=False):
    """
    Prune a decision tree based on predefined ranges for each split index, merge leaf nodes with the same distribution,
    and remove redundant paths that cannot be taken because previous splits already predetermine the path.

    Args:
    - node (dict): The decision tree node (root node initially).
    - split_ranges (dict): A dictionary where keys are split indices and values are tuples of (min_value, max_value).
    - constraints (dict): A dictionary to keep track of constraints on split indices.

    Returns:
    - dict: The pruned and merged decision tree or the subtree if the current node is pruned.
    """
    if constraints is None:
        constraints = {}

    if node['type'] == 'leaf':
        return node

    split_index = node['split_index']
    split_value = node['split_value']

    if split_index in split_ranges:
        min_value, max_value = split_ranges[split_index]
        if split_value < min_value or split_value > max_value:
            # If the split value is outside the range, prune this node
            # Return left child if it exists, otherwise right child if it exists, else None
            if split_value < min_value:# and node['right_child']['type'] != 'leaf':
                if node['right_child']['type'] != 'leaf':
                    return prune_and_merge_tree(node['right_child'], split_ranges, constraints, continuous=continuous)            
                else:
                    return node['right_child']
            elif split_value > max_value:# and node['left_child']['type'] != 'leaf':
                if node['left_child']['type'] != 'leaf':
                    return prune_and_merge_tree(node['left_child'], split_ranges, constraints, continuous=continuous)
                else:
                    return node['left_child']
            else:
                return None#node['left_child'] if node['left_child'] else node['right_child']
    
    # Check if the current split is redundant based on constraints
    if split_index in constraints:
        min_constraint, max_constraint = constraints[split_index]
        if (split_value >= min_constraint) or (split_value <= max_constraint):
            # The split is redundant, remove this node and move its child up
            if split_value <= max_constraint:# and node['right_child']['type'] != 'leaf':
                if node['right_child']['type'] != 'leaf':
                    return prune_and_merge_tree(node['right_child'], split_ranges, constraints, continuous=continuous)            
                else:
                    return node['right_child']
            elif split_value >= min_constraint:# and node['left_child']['type'] != 'leaf':
                if node['left_child']['type'] != 'leaf':
                    return prune_and_merge_tree(node['left_child'], split_ranges, constraints, continuous=continuous)
                else:
                    return node['left_child']
            else:
                return None#node['left_child'] if node['left_child'] else node['right_child']
    

    # Update constraints based on the current split
    new_constraints_left = constraints.copy()
    new_constraints_right = constraints.copy()
    if split_index in new_constraints_left:
        new_constraints_left[split_index] = (min(split_value, new_constraints_left[split_index][0]), new_constraints_left[split_index][1])
    else:
        new_constraints_left[split_index] = (split_value, -np.inf)
    if split_index in new_constraints_right:
        new_constraints_right[split_index] = (new_constraints_right[split_index][0], max(split_value, new_constraints_right[split_index][1]))
    else:
        new_constraints_right[split_index] = (np.inf, split_value)

    #print(new_constraints_left, new_constraints_right)
    # Recursively prune and merge left and right children
    node['left_child'] = prune_and_merge_tree(node['left_child'], split_ranges, new_constraints_left, continuous=continuous)
    node['right_child'] = prune_and_merge_tree(node['right_child'], split_ranges, new_constraints_right, continuous=continuous)
  
    # If both children are leaves with the same distribution, merge them
    if (not continuous and 
        node['left_child'] and node['left_child']['type'] == 'leaf' and
        node['right_child'] and node['right_child']['type'] == 'leaf' and
        #node['left_child']['distribution'] == node['right_child']['distribution']):
        np.argmax(node['left_child']['distribution']) == np.argmax(node['right_child']['distribution'])):
        return node['left_child']

    # If both children are pruned, prune this node too
    if node['left_child'] is None and node['right_child'] is None:
        return None

    return node

import graphviz
import numpy as np
from IPython.display import Image

def convert_to_child_representation(split_values, split_indices, leaf_values, features_by_estimator):
    num_internal_nodes = split_values.shape[0]
    num_leaf_nodes = leaf_values.shape[0]

    def build_tree(node_id):
        if node_id >= num_internal_nodes:
            leaf_index = node_id - num_internal_nodes
            leaf_dist = leaf_values[leaf_index]
            return {
                'type': 'leaf',
                'action': leaf_dist,#np.argmax(leaf_dist),
                'distribution': leaf_dist.tolist()
            }
        else:
            split_index = np.argmax(split_indices[node_id])
            split_index = features_by_estimator[split_index]
            split_value = split_values[node_id, split_index]
            if np.round(split_value) == 1 and split_value < 1:
                split_value = 0.99
            elif np.round(split_value) == -1 and split_value > -1:
                split_value = -0.99

            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2

            return {
                'type': 'internal',
                'split_index': int(split_index),
                'split_value': float(split_value),
                'left_child': build_tree(left_child_id),
                'right_child': build_tree(right_child_id)
            }

    return build_tree(0)

def plot_tree_from_representation(tree, image_path, filename_appendix='', observation_labels=None, continuous=False):
    def add_nodes_edges(tree, dot=None):
        if dot is None:
            dot = graphviz.Digraph()

        def traverse(node, parent=None):
            if node['type'] == 'leaf':
                label = f"Action: {node['action']}" if continuous else f"Action: {np.argmax(node['action'])}"
                node_id = str(id(node))
                dot.node(node_id, label, shape="box")
            else:
                if np.round(node['split_value']) == 1 and node['split_value'] < 1:
                    node['split_value'] = 0.99
                elif np.round(node['split_value']) == -1 and node['split_value'] > -1:
                    node['split_value'] = -0.99
                if observation_labels is not None:
                    label = f"{observation_labels[node['split_index']]} <= {node['split_value']:.2f}?"
                else:
                    label = f"X{node['split_index']} <= {node['split_value']:.2f}?"                
                node_id = str(id(node))
                dot.node(node_id, label)
                traverse(node['left_child'], node_id)
                dot.edge(node_id, str(id(node['left_child'])), label="True")
                traverse(node['right_child'], node_id)
                dot.edge(node_id, str(id(node['right_child'])), label="False")

        traverse(tree)
        return dot

    dot = add_nodes_edges(tree)
    image_path = image_path + filename_appendix
    dot.render(image_path, format='png', cleanup=True)
    return image_path



def count_nodes(tree):
    if tree['type'] == 'leaf':
        return {'internal': 0, 'leaf': 1}
    
    left_counts = count_nodes(tree['left_child'])
    right_counts = count_nodes(tree['right_child'])
    
    return {
        'internal': 1 + left_counts['internal'] + right_counts['internal'],
        'leaf': left_counts['leaf'] + right_counts['leaf']
    }

def plot_decision_tree(split_values, split_indices, leaf_values, features_by_estimator, image_path, observation_labels=None, filename_appendix='', env=None, env_params=None, prune=True, continuous=False):

    tree_representation = convert_to_child_representation(split_values, split_indices, leaf_values, features_by_estimator)
    if prune:
        if env_params is not None:
            import gymnax
            env_name = env.name
            observation_space = env.observation_space(env_params)
            if isinstance(observation_space, gymnax.environments.spaces.Box):
                ranges_dict = {}
                for i, range_tuple in enumerate(np.vstack([observation_space.low, observation_space.high]).T):
                    ranges_dict[i] = list(np.asarray(range_tuple))
                print(ranges_dict)
            elif isinstance(observation_space, gymnax.environments.spaces.Discrete):
                ranges_dict = {}
                for i in range(observation_space.n):
                    ranges_dict[i] = [0,1]       
                print(ranges_dict)
            else:
                print("Observation Space type is not handled in this snippet.")
        else:
            observation_space = env.observation_space
            env_name = env.unwrapped.spec.id
            if 'MiniGrid' in env_name:
                ranges_dict = {}
                for i, range_tuple in enumerate(np.vstack([observation_space.low, observation_space.high]).T):
                    ranges_dict[i] = [-1,1]      
                print(ranges_dict)            
            else:
                if isinstance(observation_space, gym.spaces.Box):
                    ranges_dict = {}
                    for i, range_tuple in enumerate(np.vstack([observation_space.low, observation_space.high]).T):
                        ranges_dict[i] = list(range_tuple)
                    print(ranges_dict)
                elif isinstance(observation_space, gym.spaces.Discrete):
                    ranges_dict = {}
                    for i in range(observation_space.n):
                        ranges_dict[i] = [0,1]      
                    print(ranges_dict)
                else:           
                    print("Observation Space type is not handled in this snippet.")
        tree_representation = prune_and_merge_tree(tree_representation, ranges_dict, continuous=continuous)
    node_count = count_nodes(tree_representation)
    plot_path = plot_tree_from_representation(tree_representation, image_path, filename_appendix='', observation_labels=observation_labels, continuous=continuous)
    
    return plot_path, node_count['internal'] + node_count['leaf']

import graphviz
import numpy as np
from IPython.display import Image

def convert_to_child_representation_soft(split_values, split_indices, leaf_values, temperature):
    num_internal_nodes = split_values.shape[0]
    num_leaf_nodes = leaf_values.shape[0]

    def build_tree(node_id):
        if node_id >= num_internal_nodes:
            leaf_index = node_id - num_internal_nodes
            leaf_dist = leaf_values[leaf_index]
            return {
                'type': 'leaf',
                'action': np.argmax(leaf_dist),
                'distribution': leaf_dist.tolist()
            }
        else:
            
            split_index = entmax15JAX(split_indices[node_id].T / temperature).T
            split_value = split_values[node_id]

            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2

            return {
                'type': 'internal',
                'split_index': split_index,
                'split_value': split_value,
                'left_child': build_tree(left_child_id),
                'right_child': build_tree(right_child_id)
            }

    return build_tree(0)

def plot_tree_from_representation_soft(tree, image_path, filename_appendix='', observation_labels=None):
    def add_nodes_edges(tree, dot=None):
        if dot is None:
            dot = graphviz.Digraph()

        def traverse(node, parent=None):
            if node['type'] == 'leaf':
                label = f"Action: {node['action']}"
                node_id = str(id(node))
                dot.node(node_id, label, shape="box")
            else:
                if np.round(node['split_value']) == 1 and node['split_value'] < 1:
                    node['split_value'] = 0.99
                elif np.round(node['split_value']) == -1 and node['split_value'] > -1:
                    node['split_value'] = -0.99
                label = f"{np.round(node['split_index'], 2)} - {node['split_value']:.2f}?"                
                node_id = str(id(node))
                dot.node(node_id, label)
                traverse(node['left_child'], node_id)
                dot.edge(node_id, str(id(node['left_child'])), label="True")
                traverse(node['right_child'], node_id)
                dot.edge(node_id, str(id(node['right_child'])), label="False")

        traverse(tree)
        return dot

    dot = add_nodes_edges(tree)
    image_path = image_path + filename_appendix
    dot.render(image_path, format='png', cleanup=True)
    return image_path


def plot_decision_tree_soft(split_values, split_indices, leaf_values, image_path, observation_labels=None, filename_appendix='', temperature=1.0):

    tree_representation = convert_to_child_representation_soft(split_values, split_indices, leaf_values, temperature)

    node_count = count_nodes(tree_representation)
    plot_path = plot_tree_from_representation_soft(tree_representation, image_path, filename_appendix='', observation_labels=observation_labels)
    
    return plot_path, node_count['internal'] + node_count['leaf']
