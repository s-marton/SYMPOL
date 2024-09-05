# Code mostly taken from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import flax
from distrax import Normal, MultivariateNormalDiag
from flax import linen as nn
from functools import partial

from flax.training.train_state import TrainState
import optax
import optuna
import functools
import os
import random
from typing import Any
#from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt

import datetime

from utils import build_env, ActorTrainState, EpisodeStatistics, Storage, ObservationActionBuffer, convert_to_discrete_tree, plot_decision_tree, plot_decision_tree_soft, NormalizeObservationWrapper, OBSERVATION_LABELS

from args import get_args
import configs
from sympol import SYMPOL_RL
from mlp import Critic_MLP, Actor_MLP, Actor_MLP_Continuous
from sdt import Critic_SDT, Actor_SDT


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import distrax

import graphviz
import wandb
import pickle
import gymnax
import time

import jax
import jax.numpy as jnp
from jax import lax
from gymnax.wrappers import gym as wrappers
from gymnax.wrappers.purerl import FlattenObservationWrapper, GymnaxWrapper
from gymnasium.wrappers import FlattenObservation

#os.environ['MUJOCO_GL'] = 'egl'


# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"
#os.environ["XLA_FLAGS"] = "--xla_dump_to=~/tmp/foo"

# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
#os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
#os.environ["TF_CUDNN DETERMINISTIC"] = "1"

def train_agent(args, trial=None, queue=None):
    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_number)
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if trial is not None:
        timestamp = 'TRIAL_NO' + str(trial.number) + '_' + timestamp
        
        if args.actor == 'mlp':         
            suggested_params = configs.suggest_config_mlp(trial, args.env_id)
        elif args.actor == 'sympol':         
            suggested_params = configs.suggest_config_sympol(trial, args.env_id)
        elif args.actor == 'sdt':         
            suggested_params = configs.suggest_config_sdt(trial, args.env_id)
        elif args.actor == 'd-sdt':         
            suggested_params = configs.suggest_config_dsdt(trial, args.env_id)
        elif args.actor == 'stateActionDT':         
            suggested_params = configs.suggest_config_stateActionDT(trial, args.env_id)            
        else:
            suggested_params = {}
        
        args.__dict__.update(suggested_params)    

        n_steps = args.n_steps
        
    elif not args.use_best_config:
        if False:
            if args.env_id == 'Hopper-v4':
                n_steps = 512
                args.n_envs = 2
            elif args.env_id == 'CartPole-v1':
                n_steps = 32
                args.n_envs = 8   
            elif args.env_id == 'Pendulum-v1':
                n_steps = 1024
                args.n_envs = 4
            elif args.env_id == 'BipedalWalker-v3':
                n_steps = 512
                args.n_envs = 16          
            elif args.env_id == 'LunarLander-v2':
                n_steps = 512
                args.n_envs = 8   
        else:
            n_steps = 512
            args.n_envs = 8 
    else:
        n_steps = args.n_steps
            
    if args.dynamic_buffer:
        n_steps = max(16, n_steps // 8)
        
    accumulate_gradients_every = args.accumulate_gradients_every
    accumulate_gradients_every_initial = accumulate_gradients_every
            
    initial_steps = n_steps
    # these parameters are defined dynamically
    batch_size = int(args.n_envs * n_steps)
    #minibatch_size = int(batch_size // args.n_minibatches)
    minibatch_size = args.minibatch_size
    while batch_size // minibatch_size < 2:
        minibatch_size = minibatch_size // 2
    #n_iterations = args.total_steps // batch_size
    #eval_freq = max(args.eval_freq // batch_size, 1)
    
    trial_scores = []
    for random_trial_number in range(1, args.random_trials+1):
        
        if args.actor == 'sympol':
            model_identifier = '-'.join([str(args.depth), str(args.n_estimators), str(args.seed)])
        elif args.actor != 'mlp':
            model_identifier = '-'.join([str(args.depth),  str(args.seed)])            
        else:
            model_identifier = str(args.seed)
            
        run_name = '-'.join([args.run_name, args.actor, str(np.round(args.learning_rate_actor, 6)), model_identifier, timestamp])

        group_name = run_name
        run_name = run_name + '_' + str(random_trial_number)
        
        #envs = build_env(args.env_id, n_env=args.n_envs)
        #args.n_envs = 1
        if not args.env_id in gymnax.registered_envs:
            import sys
            sys.exit("Environment not implemented in gymnax")
            
        envs, env_params = gymnax.make(args.env_id)
        if args.normEnv:
            print('NORMALIZE')
            envs = NormalizeObservationWrapper(envs, env_params)
        vmap_reset = jax.vmap(envs.reset, in_axes=(0, None))
        vmap_step = jax.vmap(envs.step, in_axes=(0, 0, 0, None))

        obs_dim = envs.observation_space(env_params).shape[-1]#envs.obs_shape[-1] #envs.single_observation_space.shape[-1]
        obs_shape = envs.observation_space(env_params).shape
        print('Observations:', obs_dim)

        if isinstance(envs.action_space(), gymnax.environments.spaces.Discrete):         
            action_dim = envs.action_space().n
        elif isinstance(envs.action_space(), gymnax.environments.spaces.Box):
            action_dim = envs.action_space().shape[-1]
        print('Actions:', action_dim)

        if args.track:
            wandb_run = wandb.init(
                project=f"{args.exp_name}_{args.env_id}",
                group=group_name,
                tags=[args.run_name],
                #sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True, 
            )
        env_seed = args.seed + (random_trial_number * 100)
        env_key = jax.random.PRNGKey(env_seed)
        
        if True:
            seed_training = args.seed + (random_trial_number * 100)
        else:
            seed_training = args.seed
            
        key = jax.random.PRNGKey(seed_training)
        model_key = jax.random.PRNGKey(args.seed)
        model_key, actor_key, critic_key = jax.random.split(model_key, 3)
    
        # agent setup
        if args.critic == "mlp":
            critic = Critic_MLP(num_layers=args.num_layers, neurons_per_layer=args.neurons_per_layer)
        elif args.critic == "sdt":
            critic = Critic_SDT(depth=args.depth, temperature=args.temperature)#, temp=1)
        critic.apply = jax.jit(critic.apply)
        if args.actor == "mlp" or args.actor == "stateActionDT":
            if args.action_type == "discrete":
                actor = Actor_MLP(action_dim=action_dim, num_layers=args.num_layers, neurons_per_layer=args.neurons_per_layer)
            else:
                actor = Actor_MLP_Continuous(action_dim=action_dim, num_layers=args.num_layers, neurons_per_layer=args.neurons_per_layer)
            actor.apply = jax.jit(actor.apply)

            #args.learning_rate_actor = args.learning_rate_critic  # same lr for MLP's
            args.accumulate_gradients_every = 1  # do not accumulate gradients for MLP's
            learning_rate_actor = args.learning_rate_actor
        elif args.actor == "sympol":
            learning_rate_actor_weights = args.learning_rate_actor_weights
            learning_rate_actor_split_values = args.learning_rate_actor_split_values
            learning_rate_actor_split_idx_array = args.learning_rate_actor_split_idx_array
            learning_rate_actor_leaf_array = args.learning_rate_actor_leaf_array
            learning_rate_actor_log_std = args.learning_rate_actor_log_std

            
            actor = SYMPOL_RL(
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=args.action_type,
                depth=args.depth,
                n_estimators=args.n_estimators,
            )
        elif args.actor == "sdt" or args.actor == "d-sdt":
            actor = Actor_SDT(action_dim=action_dim, depth=args.depth, temperature=args.temperature, action_type=args.action_type)#, temp=1)
            actor.apply = jax.jit(actor.apply)

            #args.learning_rate_actor = args.learning_rate_critic  # same lr for SDT's
            learning_rate_actor = args.learning_rate_actor
            args.accumulate_gradients_every = 1  # do not accumulate gradients for SDT's
            
        if args.adamW:
            critic_state = TrainState.create(
                apply_fn=None,
                params=critic.init(critic_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(learning_rate=args.learning_rate_critic)
                ),
            )
        else:
            critic_state = TrainState.create(
                apply_fn=None,
                params=critic.init(critic_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm), optax.adam(learning_rate=args.learning_rate_critic)
                ),
            )
    
        if args.actor == "sympol":
            def map_nested_fn(fn):
                '''Recursively apply `fn` to key-value pairs of a nested dict.'''
                def map_fn(nested_dict):
                    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                        for k, v in nested_dict.items()}
                return map_fn
            if args.SWA:    
                from optax_swag import swag
                if args.adamW:
                    actor_state = ActorTrainState.create(
                        apply_fn=None,
                        params=actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                        tx=optax.chain(
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.multi_transform(
                                {'estimator_weights': optax.inject_hyperparams(optax.adam)(learning_rate_actor_weights), 
                                 'split_values': optax.inject_hyperparams(optax.adam)(learning_rate_actor_split_values), 
                                 'split_idx_array': optax.inject_hyperparams(optax.adamw)(learning_rate_actor_split_idx_array), 
                                 'leaf_array': optax.inject_hyperparams(optax.adamw)(learning_rate_actor_leaf_array), 
                                 'log_std': optax.inject_hyperparams(optax.adamw)(learning_rate_actor_log_std),}, 
                                map_nested_fn(lambda k, _: k)),
                                swag(10, 2),
                        ),
                        grad_accum=jax.tree.map(
                            jnp.zeros_like, actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)]))
                        ),
                        indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                    )
                else:
                    actor_state = ActorTrainState.create(
                        apply_fn=None,
                        params=actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                        tx=optax.chain(
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.multi_transform(
                                {'estimator_weights': optax.inject_hyperparams(optax.adam)(learning_rate_actor_weights), 
                                 'split_values': optax.inject_hyperparams(optax.adam)(learning_rate_actor_split_values), 
                                 'split_idx_array': optax.inject_hyperparams(optax.adam)(learning_rate_actor_split_idx_array), 
                                 'leaf_array': optax.inject_hyperparams(optax.adam)(learning_rate_actor_leaf_array), 
                                 'log_std': optax.inject_hyperparams(optax.adam)(learning_rate_actor_log_std),}, 
                                map_nested_fn(lambda k, _: k)),
                                swag(10, 2),
                        ),
                        grad_accum=jax.tree.map(
                            jnp.zeros_like, actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)]))
                        ),
                        indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                    )                    
            else:
                if args.adamW:
                    actor_state = ActorTrainState.create(
                        apply_fn=None,
                        params=actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                        tx=optax.chain(
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.multi_transform(
                                {'estimator_weights': optax.inject_hyperparams(optax.adam)(learning_rate_actor_weights), 
                                 'split_values': optax.inject_hyperparams(optax.adam)(learning_rate_actor_split_values), 
                                 'split_idx_array': optax.inject_hyperparams(optax.adamw)(learning_rate_actor_split_idx_array), 
                                 'leaf_array': optax.inject_hyperparams(optax.adamw)(learning_rate_actor_leaf_array), 
                                 'log_std': optax.inject_hyperparams(optax.adamw)(learning_rate_actor_log_std),}, 
                                map_nested_fn(lambda k, _: k)),
                        ),
                        grad_accum=jax.tree.map(
                            jnp.zeros_like, actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)]))
                        ),
                        indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                    )
                else:
                    actor_state = ActorTrainState.create(
                        apply_fn=None,
                        params=actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                        tx=optax.chain(
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.multi_transform(
                                {'estimator_weights': optax.inject_hyperparams(optax.adam)(learning_rate_actor_weights), 
                                 'split_values': optax.inject_hyperparams(optax.adam)(learning_rate_actor_split_values), 
                                 'split_idx_array': optax.inject_hyperparams(optax.adam)(learning_rate_actor_split_idx_array), 
                                 'leaf_array': optax.inject_hyperparams(optax.adam)(learning_rate_actor_leaf_array), 
                                 'log_std': optax.inject_hyperparams(optax.adam)(learning_rate_actor_log_std),}, 
                                map_nested_fn(lambda k, _: k)),
                        ),
                        grad_accum=jax.tree.map(
                            jnp.zeros_like, actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)]))
                        ),
                        indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                    ) 
                    
        else:
            if args.adamW:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.inject_hyperparams(optax.adamw)(learning_rate_actor),
                    ),
                    grad_accum=jax.tree.map(
                        jnp.zeros_like, actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)]))
                    ),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )  
            else:
                actor_state = ActorTrainState.create(
                        apply_fn=None,
                        params=actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)])),
                        tx=optax.chain(
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.inject_hyperparams(optax.adam)(learning_rate_actor),
                        ),
                        grad_accum=jax.tree.map(
                            jnp.zeros_like, actor.init(actor_key, jnp.array([envs.observation_space(env_params).sample(key)]))
                        ),
                        indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                    )  
                    

        lr_scheduler = optax.contrib.reduce_on_plateau(patience=3, factor=0.5)
        lr_scheduler_state = lr_scheduler.init(actor_state.params)
        
        #actor.apply = jax.jit(actor.apply)
        #critic.apply = jax.jit(critic.apply)
            
        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        )
            
        @jax.jit
        def get_action_and_value(
            actor_state: TrainState,
            critic_state: TrainState,
            next_obs: np.ndarray,
            next_done: np.ndarray,
            storage: Storage,
            step: int,
            key: jax.random.PRNGKey,
        ):
            """sample action, calculate value, logprob, entropy, and update storage"""
            #jax.debug.print("next_obs: {}", next_obs.shape)
            if args.action_type == "discrete":
                action_logits = actor.apply(actor_state.params, next_obs, indices=actor_state.indices)
                action_distribution = distrax.Categorical(logits=action_logits)
                value = critic.apply(critic_state.params, next_obs)
        
                # Sample discrete actions from Normal distribution
                key, subkey = jax.random.split(key)
                action = action_distribution.sample(seed=subkey)

                logprob = action_distribution.log_prob(action)#.sum(-1)
                storage = storage.replace(
                    obs=storage.obs.at[step].set(next_obs),
                    dones=storage.dones.at[step].set(next_done),
                    actions=storage.actions.at[step].set(action),
                    logprobs=storage.logprobs.at[step].set(logprob),
                    values=storage.values.at[step].set(value.squeeze()),
                )
            else:
                
                result = actor.apply(actor_state.params, next_obs, indices=actor_state.indices)
                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))
                                                  
                value = critic.apply(critic_state.params, next_obs)

                # Sample continuous actions from Normal distribution
                key, subkey = jax.random.split(key)
                action = action_distribution.sample(seed=subkey)
                #jax.debug.print("action: {}", action.shape)
                logprob = action_distribution.log_prob(action)#.sum(-1)

                #jax.debug.print("action_distribution: {}", action_distribution)
                #jax.debug.print("action: {}", action)
                
                storage = storage.replace(
                    obs=storage.obs.at[step].set(next_obs),
                    dones=storage.dones.at[step].set(next_done),
                    actions=storage.actions.at[step].set(action),
                    logprobs=storage.logprobs.at[step].set(logprob),
                    values=storage.values.at[step].set(value.squeeze()),
                )

            return storage, action, key


        
        @jax.jit
        def get_action_and_value2(
            actor_state_params: flax.core.FrozenDict,
            critic_state_params: flax.core.FrozenDict,
            x: np.ndarray,
            action: np.ndarray,
        ):
            """calculate value, logprob of supplied `action`, and entropy"""

            if args.action_type == "discrete":
                logits = actor.apply(actor_state_params, x, indices=actor_state.indices)
                value = critic.apply(critic_state_params, x).squeeze()
                
                action_distribution = distrax.Categorical(logits=logits)
                logprob = action_distribution.log_prob(action)
                entropy = action_distribution.entropy()
            else:
                result = actor.apply(actor_state_params, x, indices=actor_state.indices)
                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))
                
                value = critic.apply(critic_state_params, x).squeeze()
                logprob = action_distribution.log_prob(action)
                entropy = action_distribution.entropy()

            return logprob, entropy, value
         
        def compute_gae_once(carry, inp, gamma, gae_lambda):
            advantages = carry
            nextdone, nextvalues, curvalues, reward = inp
            nextnonterminal = 1.0 - nextdone
    
            delta = reward + gamma * nextvalues * nextnonterminal - curvalues
            advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
            return advantages, advantages
    
        compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)
        

        def compute_gae(
            critic_state: TrainState,
            next_obs: np.ndarray,
            next_done: np.ndarray,
            storage: Storage,
        ):
            next_value = critic.apply(critic_state.params, next_obs).squeeze()
    
            advantages = jnp.zeros((args.n_envs,))
            dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
            values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
            _, advantages = jax.lax.scan(
                compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
            )
            storage = storage.replace(
                advantages=advantages,
                returns=advantages + storage.values,
            )
            return storage
            
        @jax.jit
        def ppo_loss_base(actor_state_params, critic_state_params, x, a, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(actor_state_params, critic_state_params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()
    
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    
            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    
            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
    
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))
        ppo_loss_base_grad_fn = jax.value_and_grad(ppo_loss_base, argnums=(0, 1), has_aux=True)
        
        
        @jax.jit
        def update_ppo(
            actor_state: TrainState,
            critic_state: TrainState,                
            storage: Storage,
            key: jax.random.PRNGKey,
            accumulate_gradients_every: int,
        ):
            def update_epoch(carry, unused_inp):
                actor_state, critic_state, key = carry
                key, subkey = jax.random.split(key)
    
                def flatten(x):
                    return x.reshape((-1,) + x.shape[2:])
    
                # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
                def convert_data(x: jnp.ndarray):
                    num_minibatches = int(np.floor(x.shape[0] / minibatch_size))
                    size = num_minibatches * minibatch_size
                    x = jax.random.permutation(subkey, x)[:size]
                    x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
                    return x
    
                flatten_storage = jax.tree_map(flatten, storage)
                shuffled_storage = jax.tree_map(convert_data, flatten_storage)
    
                def update_minibatch(carry, minibatch):
                    actor_state, critic_state = carry
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (actor_grads, critic_grads) = ppo_loss_base_grad_fn(
                        actor_state.params,
                        critic_state.params,              
                        minibatch.obs,
                        minibatch.actions,
                        minibatch.logprobs,
                        minibatch.advantages,
                        minibatch.returns,
                    )
                    critic_state = critic_state.apply_gradients(grads=critic_grads)
                    actor_grad_accum = jax.tree_util.tree_map(lambda x, y: x + y, actor_grads, actor_state.grad_accum)
                    actor_state = actor_state.apply_gradients(grads=actor_grads)
            
                    def update_fn():
                        grads = jax.tree_util.tree_map(lambda x: x / accumulate_gradients_every, actor_grad_accum)
                        new_state = actor_state.apply_gradients(
                            grads=grads,
                            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
                        )
                        return new_state
            
                    actor_state = jax.lax.cond(
                        actor_state.step % accumulate_gradients_every == 0,
                        lambda _: update_fn(),
                        lambda _: actor_state.replace(grad_accum=actor_grad_accum, step=actor_state.step + 1),
                        None,
                    )
                    
                    return (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_accum)
                
                (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_accum) = jax.lax.scan(
                    update_minibatch, (actor_state, critic_state), shuffled_storage
                )
                return (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_accum)
    
            (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_accum) = jax.lax.scan(
                update_epoch, (actor_state, critic_state, key), (), length=args.n_update_epochs
            )
            return actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key
   

        #@jax.jit
    
        global_step = 0
        env_key, key_reset = jax.random.split(env_key, 2)
        #next_obs, env_state = envs.reset(key_reset, env_params)
        vmap_keys = jax.random.split(key_reset, args.n_envs)
        next_obs, env_state = vmap_reset(vmap_keys, env_params)
        
        next_done = np.zeros(args.n_envs).astype(bool)
        #####jax.debug.print("next_obs INIT {}: {}", next_obs.shape, next_obs)
    
        #env_state, env_params, 
        #@jax.jit
        # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py

        
        #@jax.jit
        def create_rollout_gymnax(n_steps):
                
            def rollout_gymnax_(actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, global_step):
                #for step in range(0, n_steps):
                step = 0
                key, rng_step = jax.random.split(key, 2)
                step_keys = jax.random.split(rng_step, args.n_envs)                
                def policy_step(state_input, tmp):
            
                    actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, step_keys, global_step, step = state_input
                    
                    global_step += args.n_envs
                    
                    storage, action, key = get_action_and_value(
                        actor_state, critic_state, next_obs, next_done, storage, step, key,  
                    )
                    # TRY NOT TO MODIFY: execute the game and log data.
                    #print('ACTION', jax.device_get(action))
                    #next_obs, reward, next_done, trunc, info = envs.step(jax.device_get(action))
            

                    #next_obs, env_state, reward, next_done, _ = envs.step(
                    #    rng_step, env_state, action, env_params
                    #)
                    next_obs, env_state, reward, next_done, _ = vmap_step(step_keys, env_state, action, env_params)
                    
                    #print('STEP', next_obs, reward, next_done, trunc, info)
                    new_episode_return = episode_stats.episode_returns + reward
                    new_episode_length = episode_stats.episode_lengths + 1
                    episode_stats = episode_stats.replace(
                        episode_returns=(new_episode_return) * (1 - next_done), #* (1 - trunc),
                        episode_lengths=(new_episode_length) * (1 - next_done), #* (1 - trunc),
                        # only update the `returned_episode_returns` if the episode is done
                        returned_episode_returns=jnp.where(
                            next_done,# + trunc,
                            new_episode_return,
                            episode_stats.returned_episode_returns,
                        ),
                        returned_episode_lengths=jnp.where(
                            next_done,# + trunc,
                            new_episode_length,
                            episode_stats.returned_episode_lengths,
                        ),
                    )
                    storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
                    step += 1
                    return [actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, step_keys, global_step, step], [actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, step_keys, global_step, step]
                #####jax.debug.print("next_obs END {}: {}", next_obs.shape, next_obs)
                #####jax.debug.print("action END {}: {}", action.shape, action)
                
                scan_out_single, scan_out = jax.lax.scan(
                    policy_step,
                    [actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, step_keys, global_step, step],
                    (),
                    n_steps
                )
                # Return masked sum of rewards accumulated by agent in episode
                actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, step_keys, global_step, step = scan_out_single
                
                return actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, global_step
            
            return jax.jit(rollout_gymnax_)
            
        #hyperparameters = {key: value for key, value in vars(args).items()}
    
        # Save hyperparameters to wandb
        #wandb.config.update(hyperparameters)
        avg_score_list = []
        iteration = 1
        last_eval = 0

        n_steps_old = 0

        avg_episodic_return_list = []
        total_time_cleaned = 0
        
        while global_step < args.total_steps:
            #for iteration in range(1, n_iterations + 1):            
            wandb_log = {}
            # ALGO Logic: Storage setup
            #increase_index = global_step // (args.total_steps//len(increase_factor_list))
            if args.dynamic_buffer or not args.static_batch:
                #increase_index = global_step // (args.total_steps//sum(increase_factor_list))
                #increase_factor = int(increase_factor_list_long[increase_index])
                increase_factor = int(2**(np.ceil((((global_step+1)*8)/(1+args.total_steps)))-1)) # int(increase_factor_list_long[increase_index])
                increase_factor_batch = int(2**(np.ceil((((global_step+1)*8)/(1+args.total_steps)))-1)) # int(increase_factor_list_long[increase_index])
                if args.dynamic_buffer:
                    n_steps = initial_steps * increase_factor 
                else:
                    n_steps = initial_steps
                if not args.static_batch:
                    accumulate_gradients_every = int(accumulate_gradients_every_initial * increase_factor_batch)
                else:
                    accumulate_gradients_every = int(accumulate_gradients_every_initial)
                batch_size = int(args.n_envs * n_steps)
                #n_iterations = args.total_steps // batch_size
                #eval_freq = max(args.eval_freq // batch_size, 1)
                current_eval = global_step // args.eval_freq
                if n_steps != n_steps_old:
                    rollout_gymnax = create_rollout_gymnax(n_steps)
                    #compute_gae  = create_compute_gae(n_steps)
                    #update_ppo = create_update_ppo(batch_size, minibatch_size, accumulate_gradients_every)
                    n_steps_old = n_steps
            else:
                if global_step == 0:
                    rollout_gymnax = create_rollout_gymnax(n_steps)
                    #compute_gae  = create_compute_gae(n_steps)
                    #update_ppo = create_update_ppo(batch_size, minibatch_size, accumulate_gradients_every)
                current_eval = global_step // args.eval_freq
            start_time_cleaned = time.time()
            storage = Storage(
                obs=jnp.zeros((n_steps, args.n_envs) + obs_shape),
                actions=jnp.zeros((n_steps, args.n_envs) + envs.action_space().shape, dtype=jnp.int32),
                logprobs=jnp.zeros((n_steps, args.n_envs)),
                dones=jnp.zeros((n_steps, args.n_envs)),
                values=jnp.zeros((n_steps, args.n_envs)),
                advantages=jnp.zeros((n_steps, args.n_envs)),
                returns=jnp.zeros((n_steps, args.n_envs)),
                rewards=jnp.zeros((n_steps, args.n_envs)),
            )      
            #actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step = rollout(
            #    actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step
            #)
            actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, global_step = rollout_gymnax(
                actor_state, critic_state, env_state, env_params, episode_stats, next_obs, next_done, storage, key, global_step
            )

                
            #jax.debug.print("next_obs END {}: {}", next_obs.shape, next_obs)
            storage = compute_gae(critic_state, next_obs, next_done, storage)
            
            actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
                actor_state,
                critic_state,
                storage,
                key,
                accumulate_gradients_every,
            )
            elapsed_time_cleaned = time.time() - start_time_cleaned
            total_time_cleaned += elapsed_time_cleaned
            
            avg_episodic_return = np.mean(np.array(episode_stats.returned_episode_returns))
            avg_episodic_return_list.append(avg_episodic_return)            
            #writer.add_scalar("charts/avg_train_episodic_return", avg_episodic_return, global_step)
            wandb_log['charts/avg_train_episodic_return'] = avg_episodic_return
            if iteration == 1 or current_eval > last_eval or global_step + batch_size >= args.total_steps:
                last_eval = current_eval
                render_now = True if args.render_each_eval else True if global_step + batch_size >= args.total_steps else False
                                             
                def fit_stateActionDT(actor_state, env_id, n_episodes, name_appendix, seed=1_000):
                    
                    action_obs_store = ObservationActionBuffer(
                        #obs=jnp.zeros((n_steps, args.n_envs) + envs.single_observation_space.shape),
                        obs=jnp.zeros((n_steps, n_episodes) + obs_shape),
                        #actions=jnp.zeros((n_steps, args.n_envs) + envs.single_action_space.shape,
                        actions=jnp.zeros((n_steps, n_episodes) + envs.action_space().shape,
                                          dtype=jnp.int32)
                    )
                
                    total_eval_steps = 0
                    for episode_index in range(n_episodes):
                        #temp_env = build_env(env_id, n_env=1)
                        env_gymnax, env_params_eval = gymnax.make(args.env_id)                  
                        temp_env = wrappers.GymnaxToGymWrapper(env_gymnax, env_params_eval, 0)                               
                        
                        done, trunc = False, False
                        obs, info = temp_env.reset(seed=seed + episode_index)#random.randint(0, 1000))
                        step_counter = 0
                        while not done and not trunc:
                            actor_params = actor_state.params
                
                            if args.action_type == 'discrete':
                                action_logits = actor.apply(actor_params, np.array([obs]), indices=actor_state.indices)
                                action = jnp.argmax(action_logits, axis=1)
                                action = jnp.squeeze(action, axis=0) #jnp.squeeze(action, axis=0) if action.shape[0] == 1 else action #action[0]
                            else:
                                result = actor.apply(actor_params, np.array([obs]), indices=actor_state.indices)
                                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))
                                action = action_distribution.mean()
                                action = jnp.squeeze(action, axis=0)
                
                            action_obs_store = action_obs_store.replace(
                                obs=storage.obs.at[total_eval_steps].set(obs),
                                actions=storage.actions.at[total_eval_steps].set(action)
                            )
                            
                            next_obs, rewards, done, trunc, info = temp_env.step(jax.device_get(action))
                
                            obs = next_obs
                            step_counter += 1
                            total_eval_steps += 1
                
                        temp_env.close()
                
                    # Initialize decision tree
                    if args.action_type == 'discrete':
                        decision_tree = DecisionTreeClassifier(max_depth=args.depth)
                    else:
                        if action_dim == 1:
                            decision_tree = DecisionTreeRegressor(max_depth=args.depth)
                        else:
                            decision_tree = [DecisionTreeRegressor(max_depth=args.depth) for _ in range(action_dim)]
                
                    # Train the decision tree
                    X = np.array(action_obs_store.obs).reshape(-1, temp_env.observation_space.shape[-1])
                           
                    if args.action_type == 'discrete' or action_dim == 1:
                        y = np.array(action_obs_store.actions).reshape(-1)
                        decision_tree.fit(X, y)
                    else:
                        for i in range(action_dim):
                            y = np.array(action_obs_store.actions).reshape(-1, action_dim)
                            decision_tree[i].fit(X, y[:,i])
                    
                    return decision_tree
                
                          
                def evaluate_agent(actor_state, env_id, n_episodes, name_appendix, seed=100, decision_tree=None):
                    video_folder = 'videos/wandb'
                    if not os.path.exists(video_folder):
                        os.makedirs(video_folder)
                    #temp_env = Monitor(temp_env, video_folder) #, force=True        
        
                    score = []
                    score_interpretable = []
                    node_count = 0
                    for episode_index in range(n_episodes):
                        if args.actor == "stateActionDT":
                        
                            #temp_env = build_env(env_id, n_env=1)
                            env_gymnax, env_params_eval = gymnax.make(args.env_id)                  
                            temp_env = wrappers.GymnaxToGymWrapper(env_gymnax, env_params_eval, 0)                                
                            video_path = os.path.join(video_folder, run_name + "-" + "-" + env_id  + str(episode_index) + ".mp4")
                            image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)#  + str(episode_index))
                            
                            done, trunc = False, False
                            obs, info = temp_env.reset(seed=seed + episode_index)#random.randint(0, 1000))
                            running_reward = 0
                            frames = []
                            dones = False
                            step_counter = 0
                            while not done and not trunc:
                                if args.render_env and render_now:                      
                                    if False:#frame is not None:
                                        frame = temp_env.render()
                                        frame = frame[0]
    
                                        # Draw the figure on the canvas
                                        frame.canvas.draw()
                                        
                                        # Get the RGBA buffer from the figure
                                        w, h = frame.canvas.get_width_height()
                                        buf = np.frombuffer(frame.canvas.tostring_argb(), dtype=np.uint8)
                                        buf.shape = (h, w, 4)
                                        
                                        # Convert from ARGB to RGBA
                                        buf = np.roll(buf, 3, axis=2)
                                        
                                        # Remove the alpha channel
                                        frame = buf[:, :, :3]                                    
                                        image = Image.fromarray(frame)
                                        draw = ImageDraw.Draw(image)
                                        text_step = f'Step: {step_counter}'
                                        font_size = frame.shape[0] // 20
                                        draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                        text_reward = f'Reward: {running_reward}'
                                        draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                        
                                        frames.append(np.array(image))
                                    
                                actor_params = actor_state.params
                                
                                flat_obs = obs.reshape(1, -1)
                                if args.action_type == 'discrete':
                                    action = decision_tree.predict(flat_obs)[0]
                                    #print('decision_tree.predict(flat_obs)', decision_tree.predict(flat_obs))
                                    #print('action', action)
                                else:
                                    if action_dim == 1:
                                        action = decision_tree.predict(flat_obs)
                                    else:
                                        action_list = []
                                        for i in range(action_dim):
                                            action_by_tree = decision_tree[i].predict(flat_obs)[0]
                                            action_list.append(action_by_tree)
                                        action = np.array(action_list)
                                        
                                next_obs, rewards, done, trunc, info = temp_env.step(action)
    
                                running_reward += rewards
                                #if "final_info" in info:
                                #    episode_reward = info["final_info"][0]["episode"]["r"]
                                #    score.append(episode_reward)
                                obs = next_obs
                                step_counter += 1
                                
                            score_interpretable.append(running_reward)
                            if (args.render_env and render_now):              
                                if False:#frame is not None:
                                    frame = temp_env.render()
                                    frame = frame[0]
                                    # Draw the figure on the canvas
                                    frame.canvas.draw()
                                    
                                    # Get the RGBA buffer from the figure
                                    w, h = frame.canvas.get_width_height()
                                    buf = np.frombuffer(frame.canvas.tostring_argb(), dtype=np.uint8)
                                    buf.shape = (h, w, 4)
                                    
                                    # Convert from ARGB to RGBA
                                    buf = np.roll(buf, 3, axis=2)
                                    
                                    # Remove the alpha channel
                                    frame = buf[:, :, :3]    
                                    
                                    image = Image.fromarray(frame)
                                    draw = ImageDraw.Draw(image)
                                    text_step = f'Step: {step_counter}'
                                    font_size = frame.shape[0] // 20
                                    draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                    text_reward = f'Reward: {running_reward}'
                                    draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                    
                                    frames.append(np.array(image))
            
                                    numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2)) 
                                    fps = 5 if 'MiniGrid' in env_id else 25
                                    if args.track:
                                        wandb.log({"gameplay_" + name_appendix + '_trial' + str(episode_index): wandb.Video(numpy_clip, fps=fps, format="mp4")}, commit=False)
                                        #wandb_log["gameplay_" + name_appendix + '_trial' + str(episode_index)] = wandb.Video(numpy_clip, fps=fps, format="mp4")
                                if episode_index==0:
                                    if args.action_type == 'discrete' or action_dim == 1:
                                        # Plot the decision tree
                                        plt.figure(figsize=(20, 10))
                                        plot_tree(decision_tree, filled=True)
                                        plt.title("Decision Tree")
                                
                                        # Save the plot to a file
                                        video_folder = 'videos/wandb'
                                        image_path = os.path.join(video_folder, run_name + "-" + "-" + args.env_id)
                                        plot_filename = image_path + "state_action_DT.png"
                                        plt.savefig(plot_filename)
                                        plt.close()
                                
                                        # Log the image to wandb
                                        if args.track:
                                            wandb.log({"state_action_DT": wandb.Image(plot_filename)})
                                            node_count += decision_tree.tree_.node_count
                                    else:
                                        node_count = 0
                                        for i in range(action_dim):                    
                                            # Plot the decision tree
                                            plt.figure(figsize=(20, 10))
                                            plot_tree(decision_tree[i], filled=True)
                                            plt.title("Decision Tree " + str(i))
                                    
                                            # Save the plot to a file
                                            video_folder = 'videos/wandb'
                                            image_path = os.path.join(video_folder, run_name + "-" + "-" + args.env_id)
                                            plot_filename = image_path + "state_action_DT_reg" + str(i) + ".png"
                                            plt.savefig(plot_filename)
                                            plt.close()
                                            node_count += decision_tree[i].tree_.node_count
                                            if args.track:
                                                # Log the image to wandb
                                                wandb.log({"state_action_DT_" + str(i): wandb.Image(plot_filename)})

                            temp_env.close()
                    
                        elif args.actor == "d-sdt":
                            #temp_env = build_env(env_id, n_env=1)
                            env_gymnax, env_params_eval = gymnax.make(args.env_id)                  
                            temp_env = wrappers.GymnaxToGymWrapper(env_gymnax, env_params_eval, 0)       
                       
                            video_path = os.path.join(video_folder, run_name + "-" + "-" + env_id  + str(episode_index) + ".mp4")
                            image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)#  + str(episode_index))
                            
                            done, trunc = False, False
                            obs, info = temp_env.reset(seed=seed + episode_index)#random.randint(0, 1000))
                            running_reward = 0
                            frames = []
                            dones = False
                            step_counter = 0
                            while not done and not trunc:
                                if args.render_env and render_now:                                
                                    if False:#frame is not None:
                                        frame = temp_env.render()
                                        frame = frame[0]
    
                                        # Draw the figure on the canvas
                                        frame.canvas.draw()
                                        
                                        # Get the RGBA buffer from the figure
                                        w, h = frame.canvas.get_width_height()
                                        buf = np.frombuffer(frame.canvas.tostring_argb(), dtype=np.uint8)
                                        buf.shape = (h, w, 4)
                                        
                                        # Convert from ARGB to RGBA
                                        buf = np.roll(buf, 3, axis=2)
                                        
                                        # Remove the alpha channel
                                        frame = buf[:, :, :3]                                    
                                        image = Image.fromarray(frame)
                                        draw = ImageDraw.Draw(image)
                                        text_step = f'Step: {step_counter}'
                                        font_size = frame.shape[0] // 20
                                        draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                        text_reward = f'Reward: {running_reward}'
                                        draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                        
                                        frames.append(np.array(image))
                                    
                                actor_params = actor_state.params
                                #####jax.debug.print("np.array([obs]): {}", np.array([obs]).shape)
                                actor_params_discrete = convert_to_discrete_tree(actor_params, args.action_type, temperature=args.temperature)
                                if args.action_type == 'discrete':
                                    action_logits = actor.apply(actor_params_discrete, np.array([obs]), max_path=True, indices=actor_state.indices)
                                    action = jnp.argmax(action_logits, axis=1)
                                    action = jnp.squeeze(action, axis=0) #jnp.squeeze(action, axis=0) if action.shape[0] == 1 else action #action[0]
                                else:
                                    result = actor.apply(actor_params_discrete, np.array([obs]), max_path=True, indices=actor_state.indices)
                                    action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))
                                    action = action_distribution.mean()
                                    action = jnp.squeeze(action, axis=0)                                    
    
                                    action = jax.device_get(action)
    
                                next_obs, rewards, done, trunc, info = temp_env.step(action)
    
                                running_reward += rewards
                                #if "final_info" in info:
                                #    episode_reward = info["final_info"][0]["episode"]["r"]
                                #    score.append(episode_reward)
                                obs = next_obs
                                step_counter += 1
                                
                            score_interpretable.append(running_reward)
                            if (args.render_env and render_now):        
                                if False:#frame is not None:
                                    frame = temp_env.render()
                                    frame = frame[0]
                                    # Draw the figure on the canvas
                                    frame.canvas.draw()
                                    
                                    # Get the RGBA buffer from the figure
                                    w, h = frame.canvas.get_width_height()
                                    buf = np.frombuffer(frame.canvas.tostring_argb(), dtype=np.uint8)
                                    buf.shape = (h, w, 4)
                                    
                                    # Convert from ARGB to RGBA
                                    buf = np.roll(buf, 3, axis=2)
                                    
                                    # Remove the alpha channel
                                    frame = buf[:, :, :3]    
                                    
                                    image = Image.fromarray(frame)
                                    draw = ImageDraw.Draw(image)
                                    text_step = f'Step: {step_counter}'
                                    font_size = frame.shape[0] // 20
                                    draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                    text_reward = f'Reward: {running_reward}'
                                    draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                    
                                    frames.append(np.array(image))
            
                                    numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2)) 
                                    fps = 5 if 'MiniGrid' in env_id else 25
                                    if args.track:
                                        wandb.log({"gameplay_" + name_appendix + '_trial' + str(episode_index): wandb.Video(numpy_clip, fps=fps, format="mp4")}, commit=False)                          
                                if episode_index==0:
                                    split_values = actor_params_discrete['params']['SDT_0']['inner_nodes']['layers_0']['kernel'].T * jnp.expand_dims(actor_params_discrete['params']['SDT_0']['inner_nodes']['layers_0']['bias'],1)
                                    split_indices = actor_params_discrete['params']['SDT_0']['inner_nodes']['layers_0']['kernel'].T
                                    leaf_values = actor_params_discrete['params']['SDT_0']['leaf_nodes']['kernel']
           
                                    image_path, node_count = plot_decision_tree(
                                                                    split_values=split_values, 
                                                                    split_indices=split_indices, 
                                                                    leaf_values=leaf_values,
                                                                    features_by_estimator=[i for i in range(obs_dim)],
                                                                    image_path=image_path,
                                                                    observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                                    filename_appendix = 'D-SDT',
                                                                    env=env_gymnax, 
                                                                    env_params=env_params_eval,
                                                                    prune=True,
                                                                    continuous = args.action_type != 'discrete'
                                                                   )
                                    image_path_plot = image_path + '.png'     
                                    if args.track:
                                        wandb.log({"D-SDT_"+ name_appendix + '_trial' + str(episode_index): wandb.Image(image_path_plot)}, commit=False)
                                    
                                    image_path_complete = image_path + '_COMPLETE'
                                    image_path_complete, _ = plot_decision_tree(
                                                                    split_values=split_values, 
                                                                    split_indices=split_indices, 
                                                                    leaf_values=leaf_values,
                                                                    features_by_estimator=[i for i in range(obs_dim)],
                                                                    image_path=image_path_complete,
                                                                    observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                                    filename_appendix = 'D-SDT',
                                                                    env=env_gymnax, 
                                                                    env_params=env_params_eval,
                                                                    prune=False,
                                                                    continuous = args.action_type != 'discrete'
                                                                   )
                                    
                                    image_path_plot = image_path_complete + '.png'     
                                    if args.track:
                                        wandb.log({"D-SDT_COMPLETE"+ name_appendix + '_trial' + str(episode_index): wandb.Image(image_path_plot)}, commit=False)
                                
                            
                            temp_env.close()                    
                        
                        #temp_env = build_env(env_id, n_env=1)
                        env_gymnax, env_params_eval = gymnax.make(args.env_id)                  
                        temp_env = wrappers.GymnaxToGymWrapper(env_gymnax, env_params_eval, 0)       
                          
                        video_path = os.path.join(video_folder, run_name + "-" + "-" + env_id  + str(episode_index) + ".mp4")
                        image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)#  + str(episode_index))
                        
                        done, trunc = False, False
                        obs, info = temp_env.reset(seed=seed + episode_index)#random.randint(0, 1000))
                        running_reward = 0
                        frames = []
                        dones = False
                        step_counter = 0
                        while not done and not trunc:
                            if args.render_env and render_now:
                                if False:#frame is not None:
                                    frame = temp_env.render()
                                    frame = frame[0]

                                    # Draw the figure on the canvas
                                    frame.canvas.draw()
                                    
                                    # Get the RGBA buffer from the figure
                                    w, h = frame.canvas.get_width_height()
                                    buf = np.frombuffer(frame.canvas.tostring_argb(), dtype=np.uint8)
                                    buf.shape = (h, w, 4)
                                    
                                    # Convert from ARGB to RGBA
                                    buf = np.roll(buf, 3, axis=2)
                                    
                                    # Remove the alpha channel
                                    frame = buf[:, :, :3]                                    
                                    image = Image.fromarray(frame)
                                    draw = ImageDraw.Draw(image)
                                    text_step = f'Step: {step_counter}'
                                    font_size = frame.shape[0] // 20
                                    draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                    text_reward = f'Reward: {running_reward}'
                                    draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                    
                                    frames.append(np.array(image))
                                
                            actor_params = actor_state.params
                            #####jax.debug.print("np.array([obs]): {}", np.array([obs]).shape)

                            if args.action_type == 'discrete':
                                action_logits = actor.apply(actor_params, np.array([obs]), indices=actor_state.indices)
                                action = jnp.argmax(action_logits, axis=1)
                                action = jnp.squeeze(action, axis=0) #jnp.squeeze(action, axis=0) if action.shape[0] == 1 else action #action[0]
                            else:
                                result = actor.apply(actor_params, np.array([obs]), indices=actor_state.indices)
                                
                                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))
                                action = action_distribution.mean()
                                action = jnp.squeeze(action, axis=0)                                    

                            action = jax.device_get(action)

                            next_obs, rewards, done, trunc, info = temp_env.step(action)

                            running_reward += rewards
                            #if "final_info" in info:
                            #    episode_reward = info["final_info"][0]["episode"]["r"]
                            #    score.append(episode_reward)
                            obs = next_obs
                            step_counter += 1
                            
                        score.append(running_reward)
                        if (args.render_env and render_now):    
                            if False:#frame is not None:
                                frame = temp_env.render()
                                frame = frame[0]
                                # Draw the figure on the canvas
                                frame.canvas.draw()
                                
                                # Get the RGBA buffer from the figure
                                w, h = frame.canvas.get_width_height()
                                buf = np.frombuffer(frame.canvas.tostring_argb(), dtype=np.uint8)
                                buf.shape = (h, w, 4)
                                
                                # Convert from ARGB to RGBA
                                buf = np.roll(buf, 3, axis=2)
                                
                                # Remove the alpha channel
                                frame = buf[:, :, :3]    
                                
                                image = Image.fromarray(frame)
                                draw = ImageDraw.Draw(image)
                                text_step = f'Step: {step_counter}'
                                font_size = frame.shape[0] // 20
                                draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                text_reward = f'Reward: {running_reward}'
                                draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size))
                                
                                frames.append(np.array(image))
        
                                numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2)) 
                                fps = 5 if 'MiniGrid' in env_id else 25
                                if args.track:
                                    wandb.log({"gameplay_" + name_appendix + '_trial' + str(episode_index): wandb.Video(numpy_clip, fps=fps, format="mp4")}, commit=False)
                                    #wandb_log["gameplay_" + name_appendix + '_trial' + str(episode_index)] = wandb.Video(numpy_clip, fps=fps, format="mp4")
                        
                            if args.n_estimators <= 5 and args.actor == "sympol" and episode_index==0:
                                for estimator_number in range(args.n_estimators):
                                    filename_appendix = '_' + str(estimator_number)
                                    env_gymnax_plot = env_gymnax
                                    if args.normEnv:
                                        env_gymnax_plot = NormalizeObservationWrapper(env_gymnax_plot, env_params_eval)    
                                        
                                    image_path, node_count = plot_decision_tree(
                                                                    split_values=actor_params['split_values'][estimator_number], 
                                                                    split_indices=actor_params['split_idx_array'][estimator_number], 
                                                                    leaf_values=actor_params['leaf_array'][estimator_number],
                                                                    features_by_estimator=actor_state.indices['features_by_estimator'][estimator_number],
                                                                    image_path=image_path,
                                                                    observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                                    filename_appendix = filename_appendix,
                                                                    env=env_gymnax_plot, 
                                                                    env_params=env_params_eval, 
                                                                    prune=True,
                                                                    continuous = args.action_type != 'discrete'
                                                                   )
                                    #print(f"global_step={global_step}, actor_params={actor_params}")
                                    #print(f"global_step={global_step}, actor_state.indices={actor_state.indices}")

                                    
                                image_path_plot = image_path + '.png'  
                                if args.track:
                                    wandb.log({"DT_"+ name_appendix + '_trial' + str(episode_index) + '_estNumber' + str(estimator_number): wandb.Image(image_path_plot)}, commit=False)
                                    #wandb_log["DT_"+ name_appendix + '_trial' + str(episode_index) + '_estNumber' + str(estimator_number)] = wandb.Image(image_path)
                                for estimator_number in range(args.n_estimators):
                                    filename_appendix = '_' + str(estimator_number)
                                    env_gymnax_plot = env_gymnax   
                                    if args.normEnv:
                                        env_gymnax_plot = NormalizeObservationWrapper(env_gymnax_plot, env_params_eval)                                            
                                    image_path_complete = image_path + '_COMPLETE'
                                    image_path_complete, _ = plot_decision_tree(
                                                                    split_values=actor_params['split_values'][estimator_number], 
                                                                    split_indices=actor_params['split_idx_array'][estimator_number], 
                                                                    leaf_values=actor_params['leaf_array'][estimator_number],
                                                                    features_by_estimator=actor_state.indices['features_by_estimator'][estimator_number],
                                                                    image_path=image_path_complete,
                                                                    observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                                    filename_appendix = filename_appendix,
                                                                    env=env_gymnax_plot, 
                                                                    env_params=env_params_eval,
                                                                    prune=False,
                                                                    continuous = args.action_type != 'discrete'
                                                                   )
                                    
                                image_path_plot = image_path_complete + '.png'     
                                if args.track:
                                    wandb.log({"DT_COMPLETE"+ name_appendix + '_trial' + str(episode_index) + '_estNumber' + str(estimator_number): wandb.Image(image_path_plot)}, commit=False)
                                    #wandb_log["DT_"+ name_appendix + '_trial' + str(episode_index) + '_estNumber' + str(estimator_number)] = wandb.Image(image_path)
                            elif (args.actor == 'sdt' or args.actor == 'd-sdt') and episode_index==0:
    
                                split_values = actor_params['params']['SDT_0']['inner_nodes']['layers_0']['bias']
                                split_indices = actor_params['params']['SDT_0']['inner_nodes']['layers_0']['kernel'].T
                                leaf_values = actor_params['params']['SDT_0']['leaf_nodes']['kernel']
    
                                
                                image_path, node_count_sdt = plot_decision_tree_soft(
                                                                split_values=split_values, 
                                                                split_indices=split_indices, 
                                                                leaf_values=leaf_values,
                                                                image_path=image_path,
                                                                observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                                filename_appendix =  'SDT', 
                                                                temperature=args.temperature
                                                               )
                                if args.actor == 'sdt':
                                    node_count = node_count_sdt
                                image_path_plot = image_path + '.png' 
                                if args.track:
                                    wandb.log({"SDT_"+ name_appendix + '_trial' + str(episode_index): wandb.Image(image_path_plot)}, commit=False)
                            
                        temp_env.close()
                        
                    #avg_score = np.mean(score)
                    #avg_score_interpretable = np.mean(score_interpretable)
                    #return avg_score.item(), avg_score_interpretable.item(), node_count
                    return score, score_interpretable, node_count
                
                if args.actor == "stateActionDT":
                    decision_tree = fit_stateActionDT(actor_state, args.env_id,
                                               n_episodes=25,
                                               name_appendix='',
                                            seed=args.seed)          
                else:
                    decision_tree = None
                
                score, score_interpretable, node_count = evaluate_agent(actor_state, args.env_id,
                                           n_episodes=args.n_eval_episodes,
                                           name_appendix='',
                                           decision_tree=decision_tree,
                                           seed=env_seed)

                avg_score = np.mean(score).item()
                avg_score_interpretable = np.mean(score_interpretable).item()
                std_score = np.std(score).item()
                std_score_interpretable = np.std(score_interpretable).item()      
                
                # use the negative avg score, since reduce on plataeu normally considers non-decreasing losses as a plataeu,
                # but we have a plataeu when the score is not increasing anymore
                if args.reduce_lr:
                    _, lr_scheduler_state = lr_scheduler.update(
                        updates=actor_state.params, state=lr_scheduler_state, value=avg_score
                    )
                    # [-1] is the adamw optimizer, while [0] would be the gradient clipping of the tx.chain
                    if args.actor != "sympol":
                        actor_state.opt_state[1].hyperparams["learning_rate"] = learning_rate_actor * lr_scheduler_state.scale
                    else:
                        actor_state.opt_state[1][0]['estimator_weights'][0].hyperparams["learning_rate"] = learning_rate_actor_weights * lr_scheduler_state.scale
                        actor_state.opt_state[1][0]['split_values'][0].hyperparams["learning_rate"] = learning_rate_actor_split_values * lr_scheduler_state.scale
                        actor_state.opt_state[1][0]['split_idx_array'][0].hyperparams["learning_rate"] = learning_rate_actor_split_idx_array * lr_scheduler_state.scale
                        actor_state.opt_state[1][0]['leaf_array'][0].hyperparams["learning_rate"] = learning_rate_actor_leaf_array * lr_scheduler_state.scale
                        actor_state.opt_state[1][0]['log_std'][0].hyperparams["learning_rate"] = learning_rate_actor_log_std * lr_scheduler_state.scale


                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = end_time
                if args.actor == "stateActionDT" or args.actor == "d-sdt":
                    print(f"global_step={global_step}, avg_eval_episodic_return={avg_score}, avg_eval_episodic_return_discrete={avg_score_interpretable} (Elapsed time: {elapsed_time} seconds)")
                    wandb_log['charts/avg_score'] = avg_score_interpretable
                    wandb_log['charts/avg_score_fully_complexity'] = avg_score
                    wandb_log['charts/std_score'] = std_score_interpretable
                    wandb_log['charts/std_score_fully_complexity'] = std_score
                    wandb_log['charts/score_list'] = score_interpretable
                    wandb_log['charts/score_list'] = score
                    avg_score_list.append(avg_score_interpretable)                    
                else:
                    print(f"global_step={global_step}, avg_eval_episodic_return={avg_score} (Elapsed time: {elapsed_time} seconds)")
                    wandb_log['charts/avg_score'] = avg_score
                    wandb_log['charts/std_score'] = std_score
                    wandb_log['charts/score_list'] = score
                    
                    avg_score_list.append(avg_score)

                wandb_log['charts/node_count'] = node_count
                wandb_log['charts/total_time_cleaned'] = total_time_cleaned
                
                if global_step + batch_size >= args.total_steps: #TEST EVAL
                    test_seed = 123456
                    if args.actor == "stateActionDT":
                        decision_tree = fit_stateActionDT(actor_state, 
                                                          args.env_id,
                                                          n_episodes=25,
                                                          name_appendix='',
                                                          seed=args.seed)          
                    else:
                        decision_tree = None
                    
                    
                    score_test, score_interpretable_test, node_count_test = evaluate_agent(actor_state, args.env_id,
                                               n_episodes=args.n_eval_episodes,
                                               name_appendix='',
                                               decision_tree=decision_tree,
                                               seed=test_seed)
    
                    avg_score_test = np.mean(score_test).item()
                    avg_score_interpretable_test = np.mean(score_interpretable_test).item()
                    std_score_test = np.std(score_test).item()
                    std_score_interpretable_test = np.std(score_interpretable_test).item()                
                    # use the negative avg score, since reduce on plataeu normally considers non-decreasing losses as a plataeu,
                    # but we have a plataeu when the score is not increasing anymore
                    if args.actor == "stateActionDT" or args.actor == "d-sdt":
                        print(f"global_step={global_step}, avg_eval_episodic_return={avg_score_test}, avg_eval_episodic_return_discrete={avg_score_interpretable_test} (Elapsed time: {elapsed_time} seconds)")
                        wandb_log['charts/avg_score_test'] = avg_score_interpretable_test
                        wandb_log['charts/avg_score_fully_complexity_test'] = avg_score_test
                        wandb_log['charts/std_score_test'] = std_score_interpretable_test
                        wandb_log['charts/std_score_fully_complexity_test'] = std_score_test
                        wandb_log['charts/score_list_test'] = score_interpretable_test
                        wandb_log['charts/score_fully_complexity_list_test'] =  score_test 

                    else:
                        print(f"global_step={global_step}, avg_eval_episodic_return={avg_score_test} (Elapsed time: {elapsed_time} seconds)")
                        wandb_log['charts/avg_score_test'] = avg_score_test
                        wandb_log['charts/std_score_test'] = std_score
                        wandb_log['charts/score_list_test'] = score_test
                            
                    wandb_log['charts/node_count_test'] = node_count_test


                
                try:
                    complexity_add = 1
                    while False:
                        #Evaluate next complexity level
                        string_list = args.env_id.split("-")
                        complexity_level_new = str(int(string_list[-2][-1])+complexity_add)
                        string_list[-2] = string_list[-2][:-1] + complexity_level_new
                        env_id_new = "-".join(string_list)
                        avg_score = evaluate_agent(actor_state, env_id_new, n_episodes=args.n_eval_episodes, name_appendix='complexity+' + str(complexity_add))
                        # [-1] is the adamw optimizer, while [0] would be the gradient clipping of the tx.chain

                        print(f"global_step={global_step}, complexity={complexity_level_new} avg_eval_episodic_return={avg_score}")
                        #writer.add_scalar("charts/avg_score_complexity" + complexity_level_new, avg_score, global_step)
                        wandb_log["charts/avg_score_complexity" + complexity_level_new] = avg_score
                        complexity_add += 1
                except:
                    pass
    
            if args.checkpoint:
                checkpoint_path = os.path.join(args.path, args.run_name)
                os.makedirs(checkpoint_path, exist_ok=True)
                ckpt = {'sympol': actor_state}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args)        
                
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            #writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            #writer.add_scalar(
            #    "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
            #)
            #writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            #writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            #writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            #writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            #writer.add_scalar("losses/loss", loss.item(), global_step)
            wandb_log['charts/global_step'] = global_step
            wandb_log['charts/avg_episodic_return'] = avg_episodic_return
            wandb_log['charts/avg_episodic_return_100'] = np.mean(avg_episodic_return_list[-100:])
            wandb_log['charts/avg_episodic_return_10'] = np.mean(avg_episodic_return_list[-10:])
            wandb_log['charts/avg_episodic_length'] = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            try:
                wandb_log['losses/value_loss'] = np.mean(v_loss[-1])#.item()
                wandb_log['losses/policy_loss'] = np.mean(pg_loss[-1])#.item()
                wandb_log['losses/entropy'] = np.mean(entropy_loss[-1])#.item()
                wandb_log['losses/approx_kl'] = np.mean(approx_kl[-1])#.item()
                wandb_log['losses/loss'] = np.mean(loss[-1])#.item()
            except:
                wandb_log['losses/value_loss'] = v_loss#.item()
                wandb_log['losses/policy_loss'] = pg_loss#.item()
                wandb_log['losses/entropy'] = entropy_loss#.item()
                wandb_log['losses/approx_kl'] = approx_kl#.item()
                wandb_log['losses/loss'] = loss#.item()                
            if args.track:
                wandb.log(wandb_log)   
        
            iteration = iteration + 1
        if args.track:
            wandb_run.finish()

        #trial_scores.append(np.mean(avg_score_list[-5:]))
        trial_scores.append(avg_score_list[-1])

    if queue is None:
        return np.mean(trial_scores)
    else:
        queue.put(np.mean(trial_scores))  # Put the result in the queue


import multiprocessing

def multiprocessing_objective_fn(args, trial):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=train_agent, args=(args, trial, queue))
    p.start()
    p.join()
    result = queue.get()
    return result
    

if __name__ == "__main__":
    from optuna.storages import RDBStorage
    from sqlalchemy import create_engine
    import socket


    args = get_args()
    print(args)
    if args.optimize_config:
        # used to save information about trials, delete that if you want to start new trials, e.g. after changing the range
        # of hyperparamters or adding/ removing some hyperparameters
        storage = "sqlite:///hpopt_database_" + socket.gethostname() + ".db"

        # Step 2: Create the engine with the specified timeout
        #engine = create_engine("sqlite:///optuna_database.db", connect_args={'timeout': 30})
        
        # Step 3: Use this engine to create the Optuna storage
        #storage = RDBStorage("sqlite:///optuna_database_rdb.db")
        
        study = optuna.create_study(
            direction="maximize", storage=storage, load_if_exists=True, study_name=args.exp_name + '__' + args.env_id + '__' + args.run_name + '__' + args.actor
        )
        objective_fn = functools.partial(multiprocessing_objective_fn, args)
        #objective_fn = functools.partial(train_agent, args)
        # wandb only works for n_jobs = 1 ! See README.md for more infos about that
        study.optimize(objective_fn, n_trials=args.n_trials, n_jobs=1)
        #study.optimize(objective_fn, n_trials=args.n_trials, n_jobs=1)
    else:
        train_agent(args, trial=None, queue=None)
