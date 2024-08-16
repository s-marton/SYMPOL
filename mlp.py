import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from flax.linen.initializers import constant, orthogonal


class Critic_MLP(nn.Module):
    num_layers: int = 2
    neurons_per_layer: int = 256
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs):
        for layer in range(self.num_layers):
            x = nn.Dense(self.neurons_per_layer)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor_MLP(nn.Module):
    action_dim: int
    num_layers: int = 2
    neurons_per_layer: int = 256
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, **kwargs):
        for layer in range(self.num_layers):
            obs = nn.Dense(self.neurons_per_layer)(obs)
            obs = nn.relu(obs)
            obs = nn.Dense(self.action_dim)(obs)
        return obs

class Actor_MLP_Continuous(nn.Module):
    action_dim: int
    num_layers: int = 2
    neurons_per_layer: int = 256
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, **kwargs):
        x = obs
        for layer in range(self.num_layers):
            x = nn.Dense(self.neurons_per_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        #mean = nn.Dense(self.action_dim)(x)
        #log_std = self.param("log_std", nn.initializers.zeros, (1, self.action_dim))
        #return jnp.stack([mean, jnp.tile(jnp.ravel(log_std), (mean.shape[0],1))], axis=1)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        return x, actor_logtstd#distrax.MultivariateNormalDiag(x, jnp.exp(actor_logtstd))
