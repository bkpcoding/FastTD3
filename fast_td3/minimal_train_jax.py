import os
import sys
import random

os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax
from tensordict import TensorDict
import torch

from fast_td3.fast_td3_utils import (
    SimpleReplayBuffer,
    mark_step,
)
from fast_td3.hyperparams import get_args

jax.config.update("jax_enable_x64", False)


def _jax_to_torch(tensor):
    import torch.utils.dlpack as tpack
    tensor = tpack.from_dlpack(tensor)
    return tensor


def _torch_to_jax(tensor):
    from jax.dlpack import from_dlpack
    tensor = from_dlpack(tensor)
    return tensor


class DistributionalQNetwork(nn.Module):
    n_obs: int
    n_act: int
    num_atoms: int
    v_min: float
    v_max: float
    hidden_dim: int

    @nn.compact
    def __call__(self, obs, actions):
        x = jnp.concatenate([obs, actions], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 4)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_atoms)(x)
        return x


class Critic(nn.Module):
    n_obs: int
    n_act: int
    num_atoms: int
    v_min: float
    v_max: float
    hidden_dim: int

    def setup(self):
        self.qnet1 = DistributionalQNetwork(
            n_obs=self.n_obs,
            n_act=self.n_act,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            hidden_dim=self.hidden_dim,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=self.n_obs,
            n_act=self.n_act,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            hidden_dim=self.hidden_dim,
        )
        self.q_support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    def __call__(self, obs, actions):
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def get_value(self, probs):
        q_support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        return jnp.sum(probs * q_support, axis=-1)


def distributional_projection(q1_logits, q2_logits, rewards, bootstrap, discount, v_min, v_max, num_atoms):
    """Standalone distributional projection function"""
    q_support = jnp.linspace(v_min, v_max, num_atoms)
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = rewards.shape[0]

    target_z = (
        rewards[:, None]
        + bootstrap[:, None] * discount[:, None] * q_support[None, :]
    )
    target_z = jnp.clip(target_z, v_min, v_max)
    b = (target_z - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)

    l_mask = jnp.logical_and((u > 0), (l == u))
    u_mask = jnp.logical_and((l < (num_atoms - 1)), (l == u))

    l = jnp.where(l_mask, l - 1, l)
    u = jnp.where(u_mask, u + 1, u)

    next_dist1 = nn.softmax(q1_logits, axis=-1)
    next_dist2 = nn.softmax(q2_logits, axis=-1)

    def project_dist(next_dist):
        proj_dist = jnp.zeros_like(next_dist)
        offset = (
            jnp.linspace(
                0, (batch_size - 1) * num_atoms, batch_size
            )[:, None]
            .repeat(num_atoms, axis=1)
            .astype(jnp.int32)
        )
        l_indices = (l + offset).reshape(-1)
        u_indices = (u + offset).reshape(-1)
        l_weights = (next_dist * (u.astype(jnp.float32) - b)).reshape(-1)
        u_weights = (next_dist * (b - l.astype(jnp.float32))).reshape(-1)
        
        proj_dist = proj_dist.at[l_indices // num_atoms, l_indices % num_atoms].add(l_weights)
        proj_dist = proj_dist.at[u_indices // num_atoms, u_indices % num_atoms].add(u_weights)
        return proj_dist

    q1_proj = project_dist(next_dist1)
    q2_proj = project_dist(next_dist2)
    return q1_proj, q2_proj



class Actor(nn.Module):
    n_obs: int
    n_act: int
    hidden_dim: int
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 4)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_act, 
                    kernel_init=nn.initializers.normal(self.init_scale))(x)
        return nn.tanh(x)


class EmpiricalNormalization:
    def __init__(self, shape, eps=1e-8):
        self.eps = eps
        self.shape = shape
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = 0

    def update(self, x):
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / jnp.sqrt(self.var + self.eps)


def create_train_state(rng, actor_net, critic_net, args):
    dummy_obs = jnp.ones((1, args.n_obs))
    dummy_actions = jnp.ones((1, args.n_act))
    
    actor_key, critic_key = jrandom.split(rng)
    actor_params = actor_net.init(actor_key, dummy_obs)
    critic_params = critic_net.init(critic_key, dummy_obs, dummy_actions)
    
    actor_tx = optax.adamw(args.actor_learning_rate, weight_decay=args.weight_decay)
    critic_tx = optax.adamw(args.critic_learning_rate, weight_decay=args.weight_decay)
    
    return {
        'actor_params': actor_params,
        'critic_params': critic_params,
        'critic_target_params': critic_params,
        'actor_opt_state': actor_tx.init(actor_params),
        'critic_opt_state': critic_tx.init(critic_params),
    }, actor_tx, critic_tx


def make_actor_step_fn(actor_net, critic_net, actor_tx):
    @jax.jit
    def actor_step(state, batch, use_cdq):
        def actor_loss_fn(actor_params):
            actions = actor_net.apply(actor_params, batch['observations'])
            q1, q2 = critic_net.apply(state['critic_params'], batch['observations'], actions)
            q1_value = critic_net.get_value(nn.softmax(q1, axis=-1))
            q2_value = critic_net.get_value(nn.softmax(q2, axis=-1))
            
            q_value = jax.lax.cond(
                use_cdq,
                lambda _: jnp.minimum(q1_value, q2_value),
                lambda _: (q1_value + q2_value) / 2.0,
                None
            )
            
            return -jnp.mean(q_value)
        
        loss, grads = jax.value_and_grad(actor_loss_fn)(state['actor_params'])
        
        updates, new_opt_state = actor_tx.update(grads, state['actor_opt_state'], state['actor_params'])
        new_params = optax.apply_updates(state['actor_params'], updates)
        
        new_state = state.copy()
        new_state['actor_params'] = new_params
        new_state['actor_opt_state'] = new_opt_state
        
        return new_state, loss
    
    return actor_step


def make_critic_step_fn(actor_net, critic_net, critic_tx):
    @jax.jit
    def critic_step(state, batch, policy_noise, noise_clip, gamma, use_cdq, rng):
        def critic_loss_fn(critic_params):
            noise = jrandom.normal(rng, batch['actions'].shape) * policy_noise
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            
            next_actions = actor_net.apply(state['actor_params'], batch['next_observations'])
            next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
            
            q1_next_target, q2_next_target = critic_net.apply(
                state['critic_target_params'], batch['next_observations'], next_actions
            )
            
            discount = gamma ** batch['effective_n_steps']
            q1_proj, q2_proj = distributional_projection(
                q1_next_target,
                q2_next_target,
                batch['rewards'],
                batch['bootstrap'],
                discount,
                critic_net.v_min,
                critic_net.v_max,
                critic_net.num_atoms
            )
            
            q1_target_value = critic_net.get_value(nn.softmax(q1_next_target, axis=-1))
            q2_target_value = critic_net.get_value(nn.softmax(q2_next_target, axis=-1))
            
            def use_cdq_fn(_):
                q_target_dist = jnp.where(
                    q1_target_value[:, None] < q2_target_value[:, None],
                    q1_proj,
                    q2_proj
                )
                return q_target_dist, q_target_dist
            
            def no_cdq_fn(_):
                return q1_proj, q2_proj
            
            q1_target_dist, q2_target_dist = jax.lax.cond(
                use_cdq, use_cdq_fn, no_cdq_fn, None
            )
            
            q1, q2 = critic_net.apply(critic_params, batch['observations'], batch['actions'])
            
            q1_loss = -jnp.sum(q1_target_dist * nn.log_softmax(q1, axis=-1), axis=-1).mean()
            q2_loss = -jnp.sum(q2_target_dist * nn.log_softmax(q2, axis=-1), axis=-1).mean()
            
            return q1_loss + q2_loss
        
        loss, grads = jax.value_and_grad(critic_loss_fn)(state['critic_params'])
        
        updates, new_opt_state = critic_tx.update(grads, state['critic_opt_state'], state['critic_params'])
        new_params = optax.apply_updates(state['critic_params'], updates)
        
        new_state = state.copy()
        new_state['critic_params'] = new_params
        new_state['critic_opt_state'] = new_opt_state
        
        return new_state, loss
    
    return critic_step


@jax.jit
def soft_update(target_params, source_params, tau):
    return jax.tree.map(
        lambda t, s: (1 - tau) * t + tau * s,
        target_params,
        source_params
    )


def make_get_action_fn(actor_net):
    @jax.jit
    def get_action(actor_params, obs, noise_scale, key, deterministic=False):
        action = actor_net.apply(actor_params, obs)
        if deterministic:
            return action
        
        noise = jrandom.normal(key, action.shape) * noise_scale
        return jnp.clip(action + noise, -1.0, 1.0)
    return get_action


def main():
    args = get_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jrandom.PRNGKey(args.seed)
    
    print(f"Using JAX backend: {jax.default_backend()}")
    
    if args.env_name.startswith("Maniskill-"):
        from fast_td3.environments.maniskill_env import ManiskillEnv
        env_name = args.env_name[len("Maniskill-"):]
        envs = ManiskillEnv(env_name, args.num_envs, args.seed)
    else:
        raise ValueError("This minimal script only supports ManiSkill environments")

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    n_critic_obs = n_obs
    args.n_obs = n_obs
    args.n_act = n_act
    
    actor_net = Actor(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=args.actor_hidden_dim,
        init_scale=args.init_scale,
    )
    
    critic_net = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
    )
    
    key, init_key = jrandom.split(key)
    state, actor_tx, critic_tx = create_train_state(init_key, actor_net, critic_net, args)
    
    # Create JIT-compiled functions
    get_action = make_get_action_fn(actor_net)
    actor_step = make_actor_step_fn(actor_net, critic_net, actor_tx)
    critic_step = make_critic_step_fn(actor_net, critic_net, critic_tx)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=False,
        playground_mode=False,
        maniskill_mode=True,
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )
    
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=(n_obs,))
    else:
        obs_normalizer = None
    
    obs = envs.reset()
    if type(obs) == tuple:
        obs = obs[0]
    
    global_step = 0
    dones = None
    noise_scale = 0.3
    last_log_time = 0
    log_interval = 1000
    
    print(f"Starting JAX training for {args.total_timesteps} steps...")
    
    while global_step < args.total_timesteps:
        mark_step()
        
        key, action_key = jrandom.split(key)
        obs_jax = _torch_to_jax(obs)
        
        if obs_normalizer is not None:
            obs_jax = obs_normalizer.normalize(obs_jax)
        
        actions_jax = get_action(
            state['actor_params'], 
            obs_jax, 
            noise_scale, 
            action_key
        )
        actions = _jax_to_torch(actions_jax).to(device)
        
        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]
        
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        
        transition = TensorDict(
            {
                "observations": obs,
                "actions": actions,
                "next": {
                    "observations": true_next_obs,
                    "rewards": rewards,
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        rb.extend(transition)
        
        obs = next_obs
        
        if obs_normalizer is not None:
            obs_normalizer.update(_torch_to_jax(obs))
        
        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                
                batch_jax = {
                    'observations': _torch_to_jax(data["observations"]),
                    'actions': _torch_to_jax(data["actions"]),
                    'next_observations': _torch_to_jax(data["next"]["observations"]),
                    'rewards': _torch_to_jax(data["next"]["rewards"]),
                    'bootstrap': _torch_to_jax((data["next"]["truncations"] | ~data["next"]["dones"]).float()),
                    'effective_n_steps': jnp.ones(data["next"]["rewards"].shape[0]),
                }
                
                if obs_normalizer is not None:
                    batch_jax['observations'] = obs_normalizer.normalize(batch_jax['observations'])
                    batch_jax['next_observations'] = obs_normalizer.normalize(batch_jax['next_observations'])
                
                key, critic_key = jrandom.split(key)
                state, critic_loss = critic_step(
                    state, batch_jax, args.policy_noise, args.noise_clip, 
                    args.gamma, args.use_cdq, critic_key
                )
                
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        state, actor_loss = actor_step(state, batch_jax, args.use_cdq)
                else:
                    if global_step % args.policy_frequency == 0:
                        state, actor_loss = actor_step(state, batch_jax, args.use_cdq)
                
                state['critic_target_params'] = soft_update(
                    state['critic_target_params'],
                    state['critic_params'],
                    args.tau
                )
        
        global_step += 1
        
        if global_step - last_log_time >= log_interval:
            print(f"Step {global_step}/{args.total_timesteps}, Reward: {rewards.mean().item():.3f}")
            last_log_time = global_step
    
    print("JAX training completed!")


if __name__ == "__main__":
    main()