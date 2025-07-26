import os
import sys
import random

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict

from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    mark_step,
)
from fast_td3.hyperparams import get_args
from fast_td3.fast_td3 import Actor, Critic

torch.set_float32_matmul_precision("high")
import torch._dynamo
torch._dynamo.config.suppress_errors = True


def main():
    args = get_args()
    
    # Basic setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    # AMP setup
    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    # Create ManiSkill environment
    if args.env_name.startswith("Maniskill-"):
        from fast_td3.environments.maniskill_env import ManiskillEnv
        env_name = args.env_name[len("Maniskill-"):]
        envs = ManiskillEnv(env_name, args.num_envs, args.seed)
    else:
        raise ValueError("This minimal script only supports ManiSkill environments")

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    # Create replay buffer
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

    # Create normalizers
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    # Create networks
    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    actor = Actor(**actor_kwargs)
    
    from tensordict import from_module
    actor_detach = Actor(**actor_kwargs)
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = Critic(**critic_kwargs)
    qnet_target = Critic(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    # Create optimizers
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def update_main(data):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = observations
            next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        scaler.step(q_optimizer)
        scaler.update()

    def update_pol(data):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = data["observations"]
            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        scaler.step(actor_optimizer)
        scaler.update()

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        update_main = torch.compile(update_main, mode=args.compile_mode)
        update_pol = torch.compile(update_pol, mode=args.compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward

    # Initialize environment
    obs = envs.reset()
    if type(obs) == tuple:
        obs = obs[0]

    global_step = 0
    dones = None

    print(f"Starting training for {args.total_timesteps} steps...")
    
    while global_step < args.total_timesteps:
        mark_step()
        
        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)
        
        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]
        
        # Compute 'true' next_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        rb.extend(transition)

        obs = next_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )

                update_main(data)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        update_pol(data)
                else:
                    if global_step % args.policy_frequency == 0:
                        update_pol(data)

                soft_update(qnet, qnet_target, args.tau)

        global_step += 1
        
        if global_step % 1000 == 0:
            print(f"Step {global_step}/{args.total_timesteps}, Reward: {rewards.mean().item():.3f}")

    print("Training completed!")


if __name__ == "__main__":
    main()