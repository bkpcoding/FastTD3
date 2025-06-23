"""Minimal GRPO training loop."""

from .hyperparams import get_args
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .grpo import Actor, calculate_network_norms
from .grpo_utils import GroupRolloutBuffer, save_grpo_params
import os
from tqdm import tqdm


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = args.amp and torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    torch.manual_seed(args.seed)

    envs, _, _ = make_env(
        args.env_name,
        seed=args.seed,
        num_envs=args.num_envs,
        num_eval_envs=1,
        device_rank=0,
    )
    obs = envs.reset()
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    n_act = envs.num_actions

    actor = Actor(n_obs, n_act, args.hidden_dim, device=device)
    actor_ref = Actor(n_obs, n_act, args.hidden_dim, device=device)
    actor_ref.load_state_dict(actor.state_dict())
    optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    buffer = GroupRolloutBuffer(args.group_size, device=device)

    global_step = 0
    episodes_data = [
        dict(obs=[], actions=[], logps=[], rewards=[]) for _ in range(args.num_envs)
    ]

    pbar = tqdm(total=args.total_timesteps)
    while global_step < args.total_timesteps:
        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalizer(obs)
            action, logp = actor.act(norm_obs)
        next_obs, reward, done, _ = envs.step(action)
        for i in range(args.num_envs):
            episodes_data[i]["obs"].append(obs[i].to(device))
            episodes_data[i]["actions"].append(action[i].to(device))
            episodes_data[i]["logps"].append(logp[i].to(device))
            episodes_data[i]["rewards"].append(reward[i].to(device))
        global_step += args.num_envs
        pbar.update(args.num_envs)
        for i in range(args.num_envs):
            if done[i]:
                buffer.add_episode(
                    episodes_data[i]["obs"],
                    episodes_data[i]["actions"],
                    episodes_data[i]["logps"],
                    episodes_data[i]["rewards"],
                    args.gamma,
                )
                episodes_data[i] = dict(obs=[], actions=[], logps=[], rewards=[])
        obs = next_obs
        if buffer.num_episodes >= args.group_size:
            buffer.compute_advantages()
            actor_ref.load_state_dict(actor.state_dict())
            for epoch in range(args.update_epochs):
                for b_obs, b_actions, b_logp, b_adv in buffer.get_batches(
                    args.batch_size
                ):
                    with autocast(
                        device_type=amp_device_type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        dist = actor.get_dist(normalizer(b_obs))
                        raw_actions = torch.atanh(torch.clamp(b_actions, -0.999, 0.999))
                        new_logp = dist.log_prob(raw_actions).sum(-1)
                        new_logp = new_logp - (
                            2
                            * (
                                torch.log(torch.tensor(2.0))
                                - raw_actions
                                - F.softplus(-2 * raw_actions)
                            )
                        ).sum(-1)
                        ratio = (new_logp - b_logp).exp()
                        pg_loss = -torch.min(
                            ratio * b_adv,
                            torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                            * b_adv,
                        ).mean()
                        ref_dist = actor_ref.get_dist(normalizer(b_obs))
                        kl = (
                            torch.distributions.kl.kl_divergence(dist, ref_dist)
                            .sum(-1)
                            .mean()
                        )
                        entropy = dist.entropy().sum(-1).mean()
                        loss = pg_loss + args.kl_coef * kl - args.ent_coef * entropy
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), args.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
            buffer.clear()

    save_path = os.path.join(args.output_dir, f"{args.env_name}_grpo_final.pt")
    save_grpo_params(global_step, actor, normalizer, args, save_path)


if __name__ == "__main__":
    main()
