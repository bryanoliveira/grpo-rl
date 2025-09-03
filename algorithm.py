# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
import json

import yaml
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


from environment import make as make_env


def add_args(parser):
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="the name of this experiment")
    parser.add_argument("--run-name", type=str, default=None,
                        help="the name of this run")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="grpo",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="bryanoliveira",
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--env-configs", type=str, default=None,
                        help="the configs of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1100000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Toggles advantages normalization within the minibatch")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    # Baseline / value function control
    parser.add_argument("--use-value-fn", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle using value function and its loss for baseline and training")
    parser.add_argument("--baseline-type", type=str, default="value",
                        choices=["value", "constant", "uniform", "stats", "batch_mean", "ema"],
                        help="Baseline type for advantage calculation")
    parser.add_argument("--scale-adv-batch", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Toggle using batch std for advantage scaling")
    parser.add_argument("--baseline-constant", type=float, default=None,
                        help="Constant baseline when --baseline-type=constant")
    parser.add_argument("--baseline-uniform-low", type=float, default=None,
                        help="Low bound for --baseline-type=uniform; if unset tries env.reward_range or falls back to -1")
    parser.add_argument("--baseline-uniform-high", type=float, default=None,
                        help="High bound for --baseline-type=uniform; if unset tries env.reward_range or falls back to 1")
    parser.add_argument("--baseline-ema-beta", type=float, default=0.9,
                        help="EMA beta for --baseline-type=ema (Adam-style bias correction)")

    # to be filled in runtime
    parser.add_argument("--batch-size", type=int, default=0,
                        help="the batch size (computed in runtime)")
    parser.add_argument("--minibatch-size", type=int, default=0,
                        help="the mini-batch size (computed in runtime)")
    parser.add_argument("--num-iterations", type=int, default=0,
                        help="the number of iterations (computed in runtime)")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        if isinstance(envs.single_action_space, gym.spaces.Box):
            action_dim = np.prod(envs.single_action_space.shape)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            action_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if hasattr(self, "actor_logstd"):
            action_logstd = self.actor_logstd.expand_as(logits)
            action_std = torch.exp(action_logstd)
            probs = Normal(logits, action_std)
        else:
            probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        if hasattr(self, "actor_logstd"):
            logprob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        else:
            logprob = probs.log_prob(action.long())
            entropy = probs.entropy()

        return action, logprob, entropy, self.critic(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    add_args(parser)
    args = parser.parse_args()

    # Batch/iteration sizing (episode mode if num_steps==0)
    if args.num_steps > 0:
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(max(1, args.batch_size // args.num_minibatches))
        args.num_iterations = max(1, args.total_timesteps // max(1, args.batch_size))
    else:
        # Episode mode: dynamic batch size per iteration, stop by total timesteps
        args.batch_size = 0
        args.minibatch_size = 0
        args.num_iterations = max(1, args.total_timesteps)
    
    # Baseline helpers
    # Uniform baseline bounds
    if args.baseline_uniform_low is None or args.baseline_uniform_high is None:
        try:
            rlow, rhigh = envs.envs[0].reward_range  # type: ignore[attr-defined]
            if np.isfinite(rlow) and np.isfinite(rhigh):
                if args.baseline_uniform_low is None:
                    args.baseline_uniform_low = float(rlow)
                if args.baseline_uniform_high is None:
                    args.baseline_uniform_high = float(rhigh)
        except Exception:
            pass
    if args.baseline_uniform_low is None:
        args.baseline_uniform_low = -1.0
    if args.baseline_uniform_high is None:
        args.baseline_uniform_high = 1.0

    if args.run_name:
        run_name = args.run_name
        args.exp_name = "__".join(args.run_name.split("__")[:-1])
    else:
        run_name = f"{args.exp_name}__{args.env_id.replace('/', '_').replace('-', '_').lower()}__{args.seed}"
        
    if args.track:
        import wandb

        # Try to find the wandb run by name
        api = wandb.Api()
        try:
            runs = api.runs(f"{args.wandb_entity}/{args.wandb_project_name}", filters={"display_name": run_name})
            if runs:
                print(f"Wandb run {run_name} already exists, skipping")
                exit(0)
        except Exception as exception:
            print(f"Could not find wandb run by name: {exception}")

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=args.exp_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logdir = f"runs/{run_name}"
    print(f"Logging to {logdir}")
    os.makedirs(logdir, exist_ok=True)
    with open(f"{logdir}/config.yaml", "w") as f:
        yaml.dump(vars(args), f)
    print("Configs:")
    print(json.dumps(dict(sorted(vars(args).items())), indent=2))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.backends.mps.is_available() and args.cuda:
        device = torch.device("mps")
    elif torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # env setup
    args.env_configs = json.loads(args.env_configs) if args.env_configs else {}

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.env_configs, args.gamma) for i in range(args.num_envs)],
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # EMA state for baseline_type=ema (Adam-style bias correction)
    ema_m = 0.0
    ema_t = 0

    # ALGO Logic: Storage setup (only used when num_steps>0)
    obs = torch.zeros((max(args.num_steps, 1), args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((max(args.num_steps, 1), args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((max(args.num_steps, 1), args.num_envs)).to(device)
    rewards = torch.zeros((max(args.num_steps, 1), args.num_envs)).to(device)
    dones = torch.zeros((max(args.num_steps, 1), args.num_envs)).to(device)
    values = torch.zeros((max(args.num_steps, 1), args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print(f"Starting training for {args.num_iterations} iterations")
    print(agent)
    print(f"Device: {next(agent.parameters()).device}")
    dtype = next(agent.parameters()).dtype

    pbar = tqdm(range(1, args.num_iterations + 1), desc="Iterations", dynamic_ncols=True)
    for iteration in pbar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.num_steps > 0:
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, dtype=dtype).to(device).view(-1)
                next_obs, next_done = torch.tensor(next_obs_np, dtype=dtype).to(device), torch.tensor(next_done, dtype=dtype).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            pbar.set_description(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs[:args.num_steps].reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs[:args.num_steps].reshape(-1)
            b_actions = actions[:args.num_steps].reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages[:args.num_steps].reshape(-1)
            b_returns = returns[:args.num_steps].reshape(-1)
            b_values = values[:args.num_steps].reshape(-1)
            batch_size = b_obs.shape[0]
            minibatch_size = int(max(1, batch_size // args.num_minibatches))
        else:
            # Episode mode: collect one full episode from each env (no bootstrapping)
            per_env_obs = [[] for _ in range(args.num_envs)]
            per_env_actions = [[] for _ in range(args.num_envs)]
            per_env_logprobs = [[] for _ in range(args.num_envs)]
            per_env_rewards = [[] for _ in range(args.num_envs)]
            per_env_values = [[] for _ in range(args.num_envs)]
            finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

            while not bool(finished.all()):
                # Count only steps from envs still collecting
                global_step += int((~finished).sum().item())
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminations, truncations)
                reward_t = torch.tensor(reward, dtype=dtype, device=device).view(-1)

                # Record for envs still collecting
                for i in range(args.num_envs):
                    if not bool(finished[i]):
                        per_env_obs[i].append(next_obs[i].detach())
                        per_env_actions[i].append(action[i].detach())
                        per_env_logprobs[i].append(logprob[i].detach())
                        per_env_rewards[i].append(reward_t[i].detach())
                        per_env_values[i].append(value.flatten()[i].detach())
                        if bool(done[i]):
                            finished[i] = True

                # logging
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            pbar.set_description(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                next_obs = torch.tensor(next_obs_np, dtype=dtype, device=device)

                # Early stop if we reached total timesteps budget
                if global_step >= args.total_timesteps:
                    break

            # Build tensors
            b_obs_list = []
            b_actions_list = []
            b_logprobs_list = []
            b_values_list = []
            b_returns_list = []
            episode_returns = []  # per-episode sum of rewards (undiscounted)

            for i in range(args.num_envs):
                if len(per_env_rewards[i]) == 0:
                    continue
                # MC returns per step (discounted)
                R = 0.0
                returns_i = []
                for r in reversed(per_env_rewards[i]):
                    R = float(r) + args.gamma * R
                    returns_i.append(R)
                returns_i.reverse()

                # Flatten per env into batch lists
                b_obs_list.extend(per_env_obs[i])
                b_actions_list.extend(per_env_actions[i])
                b_logprobs_list.extend(per_env_logprobs[i])
                b_values_list.extend(per_env_values[i])
                b_returns_list.extend([torch.tensor(ret, dtype=dtype, device=device) for ret in returns_i])

                # track episode sum of rewards (undiscounted)
                episode_returns.append(float(torch.stack(per_env_rewards[i]).sum().item()))

            if len(b_obs_list) == 0:
                # No data collected (shouldn't happen), skip iteration
                continue

            b_obs = torch.stack(b_obs_list, dim=0)
            b_actions = torch.stack(b_actions_list, dim=0)
            b_logprobs = torch.stack(b_logprobs_list, dim=0)
            b_values = torch.stack(b_values_list, dim=0)
            b_returns = torch.stack(b_returns_list, dim=0)

            # Baseline selection for advantages
            if args.baseline_type == "value" and args.use_value_fn:
                baseline_vals = b_values
            elif args.baseline_type == "constant":
                baseline_vals = torch.full_like(b_returns, float(args.baseline_constant))
            elif args.baseline_type == "uniform":
                sampled = random.uniform(float(args.baseline_uniform_low), float(args.baseline_uniform_high))
                baseline_vals = torch.full_like(b_returns, float(sampled))
            elif args.baseline_type == "stats":
                if len(episode_returns) > 0:
                    mean_ret = float(np.mean(episode_returns))
                    std_ret = float(np.std(episode_returns))
                else:
                    mean_ret = 0.0
                    std_ret = 1.0
                # Sample baseline values from a normal distribution with mean and std of episode returns
                baseline_vals = torch.normal(
                    mean=torch.full_like(b_returns, mean_ret),
                    std=torch.full_like(b_returns, std_ret)
                )
            elif args.baseline_type == "batch_mean":
                if len(episode_returns) > 0:
                    mean_ret = float(np.mean(episode_returns))
                else:
                    mean_ret = 0.0
                baseline_vals = torch.full_like(b_returns, float(mean_ret))
            elif args.baseline_type == "ema":
                if len(episode_returns) > 0:
                    batch_mean = float(np.mean(episode_returns))
                    ema_t += 1
                    beta = float(args.baseline_ema_beta)
                    ema_m = beta * ema_m + (1.0 - beta) * batch_mean
                    corrected = ema_m / (1.0 - (beta ** ema_t))
                    baseline_vals = torch.full_like(b_returns, float(corrected))
                else:
                    baseline_vals = torch.zeros_like(b_returns)
            else:
                baseline_vals = torch.zeros_like(b_returns)

            b_advantages = b_returns - baseline_vals
            batch_size = b_obs.shape[0]
            minibatch_size = int(max(1, batch_size // args.num_minibatches))

        if args.scale_adv_batch:
            b_advantages /= (b_advantages.std() + 1e-8)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.use_value_fn:
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = torch.tensor(0.0, dtype=dtype, device=device)

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Explained variance (only meaningful when using value baseline)
        if args.use_value_fn and (args.num_steps > 0 or args.baseline_type == "value"):
            y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        else:
            explained_var = np.nan

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Early stop on total timesteps in episode mode
        if args.num_steps == 0 and global_step >= args.total_timesteps:
            break

    envs.close()
    writer.close()