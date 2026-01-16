#!/usr/bin/env python3
"""
Training script for ChannelWave phase-locking experiments.

Usage:
    python -m pufferlib.ocean.channelwave.train [options]
    
Examples:
    # Basic training with LSTM
    python -m pufferlib.ocean.channelwave.train --use-lstm
    
    # Test different LSTM initializations
    python -m pufferlib.ocean.channelwave.train --use-lstm --init-strategy diagonal
    python -m pufferlib.ocean.channelwave.train --use-lstm --init-strategy oscillatory
    
    # Vary the period to test critical period hypothesis
    python -m pufferlib.ocean.channelwave.train --period 20 --distance 10  # T = 2*T_c, easy
    python -m pufferlib.ocean.channelwave.train --period 12 --distance 10  # T ≈ T_c, hard
    
    # Arnold Tongue: 2 channels with different frequencies
    python -m pufferlib.ocean.channelwave.train --two-channel --f1 0.025 --f2 0.05
"""

import argparse
import time
import numpy as np
import torch

import pufferlib
import pufferlib.vector

from pufferlib.ocean.channelwave.channelwave import (
    ChannelWave,
    make_hallway,
    make_arnold_tongue_pair,
)
from pufferlib.ocean.channelwave.torch import (
    ChannelWavePolicy,
    ChannelWaveLSTM,
    make_policy,
)


def make_env(args):
    """Create environment based on args."""
    if args.two_channel:
        return make_arnold_tongue_pair(
            f1=args.f1,
            f2=args.f2,
            distance=args.distance,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            heart_reward=args.heart_reward,
            move_penalty=args.move_penalty,
        )
    else:
        return make_hallway(
            distance=args.distance,
            period=args.period,
            produce=args.produce,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            heart_reward=args.heart_reward,
            move_penalty=args.move_penalty,
        )


def train(args):
    """Main training loop."""
    print(f"\n{'='*60}")
    print("ChannelWave Phase-Locking Experiment")
    print(f"{'='*60}")
    
    # Create environment
    env = make_env(args)
    print(f"\nEnvironment config: {env.config}")
    print(f"Critical period T_c = {env.config['critical_period']}")
    if not args.two_channel:
        print(f"Period T = {args.period} ({args.period / env.config['critical_period']:.2f} * T_c)")
    
    # Create policy
    if args.use_lstm:
        oscillator_freqs = None
        if args.init_strategy in ["oscillatory", "frequency_prior"]:
            # Set oscillator frequencies based on environment periods
            if args.two_channel:
                oscillator_freqs = [args.f1, args.f2, args.f1/2, args.f2/2]
            else:
                T = args.period
                oscillator_freqs = [1/T, 2/T, 1/(2*T), 1/(4*T)]
        
        policy = make_policy(
            env,
            use_lstm=True,
            hidden_size=args.hidden_size,
            init_strategy=args.init_strategy,
            oscillator_frequencies=oscillator_freqs,
        )
        print(f"\nUsing LSTM with '{args.init_strategy}' initialization")
        if oscillator_freqs:
            print(f"Oscillator frequencies: {oscillator_freqs}")
    else:
        policy = make_policy(env, use_lstm=False, hidden_size=args.hidden_size)
        print("\nUsing MLP (no LSTM)")
    
    policy = policy.to(args.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # Vectorized environment setup
    print(f"\nCreating {args.num_envs} vectorized environments...")
    
    # Create a factory that accepts the buf/seed args from vector.make
    def env_factory(buf=None, seed=0):
        if args.two_channel:
            return make_arnold_tongue_pair(
                f1=args.f1,
                f2=args.f2,
                distance=args.distance,
                num_agents=args.num_agents,
                max_steps=args.max_steps,
                heart_reward=args.heart_reward,
                move_penalty=args.move_penalty,
                buf=buf,
                seed=seed,
            )
        else:
            return make_hallway(
                distance=args.distance,
                period=args.period,
                produce=args.produce,
                num_agents=args.num_agents,
                max_steps=args.max_steps,
                heart_reward=args.heart_reward,
                move_penalty=args.move_penalty,
                buf=buf,
                seed=seed,
            )
    
    # Use Serial backend for simplicity (no pickling issues)
    vecenv = pufferlib.vector.make(
        env_factory,
        num_envs=args.num_envs,
        backend=pufferlib.vector.Serial,
    )
    
    # Training state
    obs = vecenv.reset()[0]
    total_steps = 0
    episodes_completed = 0
    start_time = time.time()
    
    # LSTM state
    if args.use_lstm:
        lstm_h = torch.zeros(args.num_envs * args.num_agents, args.hidden_size, device=args.device)
        lstm_c = torch.zeros(args.num_envs * args.num_agents, args.hidden_size, device=args.device)
    
    # Logging
    episode_returns = []
    episode_lengths = []
    hearts_collected = []
    sync_accuracies = []
    
    print(f"\nStarting training for {args.total_steps:,} steps...")
    print(f"Logging every {args.log_interval} steps")
    print()
    
    while total_steps < args.total_steps:
        # Collect rollout
        batch_obs = []
        batch_actions = []
        batch_logprobs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        
        for step in range(args.rollout_steps):
            total_steps += args.num_envs * args.num_agents
            
            # Convert observation to tensor
            obs_t = torch.tensor(obs, device=args.device, dtype=torch.float32)
            
            with torch.no_grad():
                if args.use_lstm:
                    state = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
                    logits, values = policy.forward_eval(obs_t, state)
                    lstm_h = state['lstm_h']
                    lstm_c = state['lstm_c']
                else:
                    logits, values = policy(obs_t)
                
                # Sample actions from multidiscrete logits
                actions = []
                logprobs = []
                for i, logit in enumerate(logits):
                    dist = torch.distributions.Categorical(logits=logit)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    actions.append(action)
                    logprobs.append(logprob)
                
                actions = torch.stack(actions, dim=1)  # (batch, num_action_dims)
                logprobs = torch.stack(logprobs, dim=1).sum(dim=1)  # Sum log probs
            
            # Step environment
            actions_np = actions.cpu().numpy().astype(np.int32)
            next_obs, rewards, dones, truncs, infos = vecenv.step(actions_np)
            
            # Store batch
            batch_obs.append(obs_t)
            batch_actions.append(actions)
            batch_logprobs.append(logprobs)
            batch_rewards.append(torch.tensor(rewards, device=args.device))
            batch_dones.append(torch.tensor(dones, device=args.device))
            batch_values.append(values.squeeze())
            
            # Reset LSTM state for done episodes
            if args.use_lstm:
                done_mask = torch.tensor(dones, device=args.device, dtype=torch.bool)
                lstm_h = lstm_h * (~done_mask).float().unsqueeze(1)
                lstm_c = lstm_c * (~done_mask).float().unsqueeze(1)
            
            # Collect episode stats
            for info in infos:
                if info and 'score' in info:
                    episode_returns.append(info.get('score', 0))
                    episode_lengths.append(info.get('episode_length', 0))
                    hearts_collected.append(info.get('hearts_collected', 0))
                    sync_accuracies.append(info.get('sync_accuracy', 0))
                    episodes_completed += info.get('n', 1)
            
            obs = next_obs
        
        # Stack batch
        batch_obs = torch.stack(batch_obs)
        batch_actions = torch.stack(batch_actions)
        batch_logprobs = torch.stack(batch_logprobs)
        batch_rewards = torch.stack(batch_rewards)
        batch_dones = torch.stack(batch_dones)
        batch_values = torch.stack(batch_values)
        
        # Compute advantages (simple version - no GAE for simplicity)
        with torch.no_grad():
            returns = batch_rewards.clone()
            not_done = (~batch_dones.bool()).float()
            # Simple MC returns
            for t in reversed(range(len(returns) - 1)):
                returns[t] = returns[t] + args.gamma * returns[t + 1] * not_done[t]
            advantages = returns - batch_values
        
        # Flatten batch
        T, B = batch_obs.shape[:2]
        flat_obs = batch_obs.reshape(T * B, -1)
        flat_actions = batch_actions.reshape(T * B, -1)
        flat_logprobs = batch_logprobs.reshape(T * B)
        flat_returns = returns.reshape(T * B)
        flat_advantages = advantages.reshape(T * B)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(args.update_epochs):
            # Shuffle
            indices = torch.randperm(T * B, device=args.device)
            
            for start in range(0, T * B, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idx = indices[start:end]
                
                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_logprobs = flat_logprobs[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                
                # Forward pass (no LSTM state for training minibatches)
                if args.use_lstm:
                    # For simplicity, use feedforward during PPO updates
                    # (proper implementation would maintain LSTM state)
                    new_logits, new_values = policy.policy(mb_obs)
                else:
                    new_logits, new_values = policy(mb_obs)
                
                # Compute new log probs
                new_logprobs = []
                entropies = []
                for i, logit in enumerate(new_logits):
                    dist = torch.distributions.Categorical(logits=logit)
                    new_logprobs.append(dist.log_prob(mb_actions[:, i]))
                    entropies.append(dist.entropy())
                
                new_logprobs = torch.stack(new_logprobs, dim=1).sum(dim=1)
                entropy = torch.stack(entropies, dim=1).mean()
                
                # PPO loss
                logratio = new_logprobs - mb_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((new_values.squeeze() - mb_returns) ** 2).mean()
                
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # Logging
        if total_steps % args.log_interval < args.num_envs * args.num_agents * args.rollout_steps:
            elapsed = time.time() - start_time
            sps = total_steps / elapsed
            
            print(f"Steps: {total_steps:>10,} | "
                  f"SPS: {sps:>8,.0f} | "
                  f"Episodes: {episodes_completed:>6} | ", end="")
            
            if episode_returns:
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                avg_hearts = np.mean(hearts_collected[-100:])
                avg_sync = np.mean(sync_accuracies[-100:])
                
                print(f"Return: {avg_return:>6.2f} | "
                      f"Hearts: {avg_hearts:>5.1f} | "
                      f"Sync: {avg_sync:>5.1%} | "
                      f"Len: {avg_length:>5.0f}")
            else:
                print()
    
    # Final stats
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Episodes completed: {episodes_completed}")
    
    if episode_returns:
        print(f"\nFinal 100-episode stats:")
        print(f"  Return: {np.mean(episode_returns[-100:]):.2f} ± {np.std(episode_returns[-100:]):.2f}")
        print(f"  Hearts: {np.mean(hearts_collected[-100:]):.2f}")
        print(f"  Sync accuracy: {np.mean(sync_accuracies[-100:]):.1%}")
    
    vecenv.close()
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Train ChannelWave phase-locking agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment
    parser.add_argument("--distance", type=int, default=10,
                       help="Movement time / critical period T_c")
    parser.add_argument("--period", type=int, default=40,
                       help="Square wave period T (single-channel mode)")
    parser.add_argument("--produce", type=int, default=None,
                       help="Heart availability time (default: period/2)")
    parser.add_argument("--num-agents", type=int, default=1,
                       help="Agents per environment")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Episode length L")
    parser.add_argument("--heart-reward", type=float, default=1.0,
                       help="Reward per heart R_H")
    parser.add_argument("--move-penalty", type=float, default=0.0,
                       help="Step penalty R_S")
    
    # Two-channel mode (Arnold Tongue)
    parser.add_argument("--two-channel", action="store_true",
                       help="Use 2-channel Arnold Tongue mode")
    parser.add_argument("--f1", type=float, default=0.025,
                       help="Frequency of channel 1 (cycles/tick)")
    parser.add_argument("--f2", type=float, default=0.025,
                       help="Frequency of channel 2 (cycles/tick)")
    
    # Policy
    parser.add_argument("--use-lstm", action="store_true",
                       help="Use LSTM policy")
    parser.add_argument("--hidden-size", type=int, default=128,
                       help="Hidden layer size")
    parser.add_argument("--init-strategy", type=str, default="diagonal",
                       choices=["orthogonal", "diagonal", "oscillatory", "frequency_prior"],
                       help="LSTM initialization strategy")
    
    # Training
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                       help="Total training steps")
    parser.add_argument("--rollout-steps", type=int, default=128,
                       help="Steps per rollout")
    parser.add_argument("--update-epochs", type=int, default=4,
                       help="PPO update epochs")
    parser.add_argument("--minibatch-size", type=int, default=128,
                       help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                       help="PPO clip coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                       help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                       help="Max gradient norm")
    
    # Misc
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    parser.add_argument("--log-interval", type=int, default=10000,
                       help="Log every N steps")
    
    args = parser.parse_args()
    
    if args.produce is None:
        args.produce = args.period // 2
    
    train(args)


if __name__ == "__main__":
    main()
