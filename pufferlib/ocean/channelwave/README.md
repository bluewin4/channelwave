# ChannelWave: Phase-Locking & Entrainment Experiments

A high-performance environment for studying how RL agents synchronize with periodic reward patterns—connecting reinforcement learning to biological oscillators, foraging theory, and the Arnold Tongue framework from dynamical systems.

## Motivation

How do agents learn to track and predict periodic signals? This is fundamental to:
- **Biological rhythms**: Circadian cycles, motor pattern generators, neural oscillators
- **Foraging theory**: Patch-leaving decisions, resource renewal cycles
- **Entrainment**: How oscillating systems lock to external frequencies

ChannelWave provides a minimal, high-throughput testbed (100M+ SPS) for these questions.

## The Game

```
        +  ←── Reward appears here (positive sign)
        |
  Null ─┼── Agent starts here (can see reward location)
        |
        -  ←── Or here (negative sign)
```

**Rules:**
1. **N channels**, each with +, -, and Null positions
2. **Square wave rewards**: Appear every `T` timesteps, persist for `T_produce` timesteps
3. **M agents** choose position (+, -, or Null) with movement cooldown `t_m`
4. **Observability tradeoff**: Null position reveals reward location; +/- positions collect rewards

**Critical Period**: `T_c = t_m` — the minimum time to switch positions. When `T < T_c`, perfect synchronization is impossible.

## Quick Start

```python
from pufferlib.ocean.channelwave import ChannelWave, make_hallway

# Basic single-channel (hallway) experiment
env = make_hallway(distance=10, period=40)  # T = 4 * T_c (easy)
obs, _ = env.reset()

for _ in range(1000):
    action = env.single_action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
```

## Key Parameters

| Parameter | Document Symbol | Description |
|-----------|----------------|-------------|
| `t_m` | T_c | Movement cooldown (critical period) |
| `default_period` | T | Square wave period |
| `default_produce` | T_produce | Reward availability time |
| `max_steps` | L | Episode length |
| `heart_reward` | R_H | Reward per collection |
| `move_penalty` | R_S | Step cost |

## Experimental Regimes

| Regime | Period | Difficulty | Notes |
|--------|--------|------------|-------|
| Easy | T = 4×T_c | Low | Agent has plenty of time |
| Medium | T = 2×T_c | Medium | Requires efficient switching |
| Hard | T ≈ 1.2×T_c | High | Tight timing required |
| Critical | T = T_c | Impossible | Cannot catch all rewards |

## Spawn Modes

- **`deterministic`**: Alternating +/- pattern (primary experiment)
- **`random`**: Random sign each spawn
- **`independent`**: Each channel has its own frequency (Arnold Tongue experiments)

## Arnold Tongue Experiments

For studying entrainment across frequency space:

```python
from pufferlib.ocean.channelwave import make_arnold_tongue_pair

# Two channels with different frequencies
env = make_arnold_tongue_pair(
    f1=1/40,  # Channel 0: period 40
    f2=1/30,  # Channel 1: period 30
    distance=10,
)
```

Sweep across (f1, f2) space to construct Arnold Tongue diagrams showing learning difficulty as a function of frequency ratio.

## LSTM Initialization

For phase-locking tasks, LSTM hidden state must maintain oscillatory patterns. We provide custom initializations:

```python
from pufferlib.ocean.channelwave.torch import make_policy

# Diagonal initialization: strengthens memory persistence
policy = make_policy(env, use_lstm=True, init_strategy="diagonal")

# Oscillatory initialization: embeds specific frequencies
policy = make_policy(env, use_lstm=True, init_strategy="oscillatory",
                     oscillator_frequencies=[1/40, 1/20])
```

## Training

```bash
# Using PufferLib's training infrastructure
python -m pufferlib.pufferl train puffer_channelwave

# With specific preset
python -m pufferlib.pufferl train puffer_channelwave --config hard

# Sweep across periods
python -m pufferlib.pufferl sweep puffer_channelwave
```

## Metrics

- **`hearts_collected`**: Total rewards obtained
- **`hearts_missed`**: Rewards that expired uncollected  
- **`sync_accuracy`**: Fraction of time at correct position when reward active
- **`episode_length`**: Steps per episode

## Theoretical Predictions

From the document:

1. **Critical Period Hypothesis**: No stable synchronization below T_c = t_m
2. **Learning Speed**: Convergence time increases as T → T_c from above
3. **Cheater Strategies**: Simple reactive policies may achieve near-optimal performance for certain frequency ratios (high "cheater gap")

## References

- Arnold Tongue framework: [Wikipedia](https://en.wikipedia.org/wiki/Arnold_tongue)
- Entrainment in biological systems: [eLife paper](https://elifesciences.org/articles/79575)
- Foraging theory & patch-leaving: Marginal Value Theorem

## File Structure

```
channelwave/
├── __init__.py
├── channelwave.h      # C header - struct definitions
├── channelwave.c      # C implementation - game logic
├── binding.c          # Python/C binding
├── channelwave.py     # Python environment wrapper
├── torch.py           # Neural network policies (in ocean/torch.py)
├── train.py           # Standalone training script
└── README.md          # This file
```

## Performance

| Metric | Value |
|--------|-------|
| Steps per second (C core) | ~300M |
| Steps per second (with Python) | ~100M |
| Observation size | 6 bytes |
| Action size | 2 int32 |

## Contributing

This is part of PufferLib's Ocean suite of high-performance environments. Issues and PRs welcome!
