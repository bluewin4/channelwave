"""
ChannelWave Environment for PufferLib

A high-performance environment for studying phase-locking and synchronization
dynamics in reinforcement learning, inspired by periodic reward experiments.

Key Concepts:
- N channels, each with +, -, and Null positions
- Rewards appear as square waves with configurable period and duty cycle
- Agents must learn to synchronize with reward patterns
- Null position provides observability (like being at center of hallway)

Spawn Modes:
- "deterministic" (0): Single channel with deterministic alternation pattern
- "random" (1): Single channel, random +/- sign each spawn
- "independent" (2): Each channel has independent timing (for Arnold Tongue experiments)

This environment maps to the "Periodic Hearts" experimental framework:
- channel_period = T (total period in ticks)
- channel_produce = T_produce (duty cycle high duration)
- channel_phase = φ (phase offset)
- t_m = d_h / r_a (movement time = critical period)
- max_steps = L (episode length)
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.channelwave import binding


class ChannelWave(pufferlib.PufferEnv):
    """High-throughput channel signal game for phase-locking experiments.

    Instance-based layout: The environment is structured as num_instances
    independent copies, each with channels_per_instance channels and
    agents_per_instance agents. Agents can only interact with channels
    in their own instance.
    
    Args:
        num_instances: Number of independent environment copies (for parallel training)
        channels_per_instance: Channels per instance (each agent's "world")
        agents_per_instance: Agents per instance (sharing those channels)
        max_steps: Episode length in ticks (L)
        
        # Timing parameters
        default_period: Default T for all channels (ticks per cycle)
        default_produce: Default T_produce (ticks reward is available)
        channel_periods: Optional list of per-channel periods [T1, T2, ...]
        channel_produces: Optional list of per-channel produce times
        channel_phases: Optional list of per-channel phase offsets [φ1, φ2, ...]
        
        # Movement
        t_m: Movement cooldown (critical period T_c)
        move_penalty: Step cost R_S (typically 0)
        
        # Rewards
        heart_reward: Reward per heart collected R_H
        reward_mode: "discrete" (once per heart) or "continuous" (every tick)
        
        # Spawn mode
        spawn_mode: "deterministic", "random", or "independent"
        alternation_pattern: Custom pattern for deterministic mode
    """

    def __init__(
        self,
        num_instances: int = 1,
        channels_per_instance: int = 1,
        agents_per_instance: int = 1,
        max_steps: int = 500,
        # Timing
        default_period: int = 40,
        default_produce: int = None,  # Auto-compute as period//2 if None
        channel_periods: list = None,
        channel_produces: list = None,
        channel_phases: list = None,
        # Movement
        t_m: int = 10,
        move_penalty: float = 0.0,
        # Rewards
        heart_reward: float = 1.0,
        reward_mode: str = "discrete",
        # Spawn mode
        spawn_mode: str = "deterministic",
        alternation_pattern: list = None,
        # PufferLib standard
        buf=None,
        seed: int = 0,
        render_mode: str = None,
    ):
        # Auto-compute produce as 50% duty cycle if not specified
        if default_produce is None:
            default_produce = default_period // 2
            
        total_agents = num_instances * agents_per_instance
        
        self.num_instances = num_instances
        self.channels_per_instance = channels_per_instance
        self.agents_per_instance = agents_per_instance
        self.num_agents = total_agents
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(6,), dtype=np.uint8
        )
        # Actions: (channel_offset, position_idx)
        # channel_offset is relative to agent's instance (0 to channels_per_instance-1)
        # position_idx: 0=minus, 1=null, 2=plus
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [channels_per_instance, 3], dtype=np.int32
        )
        self.render_mode = render_mode

        super().__init__(buf)

        # Convert string modes to int
        reward_mode_int = 0 if reward_mode == "discrete" else 1
        spawn_mode_map = {"deterministic": 0, "random": 1, "independent": 2}
        spawn_mode_int = spawn_mode_map.get(spawn_mode, 0)

        # Store config for introspection
        self._config = {
            "num_instances": num_instances,
            "channels_per_instance": channels_per_instance,
            "agents_per_instance": agents_per_instance,
            "total_agents": total_agents,
            "total_channels": num_instances * channels_per_instance,
            "max_steps": max_steps,
            "default_period": default_period,
            "default_produce": default_produce,
            "t_m": t_m,
            "spawn_mode": spawn_mode,
            "critical_period": t_m,  # T_c = t_m in this abstraction
        }

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            1,  # Single environment that handles all agents internally
            seed,
            num_instances=num_instances,
            channels_per_instance=channels_per_instance,
            agents_per_instance=agents_per_instance,
            max_steps=max_steps,
            default_period=default_period,
            default_produce=default_produce,
            channel_periods=list(channel_periods) if channel_periods else None,
            channel_produces=list(channel_produces) if channel_produces else None,
            channel_phases=list(channel_phases) if channel_phases else None,
            t_m=t_m,
            move_penalty=float(move_penalty),
            heart_reward=float(heart_reward),
            reward_mode=reward_mode_int,
            spawn_mode=spawn_mode_int,
            alternation_pattern=list(alternation_pattern) if alternation_pattern else None,
        )

    @property
    def config(self):
        """Return environment configuration for logging."""
        return self._config

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)


# ============================================================================
# Convenience constructors for specific experimental setups
# ============================================================================

def make_hallway(
    distance: int = 10,
    period: int = None,
    produce: int = None,
    duty_cycle: float = 0.5,
    num_instances: int = 1,
    max_steps: int = 500,
    **kwargs
):
    """Create a single-channel hallway experiment (primary experiment from doc).
    
    Args:
        distance: d_h - hallway distance (determines critical period T_c = distance)
        period: T - square wave period. If None, defaults to 2 * distance
        produce: T_produce - time heart is available. If None, uses duty_cycle
        duty_cycle: Fraction of period that heart is available (default 0.5)
        num_instances: Number of independent agent/channel pairs (for parallel training)
        max_steps: Episode length L
    
    The critical period is T_c = distance (since r_a = 1 in this abstraction).
    Each instance has 1 channel and 1 agent.
    """
    if period is None:
        period = 2 * distance  # Default: agent can just make it with perfect timing
    if produce is None:
        produce = int(period * duty_cycle)
    
    return ChannelWave(
        num_instances=num_instances,
        channels_per_instance=1,
        agents_per_instance=1,
        max_steps=max_steps,
        default_period=period,
        default_produce=produce,
        t_m=distance,  # Critical period = movement time
        spawn_mode="deterministic",
        **kwargs
    )


def make_arnold_tongue_pair(
    f1: float,
    f2: float,
    phase_offset: int = None,
    distance: int = 10,
    duty_cycle: float = 0.5,
    num_instances: int = 1,
    max_steps: int = 500,
    **kwargs
):
    """Create a 2-channel experiment for Arnold Tongue analysis.
    
    Args:
        f1: Frequency of channel 0 (cycles per tick)
        f2: Frequency of channel 1 (cycles per tick)
        phase_offset: Phase offset for channel 1 (if None, anti-phase)
        distance: Movement time / critical period
        duty_cycle: Duty cycle for both channels
        num_instances: Number of independent copies (for parallel training)
        
    Returns:
        ChannelWave with num_instances copies, each having 2 independently-timed channels.
    """
    T1 = max(1, int(1.0 / f1))
    T2 = max(1, int(1.0 / f2))
    
    produce1 = max(1, int(T1 * duty_cycle))
    produce2 = max(1, int(T2 * duty_cycle))
    
    if phase_offset is None:
        phase_offset = T2 // 2  # Anti-phase by default
    
    return ChannelWave(
        num_instances=num_instances,
        channels_per_instance=2,
        agents_per_instance=1,
        max_steps=max_steps,
        default_period=T1,
        default_produce=produce1,
        channel_periods=[T1, T2],
        channel_produces=[produce1, produce2],
        channel_phases=[0, phase_offset],
        t_m=distance,
        spawn_mode="independent",
        **kwargs
    )


def make_four_arm_cross(
    pattern: str = "alternating",  # or "paired"
    period: int = 40,
    distance: int = 10,
    agents_per_instance: int = 2,
    num_instances: int = 1,
    max_steps: int = 500,
    **kwargs
):
    """Create a 4-channel cross experiment (multi-agent secondary experiment).
    
    Args:
        pattern: "alternating" = [H1,H2,H1,H2] or "paired" = [H1,H1,H2,H2]
        period: Cycle period for all channels
        distance: d_side (adjacent channel distance)
        agents_per_instance: Number of competing/cooperating agents per cross
        num_instances: Number of independent cross setups (for parallel training)
    """
    produce = period // 2
    
    if pattern == "alternating":
        # [H1, H2, H1, H2] - adjacent channels have different "colors"
        phases = [0, period // 4, period // 2, 3 * period // 4]
    else:  # paired
        # [H1, H1, H2, H2] - adjacent channels have same "color"
        phases = [0, 0, period // 2, period // 2]
    
    return ChannelWave(
        num_instances=num_instances,
        channels_per_instance=4,
        agents_per_instance=agents_per_instance,
        max_steps=max_steps,
        default_period=period,
        default_produce=produce,
        channel_phases=phases,
        t_m=distance,
        spawn_mode="independent",
        **kwargs
    )


# ============================================================================
# Performance test
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Testing instance-based ChannelWave (128 independent agents)...")
    print("=" * 60)
    
    # 128 instances, each with 1 channel and 1 agent = 128 independent learners
    N = 128
    env = ChannelWave(
        num_instances=N,
        channels_per_instance=1,
        agents_per_instance=1,
        max_steps=500,
        spawn_mode="deterministic"
    )
    env.reset()
    print(f"Config: {env.config}")
    
    CACHE = 1024
    actions = np.zeros((CACHE, N, 2), dtype=np.int32)
    actions[..., 0] = 0  # Channel offset 0 (only channel in instance)
    actions[..., 1] = np.random.randint(0, 3, size=(CACHE, N))  # Random position
    
    steps = 0
    start = time.time()
    while time.time() - start < 5:
        env.step(actions[steps % CACHE])
        steps += 1
    
    sps = int(env.num_agents * steps / (time.time() - start))
    print(f"ChannelWave SPS (128 instances): {sps:,}")
    env.close()
    
    print("\n" + "=" * 60)
    print("Testing hallway experiment...")
    print("=" * 60)
    
    env = make_hallway(distance=10, period=40, num_instances=1, max_steps=100)
    obs, _ = env.reset()
    print(f"Config: {env.config}")
    print(f"Observation shape: {obs.shape}")
    
    total_reward = 0
    for _ in range(100):
        # Simple policy: stay at + position
        action = np.array([[0, 2]])  # Channel 0, position +
        obs, rewards, terminals, truncations, info = env.step(action)
        total_reward += rewards[0]
        if terminals[0]:
            print(f"Episode ended. Total reward: {total_reward:.2f}")
            print(f"Info: {info}")
            break
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Testing high-throughput (4096 instances)...")
    print("=" * 60)
    
    N = 4096
    env = ChannelWave(
        num_instances=N,
        channels_per_instance=1,
        agents_per_instance=1,
        max_steps=500,
        spawn_mode="deterministic"
    )
    env.reset()
    
    actions = np.zeros((CACHE, N, 2), dtype=np.int32)
    actions[..., 0] = 0
    actions[..., 1] = np.random.randint(0, 3, size=(CACHE, N))
    
    steps = 0
    start = time.time()
    while time.time() - start < 5:
        env.step(actions[steps % CACHE])
        steps += 1
    
    sps = int(env.num_agents * steps / (time.time() - start))
    print(f"ChannelWave SPS (4096 instances): {sps:,}")
    env.close()
    
    print("\nAll tests passed!")
