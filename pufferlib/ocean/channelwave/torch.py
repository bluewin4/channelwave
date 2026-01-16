"""
Neural network policies for ChannelWave experiments.

The key insight for phase-locking is that LSTMs need to develop internal
oscillators that can synchronize with the environment's periodic rewards.

Custom initialization strategies:
1. Diagonal initialization: Initialize weight_hh with stronger diagonal to 
   encourage each hidden unit to maintain its own oscillatory pattern
2. Frequency-biased initialization: Initialize with complex eigenvalues near 
   the unit circle to create inherent oscillatory modes
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models
import pufferlib.pytorch

from pufferlib.models import Default as BasePolicy
Recurrent = pufferlib.models.LSTMWrapper


class ChannelWavePolicy(nn.Module):
    """MLP policy for ChannelWave with proper handling of multidiscrete actions.
    
    Observation space: (6,) uint8
        [reward_sign_if_visible, reward_timer_if_visible, agent_pos, move_cd, channel, spawn_timer]
    
    Action space: MultiDiscrete([num_channels, 3])
        [channel_idx, position_idx]  where position_idx: 0=minus, 1=null, 2=plus
    """
    
    def __init__(self, env, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        
        num_obs = np.prod(env.single_observation_space.shape)
        
        # Encoder: obs -> hidden
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )
        
        # MultiDiscrete action head
        self.action_nvec = tuple(env.single_action_space.nvec)
        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01
        )
        
        # Value head
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        logits, values = self.decode_actions(hidden)
        return logits, values
    
    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)
    
    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        obs = observations.view(batch_size, -1).float() / 255.0  # Normalize uint8
        return self.encoder(obs)
    
    def decode_actions(self, hidden):
        logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        values = self.value(hidden)
        return logits, values


class ChannelWaveLSTM(pufferlib.models.LSTMWrapper):
    """LSTM wrapper for ChannelWave with custom initialization options.
    
    Initialization strategies for oscillatory behavior:
    
    1. "orthogonal" (default): Standard orthogonal initialization
    2. "diagonal": Stronger diagonal in weight_hh to encourage memory persistence
    3. "oscillatory": Initialize weight_hh to have complex eigenvalues near unit circle
    4. "frequency_prior": Initialize with known frequency structure
    """
    
    def __init__(
        self, 
        env, 
        policy, 
        input_size=128, 
        hidden_size=128,
        init_strategy="diagonal",
        diagonal_scale=1.5,
        oscillator_frequencies=None,  # List of frequencies to embed
    ):
        super().__init__(env, policy, input_size, hidden_size)
        
        self.init_strategy = init_strategy
        
        if init_strategy == "diagonal":
            self._init_diagonal(diagonal_scale)
        elif init_strategy == "oscillatory":
            self._init_oscillatory(oscillator_frequencies)
        elif init_strategy == "frequency_prior":
            self._init_frequency_prior(oscillator_frequencies)
    
    def _init_diagonal(self, scale=1.5):
        """Initialize weight_hh with stronger diagonal.
        
        This encourages each hidden unit to remember its previous value,
        which helps maintain phase information across timesteps.
        """
        with torch.no_grad():
            # Get the hidden-to-hidden weight matrix
            weight_hh = self.lstm.weight_hh_l0
            h = self.hidden_size
            
            # LSTM has 4 gates: input, forget, cell, output
            # weight_hh is (4*h, h), we want to strengthen diagonal for each gate
            for gate in range(4):
                start = gate * h
                end = (gate + 1) * h
                # Add scaled identity to strengthen diagonal
                weight_hh[start:end, :] += scale * torch.eye(h, device=weight_hh.device)
            
            # Sync with the cell
            self.cell.weight_hh = self.lstm.weight_hh_l0
    
    def _init_oscillatory(self, frequencies=None):
        """Initialize weight_hh to encourage oscillatory dynamics.
        
        Creates rotation matrices in 2D subspaces of the hidden state,
        with eigenvalues near the unit circle at specified frequencies.
        """
        if frequencies is None:
            # Default: create oscillators at multiple frequencies
            frequencies = [1/20, 1/40, 1/80, 1/160]
        
        with torch.no_grad():
            weight_hh = self.lstm.weight_hh_l0
            h = self.hidden_size
            
            # Focus on forget gate (gate 1) for memory dynamics
            forget_start = h
            forget_end = 2 * h
            
            # Create rotation matrices in pairs of hidden units
            num_oscillators = min(len(frequencies), h // 2)
            
            for i, freq in enumerate(frequencies[:num_oscillators]):
                theta = 2 * math.pi * freq  # Angular frequency
                idx = 2 * i
                if idx + 1 < h:
                    # 2D rotation matrix with slight decay (eigenvalue magnitude < 1)
                    decay = 0.995  # Slight decay for stability
                    cos_t = decay * math.cos(theta)
                    sin_t = decay * math.sin(theta)
                    
                    # Set rotation in forget gate
                    weight_hh[forget_start + idx, idx] = cos_t
                    weight_hh[forget_start + idx, idx + 1] = -sin_t
                    weight_hh[forget_start + idx + 1, idx] = sin_t
                    weight_hh[forget_start + idx + 1, idx + 1] = cos_t
            
            self.cell.weight_hh = self.lstm.weight_hh_l0
    
    def _init_frequency_prior(self, frequencies=None):
        """Initialize with a mix of diagonal strengthening and oscillatory modes.
        
        This combines the benefits of both strategies:
        - Diagonal for general memory persistence
        - Oscillatory modes for specific frequencies
        """
        self._init_diagonal(scale=1.0)
        
        if frequencies:
            # Add oscillatory components
            with torch.no_grad():
                weight_hh = self.lstm.weight_hh_l0
                h = self.hidden_size
                
                # Use the last quarter of hidden units for oscillators
                oscillator_start = 3 * h // 4
                num_oscillators = min(len(frequencies), (h - oscillator_start) // 2)
                
                for gate in range(4):
                    gate_start = gate * h
                    
                    for i, freq in enumerate(frequencies[:num_oscillators]):
                        theta = 2 * math.pi * freq
                        idx = oscillator_start + 2 * i
                        
                        if idx + 1 < h:
                            decay = 0.99
                            cos_t = decay * math.cos(theta)
                            sin_t = decay * math.sin(theta)
                            
                            weight_hh[gate_start + idx, idx] += cos_t * 0.5
                            weight_hh[gate_start + idx, idx + 1] += -sin_t * 0.5
                            weight_hh[gate_start + idx + 1, idx] += sin_t * 0.5
                            weight_hh[gate_start + idx + 1, idx + 1] += cos_t * 0.5
                
                self.cell.weight_hh = self.lstm.weight_hh_l0


# Convenience aliases
Policy = ChannelWavePolicy
Recurrent = ChannelWaveLSTM


def make_policy(env, use_lstm=True, hidden_size=128, init_strategy="diagonal", **kwargs):
    """Factory function to create ChannelWave policy.
    
    Args:
        env: PufferLib environment
        use_lstm: Whether to wrap with LSTM
        hidden_size: Size of hidden layers
        init_strategy: LSTM initialization strategy
            - "orthogonal": Standard
            - "diagonal": Strengthen diagonal for memory
            - "oscillatory": Create oscillator modes
            - "frequency_prior": Combine diagonal + oscillatory
    """
    policy = ChannelWavePolicy(env, hidden_size=hidden_size)
    
    if use_lstm:
        return ChannelWaveLSTM(
            env, 
            policy, 
            input_size=hidden_size,
            hidden_size=hidden_size,
            init_strategy=init_strategy,
            **kwargs
        )
    return policy
