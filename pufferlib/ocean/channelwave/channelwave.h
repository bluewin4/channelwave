#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    float score;
    float hearts_collected;
    float hearts_missed;
    float sync_accuracy;      // Fraction of time agent was at correct position
    float episode_length;
    float n; // Required as the last field
} Log;

typedef struct {
    Log log;                     // Required field
    unsigned char* observations; // Required field. Matches numpy dtype in Python
    int* actions;                // Required field. Matches numpy dtype in Python
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    // Config - Instance-based layout
    int num_instances;         // Number of independent replications
    int channels_per_instance; // Channels per instance (user sets this)
    int agents_per_instance;   // Agents per instance (user sets this)
    int num_channels;          // Total = num_instances * channels_per_instance
    int num_agents;            // Total = num_instances * agents_per_instance
    int max_steps;             // Episode length (L in document)
    float move_penalty;        // R_S step penalty
    float heart_reward;        // R_H reward per heart
    unsigned char reward_mode; // 0: discrete-once, 1: continuous

    // Per-channel timing config (for Arnold Tongue experiments)
    int16_t* channel_period;   // T per channel (len=num_channels)
    int16_t* channel_produce;  // T_produce per channel (duty cycle high)
    int16_t* channel_phase;    // Initial phase offset per channel
    
    // Spawn mode: 0=deterministic alternating, 1=random channel, 2=all channels independent
    unsigned char spawn_mode;
    
    // For deterministic alternating mode (single-channel experiment)
    int8_t* alternation_pattern;  // e.g., [+1, -1, +1, -1] or custom
    int alternation_len;
    int alternation_idx;

    // Movement config
    int t_m;                   // move cooldown (analogous to d_h/r_a)

    // State
    int32_t step;
    int8_t* reward_pos;        // len=num_channels, values {-1,0,1}
    int16_t* reward_timer;     // len=num_channels, countdown until reward disappears
    int16_t* spawn_timer;      // len=num_channels, countdown until next spawn
    unsigned char* reward_claimed; // len=num_channels

    int8_t* agent_pos;         // len=num_agents, {-1,0,1}
    int16_t* agent_channel;    // len=num_agents, which channel agent is attending (supports up to 32767 channels)
    int16_t* move_cd;          // len=num_agents

    // Episode tracking per agent
    float* episode_return;     // len=num_agents
    int32_t* episode_step;     // len=num_agents
    int16_t* hearts_collected; // len=num_agents
    int16_t* hearts_missed;    // len=num_agents
    int16_t* sync_ticks;       // len=num_agents, ticks spent at correct position
} ChannelWave;

void c_reset(ChannelWave* env);
void c_step(ChannelWave* env);
void c_render(ChannelWave* env);
void c_close(ChannelWave* env);
