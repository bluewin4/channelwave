#include "channelwave.h"

// Helper to sample uniform ints
static inline int rand_int(int max_exclusive) {
    return rand() % max_exclusive;
}

static inline int8_t rand_sign() {
    return (rand() & 1) ? 1 : -1;
}

// Log episode completion
static void log_episode(ChannelWave* env, int agent_idx) {
    float ep_len = (float)env->episode_step[agent_idx];
    if (ep_len > 0) {
        env->log.score += env->episode_return[agent_idx];
        env->log.hearts_collected += env->hearts_collected[agent_idx];
        env->log.hearts_missed += env->hearts_missed[agent_idx];
        env->log.sync_accuracy += (float)env->sync_ticks[agent_idx] / ep_len;
        env->log.episode_length += ep_len;
        env->log.n += 1.0f;
    }
}

// Get instance index for an agent
static inline int get_instance(ChannelWave* env, int agent_idx) {
    return agent_idx / env->agents_per_instance;
}

// Get first channel index for an instance
static inline int get_instance_channel_start(ChannelWave* env, int instance) {
    return instance * env->channels_per_instance;
}

// Reset single agent state
static void reset_agent(ChannelWave* env, int agent_idx) {
    int instance = get_instance(env, agent_idx);
    int channel_start = get_instance_channel_start(env, instance);
    
    env->agent_pos[agent_idx] = 0;              // Start at Null (center)
    env->agent_channel[agent_idx] = channel_start;  // Start on first channel of instance
    env->move_cd[agent_idx] = 0;
    env->episode_return[agent_idx] = 0.0f;
    env->episode_step[agent_idx] = 0;
    env->hearts_collected[agent_idx] = 0;
    env->hearts_missed[agent_idx] = 0;
    env->sync_ticks[agent_idx] = 0;
}

// Per-channel spawn counter for deterministic alternation
static int spawn_count[32768] = {0};  // Track spawn count per channel

// Spawn reward on a channel based on spawn mode
static void spawn_reward(ChannelWave* env, int channel) {
    int8_t sign;
    
    if (env->spawn_mode == 0 || env->spawn_mode == 2) {
        // Deterministic alternating per channel
        // Each channel has its own alternation state based on spawn count
        sign = ((spawn_count[channel] % 2) == 0) ? 1 : -1;
        spawn_count[channel]++;
    } else {
        // Random sign (spawn_mode == 1)
        sign = rand_sign();
    }
    
    env->reward_pos[channel] = sign;
    env->reward_timer[channel] = env->channel_produce[channel];
    env->reward_claimed[channel] = 0;
}

// Reset environment state
void c_reset(ChannelWave* env) {
    // Reset all channels
    for (int c = 0; c < env->num_channels; c++) {
        env->reward_pos[c] = 0;
        env->reward_timer[c] = 0;
        env->reward_claimed[c] = 0;
        spawn_count[c] = 0;  // Reset per-channel spawn counter
        // Initialize spawn timers with phase offsets
        env->spawn_timer[c] = env->channel_phase[c];
    }
    
    env->step = 0;
    env->alternation_idx = 0;

    // Reset all agents
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, i);
    }

    // Initial spawn: all channels with phase=0 get a reward
    // This works for both single-channel (mode 0/1) and multi-channel (mode 2)
    for (int c = 0; c < env->num_channels; c++) {
        if (env->spawn_timer[c] == 0) {
            spawn_reward(env, c);
            env->spawn_timer[c] = env->channel_period[c];
        }
    }
}

// Single step
void c_step(ChannelWave* env) {
    const int agents = env->num_agents;
    const int channels = env->num_channels;

    // ===== AGENT MOVEMENT =====
    for (int i = 0; i < agents; i++) {
        // Get instance boundaries for this agent
        int instance = get_instance(env, i);
        int ch_start = get_instance_channel_start(env, instance);
        int ch_end = ch_start + env->channels_per_instance;
        
        // Parse action: [channel_offset, position_idx]
        // channel_offset is relative to instance's first channel
        int channel_offset = env->actions[2*i];
        if (channel_offset < 0) channel_offset = 0;
        if (channel_offset >= env->channels_per_instance) channel_offset = env->channels_per_instance - 1;
        int target_channel = ch_start + channel_offset;

        int pos_idx = env->actions[2*i + 1];
        if (pos_idx < 0) pos_idx = 0;
        if (pos_idx > 2) pos_idx = 2;
        int8_t target_pos = (int8_t)(pos_idx - 1); // map {0,1,2} -> {-1,0,1}

        // Movement with cooldown
        unsigned char moved = (target_pos != env->agent_pos[i]) || 
                              (target_channel != env->agent_channel[i]);
        
        if (env->move_cd[i] == 0 && moved) {
            env->agent_pos[i] = target_pos;
            env->agent_channel[i] = (int16_t)target_channel;
            env->move_cd[i] = env->t_m;
        } else if (env->move_cd[i] > 0) {
            env->move_cd[i]--;
        }

        // Apply movement penalty (step cost)
        env->rewards[i] = -env->move_penalty;
        env->episode_return[i] -= env->move_penalty;
    }

    // ===== REWARD COLLECTION & SYNC TRACKING =====
    for (int i = 0; i < agents; i++) {
        int ch = env->agent_channel[i];
        int8_t pos = env->agent_pos[i];
        int8_t reward_pos = env->reward_pos[ch];
        int16_t timer = env->reward_timer[ch];
        
        unsigned char reward_active = (timer > 0) && (reward_pos != 0);
        unsigned char at_reward = reward_active && (pos == reward_pos);
        unsigned char at_null = (pos == 0);

        // Track synchronization: agent is "synced" if at correct position
        if (at_reward) {
            env->sync_ticks[i]++;
        }

        // Reward collection
        if (env->reward_mode == 0) {
            // Discrete-once: reward only on first collection
            if (at_reward && !env->reward_claimed[ch]) {
                env->rewards[i] += env->heart_reward;
                env->episode_return[i] += env->heart_reward;
                env->reward_claimed[ch] = 1;
                env->hearts_collected[i]++;
            }
        } else {
            // Continuous: reward every tick at correct position
            if (at_reward) {
                env->rewards[i] += env->heart_reward;
                env->episode_return[i] += env->heart_reward;
                if (!env->reward_claimed[ch]) {
                    env->hearts_collected[i]++;
                    env->reward_claimed[ch] = 1;
                }
            }
        }

        // ===== OBSERVATIONS =====
        // [reward_sign_if_visible, reward_timer_if_visible, agent_pos, move_cd, channel_offset, spawn_timer]
        unsigned char visible = at_null;  // Only see reward location from Null
        int16_t obs_sign = visible ? reward_pos : 0;
        int16_t obs_time = visible ? timer : 0;
        int16_t obs_spawn = visible ? env->spawn_timer[ch] : 0;
        
        // Report channel as offset within instance (not global index)
        int instance = get_instance(env, i);
        int ch_start = get_instance_channel_start(env, instance);
        int ch_offset = ch - ch_start;

        int obs_base = i * 6;
        env->observations[obs_base + 0] = (unsigned char)(obs_sign + 128);
        env->observations[obs_base + 1] = (unsigned char)(obs_time > 255 ? 255 : (obs_time < 0 ? 0 : obs_time));
        env->observations[obs_base + 2] = (unsigned char)(pos + 1);
        env->observations[obs_base + 3] = (unsigned char)(env->move_cd[i] > 255 ? 255 : env->move_cd[i]);
        env->observations[obs_base + 4] = (unsigned char)ch_offset;
        env->observations[obs_base + 5] = (unsigned char)(obs_spawn > 255 ? 255 : (obs_spawn < 0 ? 0 : obs_spawn));
    }

    // ===== CHANNEL TIMER UPDATES =====
    for (int c = 0; c < channels; c++) {
        // Decay reward timer
        if (env->reward_timer[c] > 0) {
            env->reward_timer[c]--;
            if (env->reward_timer[c] == 0) {
                // Reward expired - count as missed if unclaimed
                if (!env->reward_claimed[c]) {
                    // Attribute miss to agents on this channel
                    for (int i = 0; i < agents; i++) {
                        if (env->agent_channel[i] == c) {
                            env->hearts_missed[i]++;
                        }
                    }
                }
                env->reward_pos[c] = 0;
                env->reward_claimed[c] = 0;
            }
        }

        // Spawn timer countdown
        if (env->spawn_timer[c] > 0) {
            env->spawn_timer[c]--;
        }
        
        // Spawn new reward when timer hits zero
        // All channels spawn independently (works for instance-based layout)
        if (env->spawn_timer[c] == 0 && env->reward_timer[c] == 0) {
            spawn_reward(env, c);
            env->spawn_timer[c] = env->channel_period[c];
        }
    }

    env->step++;

    // ===== EPISODE TERMINATION =====
    for (int i = 0; i < agents; i++) {
        env->episode_step[i]++;
        
        if (env->max_steps > 0 && env->episode_step[i] >= env->max_steps) {
            env->terminals[i] = 1;
            log_episode(env, i);
            reset_agent(env, i);
        } else {
            env->terminals[i] = 0;
        }
    }
}

void c_render(ChannelWave* env) {
    // No-op render stub
}

void c_close(ChannelWave* env) {
    free(env->channel_period);
    free(env->channel_produce);
    free(env->channel_phase);
    free(env->reward_pos);
    free(env->reward_timer);
    free(env->spawn_timer);
    free(env->reward_claimed);
    free(env->agent_pos);
    free(env->agent_channel);
    free(env->move_cd);
    free(env->episode_return);
    free(env->episode_step);
    free(env->hearts_collected);
    free(env->hearts_missed);
    free(env->sync_ticks);
    if (env->alternation_pattern) {
        free(env->alternation_pattern);
    }
}
