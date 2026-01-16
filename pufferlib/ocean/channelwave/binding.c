#include "channelwave.h"
#include "channelwave.c"

#define Env ChannelWave
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Instance-based layout: replicate (channels_per_instance, agents_per_instance) N times
    env->num_instances = (int)unpack(kwargs, "num_instances");
    env->channels_per_instance = (int)unpack(kwargs, "channels_per_instance");
    env->agents_per_instance = (int)unpack(kwargs, "agents_per_instance");
    
    // Compute totals
    env->num_channels = env->num_instances * env->channels_per_instance;
    env->num_agents = env->num_instances * env->agents_per_instance;
    
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->t_m = (int)unpack(kwargs, "t_m");
    env->move_penalty = (float)unpack(kwargs, "move_penalty");
    env->heart_reward = (float)unpack(kwargs, "heart_reward");
    env->reward_mode = (unsigned char)unpack(kwargs, "reward_mode");
    env->spawn_mode = (unsigned char)unpack(kwargs, "spawn_mode");
    
    int num_channels = env->num_channels;
    int num_agents = env->num_agents;

    // Allocate per-channel timing arrays
    env->channel_period = (int16_t*)calloc(num_channels, sizeof(int16_t));
    env->channel_produce = (int16_t*)calloc(num_channels, sizeof(int16_t));
    env->channel_phase = (int16_t*)calloc(num_channels, sizeof(int16_t));
    env->reward_pos = (int8_t*)calloc(num_channels, sizeof(int8_t));
    env->reward_timer = (int16_t*)calloc(num_channels, sizeof(int16_t));
    env->spawn_timer = (int16_t*)calloc(num_channels, sizeof(int16_t));
    env->reward_claimed = (unsigned char*)calloc(num_channels, sizeof(unsigned char));

    // Allocate per-agent arrays
    env->agent_pos = (int8_t*)calloc(num_agents, sizeof(int8_t));
    env->agent_channel = (int8_t*)calloc(num_agents, sizeof(int8_t));
    env->move_cd = (int16_t*)calloc(num_agents, sizeof(int16_t));
    env->episode_return = (float*)calloc(num_agents, sizeof(float));
    env->episode_step = (int32_t*)calloc(num_agents, sizeof(int32_t));
    env->hearts_collected = (int16_t*)calloc(num_agents, sizeof(int16_t));
    env->hearts_missed = (int16_t*)calloc(num_agents, sizeof(int16_t));
    env->sync_ticks = (int16_t*)calloc(num_agents, sizeof(int16_t));

    // Check allocations
    if (!env->channel_period || !env->channel_produce || !env->channel_phase ||
        !env->reward_pos || !env->reward_timer || !env->spawn_timer || !env->reward_claimed ||
        !env->agent_pos || !env->agent_channel || !env->move_cd ||
        !env->episode_return || !env->episode_step || 
        !env->hearts_collected || !env->hearts_missed || !env->sync_ticks) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate env buffers");
        return 1;
    }

    // Parse per-channel periods (list or single int)
    PyObject* periods = PyDict_GetItemString(kwargs, "channel_periods");
    int default_period = (int)unpack(kwargs, "default_period");
    int default_produce = (int)unpack(kwargs, "default_produce");
    
    if (periods && PyList_Check(periods)) {
        Py_ssize_t n = PyList_Size(periods);
        for (int i = 0; i < num_channels; i++) {
            if (i < n) {
                PyObject* item = PyList_GetItem(periods, i);
                env->channel_period[i] = (int16_t)PyLong_AsLong(item);
            } else {
                env->channel_period[i] = (int16_t)default_period;
            }
        }
    } else {
        for (int i = 0; i < num_channels; i++) {
            env->channel_period[i] = (int16_t)default_period;
        }
    }

    // Parse per-channel produce times (duty cycle)
    PyObject* produces = PyDict_GetItemString(kwargs, "channel_produces");
    if (produces && PyList_Check(produces)) {
        Py_ssize_t n = PyList_Size(produces);
        for (int i = 0; i < num_channels; i++) {
            if (i < n) {
                PyObject* item = PyList_GetItem(produces, i);
                env->channel_produce[i] = (int16_t)PyLong_AsLong(item);
            } else {
                env->channel_produce[i] = (int16_t)default_produce;
            }
        }
    } else {
        for (int i = 0; i < num_channels; i++) {
            env->channel_produce[i] = (int16_t)default_produce;
        }
    }

    // Parse per-channel phase offsets
    PyObject* phases = PyDict_GetItemString(kwargs, "channel_phases");
    if (phases && PyList_Check(phases)) {
        Py_ssize_t n = PyList_Size(phases);
        for (int i = 0; i < num_channels; i++) {
            if (i < n) {
                PyObject* item = PyList_GetItem(phases, i);
                env->channel_phase[i] = (int16_t)PyLong_AsLong(item);
            } else {
                env->channel_phase[i] = 0;
            }
        }
    } else {
        for (int i = 0; i < num_channels; i++) {
            env->channel_phase[i] = 0;
        }
    }

    // Parse alternation pattern
    env->alternation_pattern = NULL;
    env->alternation_len = 0;
    env->alternation_idx = 0;
    
    PyObject* alt_pattern = PyDict_GetItemString(kwargs, "alternation_pattern");
    if (alt_pattern && PyList_Check(alt_pattern)) {
        Py_ssize_t n = PyList_Size(alt_pattern);
        if (n > 0) {
            env->alternation_pattern = (int8_t*)calloc(n, sizeof(int8_t));
            if (!env->alternation_pattern) {
                PyErr_SetString(PyExc_MemoryError, "Failed to allocate alternation_pattern");
                return 1;
            }
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject* item = PyList_GetItem(alt_pattern, i);
                long val = PyLong_AsLong(item);
                env->alternation_pattern[i] = (int8_t)(val >= 0 ? 1 : -1);
            }
            env->alternation_len = (int)n;
        }
    }

    // Initialize log
    memset(&env->log, 0, sizeof(Log));

    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "hearts_collected", log->hearts_collected);
    assign_to_dict(dict, "hearts_missed", log->hearts_missed);
    assign_to_dict(dict, "sync_accuracy", log->sync_accuracy);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
