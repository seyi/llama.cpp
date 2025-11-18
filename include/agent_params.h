// agent_params.h
#pragma once

#include "agent_types.h"
#include <cstddef>
#include <cstdint>

enum agent_memory_type {
    AGENT_MEMORY_BUFFER,
    AGENT_MEMORY_VECTOR,
    AGENT_MEMORY_GRAPH
};

struct agent_params {
    // Model configuration
    std::string model_name;
    std::string api_key;
    std::string api_base_url;

    // Context sizing
    size_t max_context_tokens;
    size_t max_completion_tokens;
    size_t max_iterations;

    // Memory configuration
    agent_memory_type memory_type;
    size_t memory_window_size;
    bool enable_memory_persistence;

    // Tool configuration
    std::vector<std::string> enabled_tools;
    bool allow_parallel_tool_calls;
    size_t max_tool_calls_per_step;

    // Generation parameters
    float temperature;
    float top_p;
    int seed;

    // Callbacks (optional)
    progress_callback_fn on_progress;
    tool_callback_fn on_tool_call;
    error_callback_fn on_error;

    // Feature flags
    bool enable_streaming;
    bool enable_logging;
    bool enable_metrics;
    bool enable_caching;

    // Threading
    int32_t n_threads;
};

// Default parameters
inline agent_params agent_default_params() {
    agent_params params;

    // Model defaults
    params.model_name = "gpt-4";
    params.api_key = "";
    params.api_base_url = "https://api.openai.com/v1";

    // Context defaults
    params.max_context_tokens = 8192;
    params.max_completion_tokens = 2048;
    params.max_iterations = 20;

    // Memory defaults
    params.memory_type = AGENT_MEMORY_BUFFER;
    params.memory_window_size = 10;
    params.enable_memory_persistence = false;

    // Tool defaults
    params.enabled_tools = {};
    params.allow_parallel_tool_calls = false;
    params.max_tool_calls_per_step = 1;

    // Generation defaults
    params.temperature = 0.7f;
    params.top_p = 1.0f;
    params.seed = -1;

    // Callbacks (null by default)
    params.on_progress = nullptr;
    params.on_tool_call = nullptr;
    params.on_error = nullptr;

    // Feature flags
    params.enable_streaming = false;
    params.enable_logging = true;
    params.enable_metrics = true;
    params.enable_caching = false;

    // Threading
    params.n_threads = 1;

    return params;
}
