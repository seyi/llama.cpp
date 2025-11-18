// agent_context.h
#pragma once

#include "agent_types.h"
#include "agent_params.h"
#include "agent_memory.h"
#include "agent_tools.h"
#include <chrono>
#include <exception>

struct agent_metrics {
    size_t total_iterations;
    size_t total_tool_calls;
    size_t total_tokens_used;
    int64_t total_time_us;

    void reset() {
        total_iterations = 0;
        total_tool_calls = 0;
        total_tokens_used = 0;
        total_time_us = 0;
    }
};

struct agent_context {
    // Constructor (RAII pattern)
    agent_context(const agent_params & params_in)
        : params(params_in)
        , t_start_us(get_time_us()) {

        // Initialize memory based on type
        switch (params.memory_type) {
            case AGENT_MEMORY_BUFFER:
                memory = std::make_unique<buffer_memory>(params.memory_window_size);
                break;
            // Add other memory types here
            default:
                memory = std::make_unique<buffer_memory>(params.memory_window_size);
        }

        // Initialize tool executor
        tools = std::make_unique<function_tool_executor>();

        // Reset metrics
        metrics.reset();
    }

    // Destructor (automatic cleanup)
    ~agent_context() {
        if (params.enable_logging) {
            // Log final metrics
            // fprintf(stderr, "Agent context destroyed. Total time: %lld us\n", get_time_us() - t_start_us);
        }
    }

    // Prevent copying (follow llama.cpp pattern)
    agent_context(const agent_context &) = delete;
    agent_context & operator=(const agent_context &) = delete;

    // Get current time in microseconds
    static int64_t get_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    // Public members (following llama.cpp style)
    agent_params params;
    agent_memory_ptr memory;
    tool_executor_ptr tools;
    agent_metrics metrics;

    int64_t t_start_us;
    int64_t t_last_exec_us;
};

// Factory function (similar to llama_init_from_model)
inline agent_context * agent_init(const agent_params & params) {
    try {
        return new agent_context(params);
    } catch (const std::exception & err) {
        if (params.on_error) {
            params.on_error(std::string("Failed to initialize agent: ") + err.what());
        }
        return nullptr;
    }
}

// Cleanup function (similar to llama_free)
inline void agent_free(agent_context * ctx) {
    delete ctx;
}
