// agent_types.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <any>
#include <cstdint>

// Forward declarations
struct agent_context;
class agent_memory_i;
class tool_executor_i;

// Message types
enum message_role {
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL
};

struct message {
    message_role role;
    std::string content;
    std::map<std::string, std::any> metadata;
    int64_t timestamp_us;
};

// Action types
enum action_type {
    ACTION_REASONING,
    ACTION_TOOL_CALL,
    ACTION_FINAL_ANSWER,
    ACTION_WAIT
};

struct agent_action {
    action_type type;
    std::string description;
    std::string tool_name;  // For ACTION_TOOL_CALL
    std::string arguments;  // JSON string
    bool is_final;
};

struct agent_result {
    std::string output;
    std::vector<agent_action> steps;
    size_t tool_calls_count;
    int64_t execution_time_us;
    bool success;
};

// Task definition
struct agent_task {
    std::string instruction;
    std::map<std::string, std::any> context;
    size_t max_steps;
};

// Callbacks
using progress_callback_fn = std::function<void(size_t, const agent_action&, const std::string&)>;
using tool_callback_fn = std::function<void(const std::string&, const std::string&)>;
using error_callback_fn = std::function<void(const std::string&)>;
