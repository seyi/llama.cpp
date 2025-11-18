// agent_executor.h
#pragma once

#include "agent_context.h"
#include <sstream>
#include <fstream>

// Forward declaration
inline agent_action get_next_action(agent_context * ctx);

// Execute a task with the agent
inline agent_result agent_execute(agent_context * ctx, const agent_task & task) {
    if (!ctx) {
        return agent_result{"", {}, 0, 0, false};
    }

    auto start_time = agent_context::get_time_us();
    agent_result result;
    result.success = false;

    // Add task to memory as a user message
    message user_msg;
    user_msg.role = ROLE_USER;
    user_msg.content = task.instruction;
    user_msg.timestamp_us = start_time;
    ctx->memory->store(user_msg);

    // Execute task in steps
    size_t max_steps = task.max_steps > 0 ? task.max_steps : ctx->params.max_iterations;

    for (size_t step = 0; step < max_steps; ++step) {
        ctx->metrics.total_iterations++;

        // Get next action from agent
        // This would call your LLM here
        agent_action action = get_next_action(ctx);

        if (ctx->params.on_progress) {
            ctx->params.on_progress(step, action, "Processing...");
        }

        if (action.is_final) {
            result.output = action.description;
            result.success = true;
            break;
        }

        // Execute action
        if (action.type == ACTION_TOOL_CALL) {
            ctx->metrics.total_tool_calls++;

            if (ctx->params.on_tool_call) {
                ctx->params.on_tool_call(action.tool_name, action.arguments);
            }

            tool_result tool_res = ctx->tools->execute(action.tool_name, action.arguments);

            // Store tool result in memory
            message tool_msg;
            tool_msg.role = ROLE_TOOL;
            tool_msg.content = tool_res.output;
            tool_msg.metadata["tool_name"] = action.tool_name;
            tool_msg.timestamp_us = agent_context::get_time_us();
            ctx->memory->store(tool_msg);

            if (!tool_res.success && ctx->params.on_error) {
                ctx->params.on_error("Tool execution failed: " + tool_res.error);
            }
        }

        result.steps.push_back(action);
    }

    result.tool_calls_count = ctx->metrics.total_tool_calls;
    result.execution_time_us = agent_context::get_time_us() - start_time;
    ctx->metrics.total_time_us += result.execution_time_us;

    return result;
}

// Placeholder for LLM call (implement with your LLM provider)
inline agent_action get_next_action(agent_context * ctx) {
    // This is where you would:
    // 1. Get conversation history from memory
    // 2. Format as prompt for LLM
    // 3. Call LLM API
    // 4. Parse response to extract action

    // Simplified example:
    auto history = ctx->memory->retrieve_all();

    // TODO: Call LLM with history
    // std::string llm_response = call_llm(ctx->params.model_name, history);

    // TODO: Parse LLM response to action
    // return parse_action(llm_response);

    // Dummy action for compilation
    agent_action action;
    action.type = ACTION_FINAL_ANSWER;
    action.description = "Task complete";
    action.is_final = true;
    return action;
}

// Save context state
inline bool agent_save_state(agent_context * ctx, const std::string & filepath) {
    if (!ctx) return false;

    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    // Save memory state
    ctx->memory->save_state(file);

    // Save metrics
    file.write(reinterpret_cast<const char*>(&ctx->metrics), sizeof(ctx->metrics));

    return true;
}

// Load context state
inline bool agent_load_state(agent_context * ctx, const std::string & filepath) {
    if (!ctx) return false;

    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    // Load memory state
    ctx->memory->load_state(file);

    // Load metrics
    file.read(reinterpret_cast<char*>(&ctx->metrics), sizeof(ctx->metrics));

    return true;
}
