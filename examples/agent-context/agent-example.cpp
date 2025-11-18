#include "agent_context.h"
#include "agent_executor.h"
#include <iostream>

// Example tool: calculator
tool_result calculator_tool(const std::string & args) {
    // Parse args (simplified)
    // Expected format: "2 + 2"
    tool_result result;
    result.success = true;
    result.output = "4";  // Simplified
    return result;
}

// Example tool: web search
tool_result web_search_tool(const std::string & args) {
    tool_result result;
    result.success = true;
    result.output = "Search results for: " + args;
    return result;
}

// Progress callback
void my_progress_callback(size_t step, const agent_action & action, const std::string & status) {
    std::cout << "[Step " << step << "] " << action.description << " - " << status << std::endl;
}

int main() {
    // 1. Configure agent
    agent_params params = agent_default_params();
    params.model_name = "gpt-4";
    params.max_iterations = 10;
    params.memory_type = AGENT_MEMORY_BUFFER;
    params.memory_window_size = 20;
    params.enable_logging = true;
    params.on_progress = my_progress_callback;

    // 2. Initialize context (RAII - automatic cleanup on scope exit)
    agent_context * ctx = agent_init(params);
    if (!ctx) {
        std::cerr << "Failed to initialize agent context" << std::endl;
        return 1;
    }

    // 3. Register tools
    auto * tool_exec = static_cast<function_tool_executor*>(ctx->tools.get());
    tool_exec->register_tool("calculator", calculator_tool);
    tool_exec->register_tool("web_search", web_search_tool);

    // 4. Create and execute task
    agent_task task;
    task.instruction = "Search for the latest AI news and calculate 2+2";
    task.max_steps = 5;

    agent_result result = agent_execute(ctx, task);

    // 5. Display results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Success: " << (result.success ? "Yes" : "No") << std::endl;
    std::cout << "Output: " << result.output << std::endl;
    std::cout << "Steps taken: " << result.steps.size() << std::endl;
    std::cout << "Tool calls: " << result.tool_calls_count << std::endl;
    std::cout << "Execution time: " << result.execution_time_us / 1000.0 << " ms" << std::endl;

    // 6. Save state for later
    if (agent_save_state(ctx, "session.state")) {
        std::cout << "State saved successfully" << std::endl;
    }

    // 7. Cleanup (automatic via RAII, but explicit call shown here)
    agent_free(ctx);

    return 0;
}
