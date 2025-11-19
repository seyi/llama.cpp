#include "agent_context.h"
#include "agent_executor.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <cmath>

// Test utilities
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        std::cerr << "FAILED: " #expr << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))

// Test fixtures
static size_t g_progress_call_count = 0;
static size_t g_tool_call_count = 0;
static size_t g_error_call_count = 0;
static std::string g_last_tool_name;
static std::string g_last_error;

void reset_test_counters() {
    g_progress_call_count = 0;
    g_tool_call_count = 0;
    g_error_call_count = 0;
    g_last_tool_name.clear();
    g_last_error.clear();
}

void test_progress_callback(size_t step, const agent_action & action, const std::string & status) {
    g_progress_call_count++;
}

void test_tool_callback(const std::string & tool_name, const std::string & args) {
    g_tool_call_count++;
    g_last_tool_name = tool_name;
}

void test_error_callback(const std::string & error) {
    g_error_call_count++;
    g_last_error = error;
}

// Test tools
tool_result add_tool(const std::string & args) {
    tool_result result;
    result.success = true;
    result.output = "42";  // Simplified
    return result;
}

tool_result multiply_tool(const std::string & args) {
    tool_result result;
    result.success = true;
    result.output = "100";  // Simplified
    return result;
}

tool_result failing_tool(const std::string & args) {
    tool_result result;
    result.success = false;
    result.error = "Tool intentionally failed";
    result.output = "";
    return result;
}

// =============================================================================
// Test Cases
// =============================================================================

TEST(default_params) {
    agent_params params = agent_default_params();

    ASSERT_EQ(params.model_name, "gpt-4");
    ASSERT_EQ(params.max_context_tokens, 8192);
    ASSERT_EQ(params.max_completion_tokens, 2048);
    ASSERT_EQ(params.max_iterations, 20);
    ASSERT_EQ(params.memory_type, AGENT_MEMORY_BUFFER);
    ASSERT_EQ(params.memory_window_size, 10);
    ASSERT_EQ(params.temperature, 0.7f);
    ASSERT_EQ(params.n_threads, 1);
    ASSERT_TRUE(params.enable_logging);
    ASSERT_TRUE(params.enable_metrics);
}

TEST(context_initialization) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    ASSERT_TRUE(ctx != nullptr);
    ASSERT_TRUE(ctx->memory != nullptr);
    ASSERT_TRUE(ctx->tools != nullptr);
    ASSERT_EQ(ctx->metrics.total_iterations, 0);
    ASSERT_EQ(ctx->metrics.total_tool_calls, 0);
    ASSERT_EQ(ctx->metrics.total_tokens_used, 0);

    agent_free(ctx);
}

TEST(memory_store_and_retrieve) {
    agent_params params = agent_default_params();
    params.memory_window_size = 5;
    agent_context * ctx = agent_init(params);

    // Store messages
    message msg1;
    msg1.role = ROLE_USER;
    msg1.content = "Hello";
    msg1.timestamp_us = 1000;

    message msg2;
    msg2.role = ROLE_ASSISTANT;
    msg2.content = "Hi there!";
    msg2.timestamp_us = 2000;

    ctx->memory->store(msg1);
    ctx->memory->store(msg2);

    // Retrieve all
    auto all_msgs = ctx->memory->retrieve_all();
    ASSERT_EQ(all_msgs.size(), 2);
    ASSERT_EQ(all_msgs[0].content, "Hello");
    ASSERT_EQ(all_msgs[1].content, "Hi there!");

    // Retrieve recent
    auto recent = ctx->memory->retrieve_recent(1);
    ASSERT_EQ(recent.size(), 1);
    ASSERT_EQ(recent[0].content, "Hi there!");

    agent_free(ctx);
}

TEST(memory_buffer_overflow) {
    agent_params params = agent_default_params();
    params.memory_window_size = 3;
    agent_context * ctx = agent_init(params);

    // Add more messages than buffer size
    for (int i = 0; i < 5; i++) {
        message msg;
        msg.role = ROLE_USER;
        msg.content = "Message " + std::to_string(i);
        ctx->memory->store(msg);
    }

    // Should only have the last 3 messages
    auto all_msgs = ctx->memory->retrieve_all();
    ASSERT_EQ(all_msgs.size(), 3);
    ASSERT_EQ(all_msgs[0].content, "Message 2");
    ASSERT_EQ(all_msgs[1].content, "Message 3");
    ASSERT_EQ(all_msgs[2].content, "Message 4");

    agent_free(ctx);
}

TEST(memory_clear) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    message msg;
    msg.role = ROLE_USER;
    msg.content = "Test";
    ctx->memory->store(msg);

    ASSERT_EQ(ctx->memory->size(), 1);

    ctx->memory->clear();
    ASSERT_EQ(ctx->memory->size(), 0);

    agent_free(ctx);
}

TEST(tool_registration) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    auto * tools = static_cast<function_tool_executor*>(ctx->tools.get());

    ASSERT_FALSE(tools->has_tool("add"));

    tools->register_tool("add", add_tool);
    ASSERT_TRUE(tools->has_tool("add"));

    tools->register_tool("multiply", multiply_tool);

    auto tool_list = tools->list_tools();
    ASSERT_EQ(tool_list.size(), 2);

    agent_free(ctx);
}

TEST(tool_execution) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    auto * tools = static_cast<function_tool_executor*>(ctx->tools.get());
    tools->register_tool("add", add_tool);

    tool_result result = ctx->tools->execute("add", "2+2");

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.output, "42");
    ASSERT_TRUE(result.execution_time_us >= 0);

    agent_free(ctx);
}

TEST(tool_not_found) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    tool_result result = ctx->tools->execute("nonexistent", "");

    ASSERT_FALSE(result.success);
    ASSERT_TRUE(result.error.find("Tool not found") != std::string::npos);

    agent_free(ctx);
}

TEST(tool_failure) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    auto * tools = static_cast<function_tool_executor*>(ctx->tools.get());
    tools->register_tool("fail", failing_tool);

    tool_result result = ctx->tools->execute("fail", "");

    ASSERT_FALSE(result.success);
    ASSERT_EQ(result.error, "Tool intentionally failed");

    agent_free(ctx);
}

TEST(callbacks) {
    reset_test_counters();

    agent_params params = agent_default_params();
    params.on_progress = test_progress_callback;
    params.on_tool_call = test_tool_callback;
    params.on_error = test_error_callback;

    agent_context * ctx = agent_init(params);

    auto * tools = static_cast<function_tool_executor*>(ctx->tools.get());
    tools->register_tool("add", add_tool);

    agent_task task;
    task.instruction = "Calculate something";
    task.max_steps = 1;

    agent_result result = agent_execute(ctx, task);

    // Progress callback should be called at least once
    ASSERT_TRUE(g_progress_call_count >= 1);

    agent_free(ctx);
}

TEST(state_persistence) {
    const std::string state_file = "test_state.bin";

    // Create and save state
    {
        agent_params params = agent_default_params();
        agent_context * ctx = agent_init(params);

        // Add some messages
        message msg1;
        msg1.role = ROLE_USER;
        msg1.content = "Persist me";
        msg1.timestamp_us = 12345;
        ctx->memory->store(msg1);

        message msg2;
        msg2.role = ROLE_ASSISTANT;
        msg2.content = "Persisted!";
        msg2.timestamp_us = 67890;
        ctx->memory->store(msg2);

        // Set some metrics
        ctx->metrics.total_iterations = 42;
        ctx->metrics.total_tool_calls = 10;

        // Save state
        bool saved = agent_save_state(ctx, state_file);
        ASSERT_TRUE(saved);

        agent_free(ctx);
    }

    // Load state and verify
    {
        agent_params params = agent_default_params();
        agent_context * ctx = agent_init(params);

        bool loaded = agent_load_state(ctx, state_file);
        ASSERT_TRUE(loaded);

        // Check messages
        auto msgs = ctx->memory->retrieve_all();
        ASSERT_EQ(msgs.size(), 2);
        ASSERT_EQ(msgs[0].content, "Persist me");
        ASSERT_EQ(msgs[0].timestamp_us, 12345);
        ASSERT_EQ(msgs[1].content, "Persisted!");
        ASSERT_EQ(msgs[1].timestamp_us, 67890);

        // Check metrics
        ASSERT_EQ(ctx->metrics.total_iterations, 42);
        ASSERT_EQ(ctx->metrics.total_tool_calls, 10);

        agent_free(ctx);
    }

    // Clean up
    std::remove(state_file.c_str());
}

TEST(task_execution) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    auto * tools = static_cast<function_tool_executor*>(ctx->tools.get());
    tools->register_tool("add", add_tool);

    agent_task task;
    task.instruction = "Calculate 2+2";
    task.max_steps = 5;

    agent_result result = agent_execute(ctx, task);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.execution_time_us > 0);
    ASSERT_TRUE(ctx->metrics.total_iterations >= 1);

    // Memory should contain the task
    auto msgs = ctx->memory->retrieve_all();
    ASSERT_TRUE(msgs.size() >= 1);
    ASSERT_EQ(msgs[0].content, "Calculate 2+2");
    ASSERT_EQ(msgs[0].role, ROLE_USER);

    agent_free(ctx);
}

TEST(max_iterations_limit) {
    agent_params params = agent_default_params();
    params.max_iterations = 3;
    agent_context * ctx = agent_init(params);

    agent_task task;
    task.instruction = "Long running task";
    task.max_steps = 0;  // Use default from params

    // Note: Since get_next_action returns final answer immediately,
    // we can't fully test iteration limiting. But we verify max_steps is respected.
    agent_result result = agent_execute(ctx, task);

    ASSERT_TRUE(ctx->metrics.total_iterations <= params.max_iterations);

    agent_free(ctx);
}

TEST(metrics_tracking) {
    agent_params params = agent_default_params();
    params.enable_metrics = true;
    agent_context * ctx = agent_init(params);

    // Initial state
    ASSERT_EQ(ctx->metrics.total_iterations, 0);
    ASSERT_EQ(ctx->metrics.total_tool_calls, 0);
    ASSERT_EQ(ctx->metrics.total_time_us, 0);

    agent_task task;
    task.instruction = "Test";
    task.max_steps = 1;

    agent_result result = agent_execute(ctx, task);

    // Metrics should be updated
    ASSERT_TRUE(ctx->metrics.total_iterations >= 1);
    ASSERT_TRUE(ctx->metrics.total_time_us >= 0);  // May be 0 on very fast systems
    ASSERT_TRUE(result.execution_time_us >= 0);

    agent_free(ctx);
}

TEST(null_context_handling) {
    agent_task task;
    task.instruction = "Test";

    agent_result result = agent_execute(nullptr, task);

    ASSERT_FALSE(result.success);
    ASSERT_EQ(result.output, "");
    ASSERT_EQ(result.steps.size(), 0);

    // State operations with null
    ASSERT_FALSE(agent_save_state(nullptr, "test.bin"));
    ASSERT_FALSE(agent_load_state(nullptr, "test.bin"));
}

TEST(context_timing) {
    agent_params params = agent_default_params();
    agent_context * ctx = agent_init(params);

    int64_t start = ctx->t_start_us;
    ASSERT_TRUE(start > 0);

    // Wait a bit
    for (volatile int i = 0; i < 100000; i++);

    int64_t now = agent_context::get_time_us();
    ASSERT_TRUE(now > start);

    agent_free(ctx);
}

TEST(memory_is_full) {
    agent_params params = agent_default_params();
    params.memory_window_size = 2;
    agent_context * ctx = agent_init(params);

    ASSERT_FALSE(ctx->memory->is_full());

    message msg;
    msg.role = ROLE_USER;
    msg.content = "Test";

    ctx->memory->store(msg);
    ASSERT_FALSE(ctx->memory->is_full());

    ctx->memory->store(msg);
    ASSERT_TRUE(ctx->memory->is_full());

    agent_free(ctx);
}

TEST(custom_memory_window_size) {
    agent_params params = agent_default_params();
    params.memory_window_size = 100;

    agent_context * ctx = agent_init(params);

    // Add many messages
    for (int i = 0; i < 50; i++) {
        message msg;
        msg.role = ROLE_USER;
        msg.content = "Msg " + std::to_string(i);
        ctx->memory->store(msg);
    }

    ASSERT_EQ(ctx->memory->size(), 50);
    ASSERT_FALSE(ctx->memory->is_full());

    agent_free(ctx);
}

// =============================================================================
// Main test runner
// =============================================================================

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Agent Context Management Test Suite" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;

    RUN_TEST(default_params);
    RUN_TEST(context_initialization);
    RUN_TEST(memory_store_and_retrieve);
    RUN_TEST(memory_buffer_overflow);
    RUN_TEST(memory_clear);
    RUN_TEST(tool_registration);
    RUN_TEST(tool_execution);
    RUN_TEST(tool_not_found);
    RUN_TEST(tool_failure);
    RUN_TEST(callbacks);
    RUN_TEST(state_persistence);
    RUN_TEST(task_execution);
    RUN_TEST(max_iterations_limit);
    RUN_TEST(metrics_tracking);
    RUN_TEST(null_context_handling);
    RUN_TEST(context_timing);
    RUN_TEST(memory_is_full);
    RUN_TEST(custom_memory_window_size);

    std::cout << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "All tests PASSED!" << std::endl;
    std::cout << "====================================" << std::endl;

    return 0;
}
