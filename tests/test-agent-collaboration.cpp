// Test suite for agent collaboration framework

#include "../common/agent/agent.h"
#include "../common/agent/registry.h"
#include "../common/agent/conversation.h"
#include "../common/agent/message.h"
#include "../common/agent/failure.h"

#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <atomic>

using namespace agent;

// Test helpers
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    std::cout << "Running " << #test_func << "..." << std::endl; \
    if (test_func()) { \
        std::cout << "  PASS" << std::endl; \
        passed++; \
    } else { \
        std::cout << "  FAIL" << std::endl; \
        failed++; \
    } \
    total++; \
} while(0)

// Mock inference function
std::string mock_inference(const std::string& prompt, const std::map<std::string, std::string>& params) {
    return "Mock response to: " + prompt.substr(0, std::min<size_t>(20, prompt.length()));
}

// Forward declarations
static bool test_uuid_generation();
static bool test_timestamp_generation();
static bool test_message_serialization();
static bool test_response_serialization();
static bool test_message_queue();
static bool test_conversation_memory();
static bool test_conversation_turns();
static bool test_conversation_history();
static bool test_context_reconstruction();
static bool test_thread_expiration();
static bool test_agent_creation();
static bool test_agent_registration();
static bool test_agent_discovery();
static bool test_agent_request_processing();
static bool test_multiturn_conversation();
static bool test_failure_policy();
static bool test_circuit_breaker();
static bool test_token_estimation();
static bool test_agent_statistics();
static bool test_registry_statistics();
static bool test_thread_branching();
static bool test_thread_cleanup();
static bool test_concurrent_message_queue();
static bool test_error_type_conversion();
static bool test_agent_status();

// Test 1: UUID Generation
static bool test_uuid_generation() {
    std::string uuid1 = generate_uuid();
    std::string uuid2 = generate_uuid();

    TEST_ASSERT(uuid1.length() == 36, "UUID should be 36 characters");
    TEST_ASSERT(uuid2.length() == 36, "UUID should be 36 characters");
    TEST_ASSERT(uuid1 != uuid2, "UUIDs should be unique");
    TEST_ASSERT(uuid1.find('-') != std::string::npos, "UUID should contain hyphens");

    return true;
}

// Test 2: Timestamp Generation
static bool test_timestamp_generation() {
    int64_t ts1 = get_timestamp_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    int64_t ts2 = get_timestamp_ms();

    TEST_ASSERT(ts1 > 0, "Timestamp should be positive");
    TEST_ASSERT(ts2 > ts1, "Timestamps should increase over time");
    TEST_ASSERT(ts2 - ts1 >= 10, "Time difference should be at least 10ms");

    return true;
}

// Test 3: Message Serialization
static bool test_message_serialization() {
    agent_request req;
    req.prompt = "Test prompt";
    req.max_tokens = 100;
    req.temperature = 0.7f;
    req.files = {"file1.txt", "file2.txt"};
    req.params["key1"] = "value1";

    std::string json_str = req.to_json();
    TEST_ASSERT(!json_str.empty(), "JSON should not be empty");

    agent_request req2 = agent_request::from_json(json_str);
    TEST_ASSERT(req2.prompt == req.prompt, "Prompt should match");
    TEST_ASSERT(req2.max_tokens == req.max_tokens, "max_tokens should match");
    TEST_ASSERT(req2.temperature == req.temperature, "temperature should match");
    TEST_ASSERT(req2.files.size() == 2, "Should have 2 files");
    TEST_ASSERT(req2.params["key1"] == "value1", "Params should match");

    return true;
}

// Test 4: Response Serialization
static bool test_response_serialization() {
    agent_response resp;
    resp.status = RESPONSE_STATUS_SUCCESS;
    resp.content = "Test response";
    resp.thread_id = "test-thread-123";
    resp.tokens_used = 50;
    resp.metadata["key1"] = "value1";

    std::string json_str = resp.to_json();
    TEST_ASSERT(!json_str.empty(), "JSON should not be empty");

    agent_response resp2 = agent_response::from_json(json_str);
    TEST_ASSERT(resp2.status == resp.status, "Status should match");
    TEST_ASSERT(resp2.content == resp.content, "Content should match");
    TEST_ASSERT(resp2.thread_id == resp.thread_id, "Thread ID should match");
    TEST_ASSERT(resp2.tokens_used == resp.tokens_used, "Tokens should match");
    TEST_ASSERT(resp2.metadata["key1"] == "value1", "Metadata should match");

    return true;
}

// Test 5: Message Queue Operations
static bool test_message_queue() {
    message_queue queue(10);

    TEST_ASSERT(queue.empty(), "Queue should start empty");
    TEST_ASSERT(queue.size() == 0, "Queue size should be 0");

    agent_message msg1;
    msg1.message_id = "msg1";
    msg1.from_agent = "agent1";
    msg1.to_agent = "agent2";
    msg1.type = MESSAGE_TYPE_REQUEST;
    msg1.payload = "test payload";
    msg1.priority = 5;

    TEST_ASSERT(queue.push(msg1), "Should be able to push message");
    TEST_ASSERT(queue.size() == 1, "Queue size should be 1");
    TEST_ASSERT(!queue.empty(), "Queue should not be empty");

    agent_message msg2;
    TEST_ASSERT(queue.pop(msg2, 0), "Should be able to pop message");
    TEST_ASSERT(msg2.message_id == msg1.message_id, "Message IDs should match");
    TEST_ASSERT(queue.empty(), "Queue should be empty after pop");

    return true;
}

// Test 6: Conversation Memory Creation
static bool test_conversation_memory() {
    conversation_memory memory(1);  // 1 hour TTL

    TEST_ASSERT(memory.thread_count() == 0, "Should start with no threads");

    agent_request req;
    req.prompt = "Test prompt";

    std::string thread_id = memory.create_thread("agent1", req);
    TEST_ASSERT(!thread_id.empty(), "Thread ID should not be empty");
    TEST_ASSERT(memory.thread_count() == 1, "Should have 1 thread");
    TEST_ASSERT(memory.has_thread(thread_id), "Thread should exist");

    return true;
}

// Test 7: Conversation Turns
static bool test_conversation_turns() {
    conversation_memory memory(1);

    agent_request req;
    req.prompt = "Initial prompt";

    std::string thread_id = memory.create_thread("agent1", req);

    TEST_ASSERT(memory.add_turn(thread_id, "user", "Hello", {}, {}, "agent1"),
                "Should add user turn");
    TEST_ASSERT(memory.add_turn(thread_id, "assistant", "Hi there!", {}, {}, "agent1"),
                "Should add assistant turn");

    auto thread_opt = memory.get_thread(thread_id);
    TEST_ASSERT(thread_opt.has_value(), "Thread should exist");

    auto& thread = thread_opt.value();
    TEST_ASSERT(thread.turns.size() == 2, "Should have 2 turns");
    TEST_ASSERT(thread.turns[0].role == "user", "First turn should be user");
    TEST_ASSERT(thread.turns[0].content == "Hello", "First turn content should match");
    TEST_ASSERT(thread.turns[1].role == "assistant", "Second turn should be assistant");

    return true;
}

// Test 8: Conversation History Building
static bool test_conversation_history() {
    conversation_memory memory(1);

    agent_request req;
    req.prompt = "Test";

    std::string thread_id = memory.create_thread("agent1", req);
    memory.add_turn(thread_id, "user", "Question 1", {}, {}, "agent1");
    memory.add_turn(thread_id, "assistant", "Answer 1", {}, {}, "agent1");
    memory.add_turn(thread_id, "user", "Question 2", {}, {}, "agent1");

    auto context = memory.build_conversation_history(thread_id, 0, false);

    TEST_ASSERT(!context.full_context.empty(), "Context should not be empty");
    TEST_ASSERT(context.turns_included == 3, "Should include 3 turns");
    TEST_ASSERT(context.tokens_used > 0, "Should have token count");
    TEST_ASSERT(!context.truncated, "Should not be truncated with no limit");

    return true;
}

// Test 9: Context Reconstruction
static bool test_context_reconstruction() {
    conversation_memory memory(1);

    agent_request req;
    req.prompt = "Initial";

    std::string thread_id = memory.create_thread("agent1", req);
    memory.add_turn(thread_id, "user", "Question", {}, {}, "agent1");
    memory.add_turn(thread_id, "assistant", "Answer", {}, {}, "agent1");

    agent_request continuation;
    continuation.prompt = "Follow-up";
    continuation.thread_id = thread_id;

    agent_request reconstructed = memory.reconstruct_request(continuation);

    TEST_ASSERT(!reconstructed.prompt.empty(), "Prompt should not be empty");
    // Context reconstruction includes the full conversation history
    // Check for either "Question" or "Conversation History" marker
    bool has_context = (reconstructed.prompt.find("Question") != std::string::npos) ||
                       (reconstructed.prompt.find("Conversation") != std::string::npos) ||
                       (reconstructed.prompt.length() > continuation.prompt.length());
    TEST_ASSERT(has_context, "Should contain previous conversation or history marker");
    TEST_ASSERT(reconstructed.prompt.find("Follow-up") != std::string::npos,
                "Should contain new prompt");

    return true;
}

// Test 10: Thread Expiration
static bool test_thread_expiration() {
    conversation_memory memory(1);  // 1 hour TTL

    agent_request req;
    req.prompt = "Test";

    std::string thread_id = memory.create_thread("agent1", req);
    TEST_ASSERT(memory.has_thread(thread_id), "Thread should exist");

    // Touch thread to keep it alive
    TEST_ASSERT(memory.touch_thread(thread_id), "Should be able to touch thread");
    TEST_ASSERT(memory.has_thread(thread_id), "Thread should still exist");

    return true;
}

// Test 11: Agent Creation
static bool test_agent_creation() {
    conversation_memory memory(1);

    auto agent = agent_factory::create_local_agent(
        "Test Agent",
        "A test agent",
        {"testing", "validation"},
        &memory
    );

    TEST_ASSERT(agent != nullptr, "Agent should not be null");

    auto info = agent->get_info();
    TEST_ASSERT(info.name == "Test Agent", "Name should match");
    TEST_ASSERT(info.description == "A test agent", "Description should match");
    TEST_ASSERT(info.capabilities.size() == 2, "Should have 2 capabilities");
    TEST_ASSERT(info.has_capability("testing"), "Should have testing capability");
    TEST_ASSERT(!info.has_capability("nonexistent"), "Should not have nonexistent capability");

    return true;
}

// Test 12: Agent Registration
static bool test_agent_registration() {
    auto& registry = agent_registry::instance();
    conversation_memory memory(1);

    auto agent = agent_factory::create_local_agent(
        "Test Agent",
        "A test agent",
        {"testing"},
        &memory
    );

    std::string agent_id = agent->get_info().id;
    TEST_ASSERT(registry.register_agent(std::move(agent)), "Should register agent");

    auto agent_ptr = registry.get_agent(agent_id);
    TEST_ASSERT(agent_ptr != nullptr, "Should retrieve agent");

    auto agents = registry.list_agents();
    bool found = false;
    for (const auto& info : agents) {
        if (info.id == agent_id) {
            found = true;
            break;
        }
    }
    TEST_ASSERT(found, "Agent should be in list");

    // Cleanup
    registry.unregister_agent(agent_id);

    return true;
}

// Test 13: Agent Discovery
static bool test_agent_discovery() {
    auto& registry = agent_registry::instance();
    conversation_memory memory(1);

    auto agent1 = agent_factory::create_local_agent(
        "Code Agent",
        "Code analysis",
        {"code", "analysis"},
        &memory
    );
    std::string id1 = agent1->get_info().id;

    auto agent2 = agent_factory::create_local_agent(
        "Test Agent",
        "Test generation",
        {"testing", "qa"},
        &memory
    );
    std::string id2 = agent2->get_info().id;

    registry.register_agent(std::move(agent1));
    registry.register_agent(std::move(agent2));

    // Find by capability
    agent_query query;
    query.capabilities = {"testing"};

    auto found = registry.find_agents(query);
    TEST_ASSERT(found.size() == 1, "Should find 1 agent with testing capability");
    TEST_ASSERT(found[0].name == "Test Agent", "Should find the test agent");

    // Find by multiple capabilities (AND)
    query.capabilities = {"code", "analysis"};
    query.require_all_capabilities = true;
    found = registry.find_agents(query);
    TEST_ASSERT(found.size() == 1, "Should find 1 agent with both capabilities");

    // Cleanup
    registry.unregister_agent(id1);
    registry.unregister_agent(id2);

    return true;
}

// Test 14: Agent Request Processing
static bool test_agent_request_processing() {
    auto& registry = agent_registry::instance();
    conversation_memory memory(1);

    auto agent = agent_factory::create_local_agent(
        "Test Agent",
        "Test",
        {"testing"},
        &memory
    );

    auto agent_ptr = static_cast<local_agent*>(agent.get());
    agent_ptr->set_inference_callback(mock_inference);

    std::string agent_id = agent->get_info().id;
    registry.register_agent(std::move(agent));
    registry.set_conversation_memory(&memory);

    agent_request req;
    req.prompt = "Test prompt for processing";
    req.max_tokens = 100;

    auto response = registry.send_request(agent_id, req);

    TEST_ASSERT(response.status == RESPONSE_STATUS_SUCCESS, "Response should be successful");
    TEST_ASSERT(!response.content.empty(), "Response should have content");
    TEST_ASSERT(!response.thread_id.empty(), "Response should have thread ID");
    TEST_ASSERT(response.tokens_used > 0, "Response should have token count");

    // Cleanup
    registry.unregister_agent(agent_id);

    return true;
}

// Test 15: Multi-turn Conversation
static bool test_multiturn_conversation() {
    auto& registry = agent_registry::instance();
    conversation_memory memory(1);

    auto agent = agent_factory::create_local_agent(
        "Chat Agent",
        "Chat",
        {"chat"},
        &memory
    );

    auto agent_ptr = static_cast<local_agent*>(agent.get());
    agent_ptr->set_inference_callback(mock_inference);

    std::string agent_id = agent->get_info().id;
    registry.register_agent(std::move(agent));
    registry.set_conversation_memory(&memory);

    // First request
    agent_request req1;
    req1.prompt = "Question 1";
    auto resp1 = registry.send_request(agent_id, req1);

    TEST_ASSERT(resp1.status == RESPONSE_STATUS_SUCCESS, "First response should succeed");
    TEST_ASSERT(!resp1.thread_id.empty(), "First response should have thread ID");

    // Second request (continuation)
    agent_request req2;
    req2.prompt = "Question 2";
    req2.thread_id = resp1.thread_id;
    auto resp2 = registry.send_request(agent_id, req2);

    TEST_ASSERT(resp2.status == RESPONSE_STATUS_SUCCESS, "Second response should succeed");
    TEST_ASSERT(resp2.thread_id == resp1.thread_id, "Thread IDs should match");

    // Check conversation history
    auto thread_opt = memory.get_thread(resp1.thread_id);
    TEST_ASSERT(thread_opt.has_value(), "Thread should exist");
    TEST_ASSERT(thread_opt->turns.size() >= 4, "Should have at least 4 turns (2 requests + 2 responses)");

    // Cleanup
    registry.unregister_agent(agent_id);

    return true;
}

// Test 16: Failure Policy
static bool test_failure_policy() {
    auto default_policy = failure_policy::default_policy();
    TEST_ASSERT(default_policy.max_retries == 3, "Default should have 3 retries");
    TEST_ASSERT(default_policy.retry_delay_ms == 1000, "Default delay should be 1000ms");

    auto aggressive = failure_policy::aggressive_policy();
    TEST_ASSERT(aggressive.max_retries == 5, "Aggressive should have 5 retries");
    TEST_ASSERT(aggressive.enable_failover == true, "Aggressive should enable failover");

    auto conservative = failure_policy::conservative_policy();
    TEST_ASSERT(conservative.max_retries == 1, "Conservative should have 1 retry");
    TEST_ASSERT(conservative.enable_failover == false, "Conservative should disable failover");

    return true;
}

// Test 17: Circuit Breaker
static bool test_circuit_breaker() {
    circuit_breaker cb(3, 60000, 2);  // 3 failures, 60s timeout, 2 successes to close

    TEST_ASSERT(cb.get_state() == CIRCUIT_CLOSED, "Should start closed");
    TEST_ASSERT(cb.allow_request(), "Should allow requests when closed");

    // Record failures
    cb.record_failure();
    cb.record_failure();
    TEST_ASSERT(cb.get_state() == CIRCUIT_CLOSED, "Should still be closed after 2 failures");

    cb.record_failure();
    TEST_ASSERT(cb.get_state() == CIRCUIT_OPEN, "Should open after 3 failures");
    TEST_ASSERT(!cb.allow_request(), "Should not allow requests when open");

    // Reset
    cb.reset();
    TEST_ASSERT(cb.get_state() == CIRCUIT_CLOSED, "Should be closed after reset");

    return true;
}

// Test 18: Token Estimation
static bool test_token_estimation() {
    std::string text = "This is a test sentence with some words.";
    int tokens = token_estimator::estimate_tokens(text);

    TEST_ASSERT(tokens > 0, "Should estimate positive tokens");
    TEST_ASSERT(tokens <= text.length(), "Token count should be less than character count");

    conversation_turn turn;
    turn.role = "user";
    turn.content = "Hello world";

    int turn_tokens = token_estimator::estimate_turn_tokens(turn);
    TEST_ASSERT(turn_tokens > 0, "Should estimate turn tokens");

    return true;
}

// Test 19: Agent Statistics
static bool test_agent_statistics() {
    auto& registry = agent_registry::instance();
    conversation_memory memory(1);

    auto agent = agent_factory::create_local_agent(
        "Stats Agent",
        "Stats",
        {"stats"},
        &memory
    );

    auto agent_ptr = static_cast<local_agent*>(agent.get());
    agent_ptr->set_inference_callback(mock_inference);

    std::string agent_id = agent->get_info().id;
    registry.register_agent(std::move(agent));
    registry.set_conversation_memory(&memory);

    // Send some requests
    agent_request req;
    req.prompt = "Test";

    registry.send_request(agent_id, req);
    registry.send_request(agent_id, req);

    auto stats = registry.get_agent_stats(agent_id);
    TEST_ASSERT(stats.total_requests >= 2, "Should have at least 2 requests");
    TEST_ASSERT(stats.successful_requests >= 2, "Should have at least 2 successes");

    // Cleanup
    registry.unregister_agent(agent_id);

    return true;
}

// Test 20: Registry Statistics
static bool test_registry_statistics() {
    auto& registry = agent_registry::instance();
    conversation_memory memory(1);

    // Register multiple agents
    auto agent1 = agent_factory::create_local_agent("Agent1", "A1", {"test"}, &memory);
    auto agent2 = agent_factory::create_local_agent("Agent2", "A2", {"test"}, &memory);

    std::string id1 = agent1->get_info().id;
    std::string id2 = agent2->get_info().id;

    registry.register_agent(std::move(agent1));
    registry.register_agent(std::move(agent2));

    auto stats = registry.get_stats();
    TEST_ASSERT(stats.total_agents >= 2, "Should have at least 2 agents");

    // Cleanup
    registry.unregister_agent(id1);
    registry.unregister_agent(id2);

    return true;
}

// Test 21: Thread Branching
static bool test_thread_branching() {
    conversation_memory memory(1);

    agent_request req;
    req.prompt = "Original";

    std::string parent_id = memory.create_thread("agent1", req);
    memory.add_turn(parent_id, "user", "Question", {}, {}, "agent1");
    memory.add_turn(parent_id, "assistant", "Answer", {}, {}, "agent1");

    std::string child_id = memory.branch_thread(parent_id, "agent2");

    TEST_ASSERT(!child_id.empty(), "Child thread should be created");
    TEST_ASSERT(child_id != parent_id, "Child should have different ID");
    TEST_ASSERT(memory.has_thread(child_id), "Child thread should exist");

    auto child_opt = memory.get_thread(child_id);
    TEST_ASSERT(child_opt.has_value(), "Should retrieve child thread");
    TEST_ASSERT(child_opt->parent_id == parent_id, "Parent ID should be set");
    TEST_ASSERT(child_opt->turns.size() == 2, "Child should inherit turns");

    return true;
}

// Test 22: Thread Cleanup
static bool test_thread_cleanup() {
    conversation_memory memory(1);

    agent_request req;
    req.prompt = "Test";

    std::string thread_id = memory.create_thread("agent1", req);
    TEST_ASSERT(memory.thread_count() == 1, "Should have 1 thread");

    TEST_ASSERT(memory.delete_thread(thread_id), "Should delete thread");
    TEST_ASSERT(memory.thread_count() == 0, "Should have 0 threads");
    TEST_ASSERT(!memory.has_thread(thread_id), "Thread should not exist");

    return true;
}

// Test 23: Concurrent Message Queue
static bool test_concurrent_message_queue() {
    message_queue queue(100);
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < 50; i++) {
            agent_message msg;
            msg.message_id = "msg" + std::to_string(i);
            if (queue.push(msg)) {
                produced.fetch_add(1);
            }
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        for (int i = 0; i < 50; i++) {
            agent_message msg;
            if (queue.pop(msg, 1000)) {
                consumed.fetch_add(1);
            }
        }
    });

    producer.join();
    consumer.join();

    TEST_ASSERT(produced.load() == 50, "Should produce 50 messages");
    TEST_ASSERT(consumed.load() == 50, "Should consume 50 messages");

    return true;
}

// Test 24: Error Type Conversion
static bool test_error_type_conversion() {
    TEST_ASSERT(std::string(error_type_to_string(ERROR_TYPE_TIMEOUT)) == "timeout",
                "Timeout should convert correctly");
    TEST_ASSERT(std::string(error_type_to_string(ERROR_TYPE_CONNECTION)) == "connection",
                "Connection should convert correctly");
    TEST_ASSERT(std::string(error_type_to_string(ERROR_TYPE_OFFLINE)) == "offline",
                "Offline should convert correctly");

    return true;
}

// Test 25: Agent Status
static bool test_agent_status() {
    conversation_memory memory(1);

    auto agent = agent_factory::create_local_agent("Test", "Test", {}, &memory);

    TEST_ASSERT(agent->get_info().status == AGENT_STATUS_IDLE, "Should start idle");

    agent->set_status(AGENT_STATUS_BUSY);
    TEST_ASSERT(agent->get_info().status == AGENT_STATUS_BUSY, "Should be busy");

    agent->set_status(AGENT_STATUS_ERROR);
    TEST_ASSERT(agent->get_info().status == AGENT_STATUS_ERROR, "Should be error");

    return true;
}

int main() {
    std::cout << "=== Agent Collaboration Framework Tests ===" << std::endl << std::endl;

    int total = 0;
    int passed = 0;
    int failed = 0;

    // Core functionality tests
    RUN_TEST(test_uuid_generation);
    RUN_TEST(test_timestamp_generation);
    RUN_TEST(test_message_serialization);
    RUN_TEST(test_response_serialization);
    RUN_TEST(test_message_queue);

    // Conversation memory tests
    RUN_TEST(test_conversation_memory);
    RUN_TEST(test_conversation_turns);
    RUN_TEST(test_conversation_history);
    RUN_TEST(test_context_reconstruction);
    RUN_TEST(test_thread_expiration);
    RUN_TEST(test_thread_branching);
    RUN_TEST(test_thread_cleanup);

    // Agent tests
    RUN_TEST(test_agent_creation);
    RUN_TEST(test_agent_registration);
    RUN_TEST(test_agent_discovery);
    RUN_TEST(test_agent_request_processing);
    RUN_TEST(test_multiturn_conversation);
    RUN_TEST(test_agent_statistics);
    RUN_TEST(test_agent_status);

    // Registry tests
    RUN_TEST(test_registry_statistics);

    // Failure handling tests
    RUN_TEST(test_failure_policy);
    RUN_TEST(test_circuit_breaker);
    RUN_TEST(test_error_type_conversion);

    // Utility tests
    RUN_TEST(test_token_estimation);

    // Concurrency tests
    RUN_TEST(test_concurrent_message_queue);

    std::cout << std::endl << "=== Test Results ===" << std::endl;
    std::cout << "Total:  " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    if (failed == 0) {
        std::cout << std::endl << "All tests passed! ✓" << std::endl;
        return 0;
    } else {
        std::cout << std::endl << "Some tests failed! ✗" << std::endl;
        return 1;
    }
}
