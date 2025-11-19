// Simple agent collaboration example
// Demonstrates multi-agent conversation with memory and failure handling

#include "../../common/agent/agent.h"
#include "../../common/agent/registry.h"
#include "../../common/agent/conversation.h"
#include "../../common/agent/message.h"
#include "../../common/agent/failure.h"

#include <iostream>
#include <string>
#include <vector>

using namespace agent;

// Simple inference callback for demo purposes
std::string simple_inference(const std::string& prompt, const std::map<std::string, std::string>& params) {
    // This is a mock inference function
    // In a real implementation, this would call llama.cpp inference
    return "This is a simulated response to: " + prompt.substr(0, 50) + "...";
}

int main(int argc, char** argv) {
    std::cout << "=== Agent Collaboration Example ===\n\n";

    // 1. Create conversation memory
    std::cout << "1. Creating conversation memory...\n";
    conversation_memory memory(3);  // 3-hour TTL
    std::cout << "   ✓ Memory created with 3-hour TTL\n\n";

    // 2. Get registry instance
    std::cout << "2. Getting agent registry...\n";
    auto& registry = agent_registry::instance();
    registry.set_conversation_memory(&memory);
    std::cout << "   ✓ Registry initialized\n\n";

    // 3. Create and register agents
    std::cout << "3. Creating agents...\n";

    // Create code analysis agent
    auto code_agent = agent_factory::create_local_agent(
        "Code Analyzer",
        "Analyzes code for quality and best practices",
        {"code_analysis", "refactoring", "optimization"},
        &memory
    );

    auto code_agent_ptr = static_cast<local_agent*>(code_agent.get());
    code_agent_ptr->set_inference_callback(simple_inference);
    auto code_agent_id = code_agent->get_info().id;
    std::cout << "   ✓ Code Analyzer agent created (ID: " << code_agent_id << ")\n";

    // Create documentation agent
    auto doc_agent = agent_factory::create_local_agent(
        "Documentation Writer",
        "Generates and reviews documentation",
        {"documentation", "technical_writing", "code_explanation"},
        &memory
    );

    auto doc_agent_ptr = static_cast<local_agent*>(doc_agent.get());
    doc_agent_ptr->set_inference_callback(simple_inference);
    auto doc_agent_id = doc_agent->get_info().id;
    std::cout << "   ✓ Documentation Writer agent created (ID: " << doc_agent_id << ")\n";

    // Create testing agent
    auto test_agent = agent_factory::create_local_agent(
        "Test Generator",
        "Creates and reviews test cases",
        {"testing", "test_generation", "qa"},
        &memory
    );

    auto test_agent_ptr = static_cast<local_agent*>(test_agent.get());
    test_agent_ptr->set_inference_callback(simple_inference);
    auto test_agent_id = test_agent->get_info().id;
    std::cout << "   ✓ Test Generator agent created (ID: " << test_agent_id << ")\n\n";

    // Register agents
    std::cout << "4. Registering agents...\n";
    registry.register_agent(std::move(code_agent));
    registry.register_agent(std::move(doc_agent));
    registry.register_agent(std::move(test_agent));
    std::cout << "   ✓ All agents registered\n\n";

    // 5. List all agents
    std::cout << "5. Listing all agents:\n";
    auto agents = registry.list_agents();
    for (const auto& info : agents) {
        std::cout << "   - " << info.name << " (" << info.id << ")\n";
        std::cout << "     Status: " << agent_status_to_string(info.status) << "\n";
        std::cout << "     Capabilities: ";
        for (size_t i = 0; i < info.capabilities.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << info.capabilities[i];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // 6. Send request to code agent
    std::cout << "6. Sending request to Code Analyzer...\n";
    agent_request req1;
    req1.prompt = "Analyze this function for potential improvements:\nvoid process(int x) { return x * 2; }";
    req1.max_tokens = 500;
    req1.temperature = 0.7f;

    auto response1 = registry.send_request(code_agent_id, req1);
    std::cout << "   Status: " << response_status_to_string(response1.status) << "\n";
    std::cout << "   Response: " << response1.content << "\n";
    std::cout << "   Thread ID: " << response1.thread_id << "\n";
    std::cout << "   Tokens used: " << response1.tokens_used << "\n\n";

    // 7. Continue conversation
    if (!response1.thread_id.empty()) {
        std::cout << "7. Continuing conversation...\n";
        agent_request req2;
        req2.prompt = "Can you provide a refactored version?";
        req2.thread_id = response1.thread_id;
        req2.max_tokens = 500;

        auto response2 = registry.send_request(code_agent_id, req2);
        std::cout << "   Status: " << response_status_to_string(response2.status) << "\n";
        std::cout << "   Response: " << response2.content << "\n\n";
    }

    // 8. Multi-agent consensus
    std::cout << "8. Getting multi-agent consensus...\n";
    agent_request consensus_req;
    consensus_req.prompt = "What are the best practices for error handling in C++?";
    consensus_req.max_tokens = 300;

    auto consensus_result = registry.consensus_request(
        {code_agent_id, doc_agent_id, test_agent_id},
        consensus_req,
        true  // synthesize responses
    );

    std::cout << "   Received " << consensus_result.responses.size() << " responses:\n";
    for (size_t i = 0; i < consensus_result.responses.size(); ++i) {
        std::cout << "   Agent " << (i+1) << " response: "
                  << consensus_result.responses[i].content.substr(0, 50) << "...\n";
    }
    if (!consensus_result.synthesized_response.empty()) {
        std::cout << "\n   Synthesized response:\n";
        std::cout << consensus_result.synthesized_response << "\n";
    }
    std::cout << "\n";

    // 9. Agent discovery by capability
    std::cout << "9. Finding agents with 'testing' capability...\n";
    agent_query query;
    query.capabilities = {"testing"};
    query.min_status = AGENT_STATUS_IDLE;

    auto found_agents = registry.find_agents(query);
    std::cout << "   Found " << found_agents.size() << " agent(s):\n";
    for (const auto& info : found_agents) {
        std::cout << "   - " << info.name << "\n";
    }
    std::cout << "\n";

    // 10. Failure handling with retry policy
    std::cout << "10. Testing failure handling...\n";
    failure_policy policy = failure_policy::default_policy();
    policy.max_retries = 2;
    policy.enable_failover = true;
    policy.fallback_agents = {doc_agent_id};  // Fallback to doc agent

    agent_request retry_req;
    retry_req.prompt = "Test retry logic";
    retry_req.max_tokens = 100;

    auto retry_response = registry.send_request_with_policy(
        code_agent_id,
        retry_req,
        policy
    );
    std::cout << "   Status: " << response_status_to_string(retry_response.status) << "\n";
    std::cout << "   Response: " << retry_response.content << "\n\n";

    // 11. Get statistics
    std::cout << "11. Agent statistics:\n";
    auto stats = registry.get_stats();
    std::cout << "   Total agents: " << stats.total_agents << "\n";
    std::cout << "   Active agents: " << stats.active_agents << "\n";
    std::cout << "   Total requests: " << stats.total_requests << "\n";
    std::cout << "   Total messages: " << stats.total_messages << "\n";
    std::cout << "   Total failures: " << stats.total_failures << "\n\n";

    for (const auto& [agent_id, agent_stats] : stats.agent_stats_map) {
        auto agent_info = registry.get_agent(agent_id)->get_info();
        std::cout << "   Agent: " << agent_info.name << "\n";
        std::cout << "   - Total requests: " << agent_stats.total_requests << "\n";
        std::cout << "   - Successful: " << agent_stats.successful_requests << "\n";
        std::cout << "   - Failed: " << agent_stats.failed_requests << "\n";
        std::cout << "   - Total tokens: " << agent_stats.total_tokens << "\n";
        std::cout << "   - Avg response time: " << agent_stats.avg_response_time_ms << " ms\n\n";
    }

    // 12. Conversation memory stats
    std::cout << "12. Conversation memory:\n";
    std::cout << "   Total threads: " << memory.thread_count() << "\n";
    std::cout << "   Code agent threads: " << memory.get_agent_threads(code_agent_id).size() << "\n\n";

    // 13. Cleanup
    std::cout << "13. Cleaning up...\n";
    auto expired_count = memory.cleanup_expired();
    std::cout << "   Cleaned up " << expired_count << " expired threads\n";

    registry.unregister_agent(code_agent_id);
    registry.unregister_agent(doc_agent_id);
    registry.unregister_agent(test_agent_id);
    std::cout << "   All agents unregistered\n\n";

    std::cout << "=== Example completed successfully ===\n";
    return 0;
}
