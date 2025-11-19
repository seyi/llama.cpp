#pragma once

#include "agent.h"
#include "message.h"
#include "conversation.h"
#include "failure.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>

namespace agent {

// Agent discovery query
struct agent_query {
    std::vector<std::string> capabilities;  // Required capabilities
    agent_status min_status;                // Minimum status (default: ACTIVE)
    bool require_all_capabilities;          // AND vs OR for capabilities
    std::map<std::string, std::string> metadata_filters;  // Metadata filters

    agent_query() : min_status(AGENT_STATUS_ACTIVE), require_all_capabilities(true) {}
};

// Agent registry (singleton)
class agent_registry {
public:
    // Get singleton instance
    static agent_registry& instance();

    // Disable copy and move
    agent_registry(const agent_registry&) = delete;
    agent_registry& operator=(const agent_registry&) = delete;

    // Register agent
    bool register_agent(std::unique_ptr<agent_interface> agent);

    // Unregister agent
    bool unregister_agent(const std::string& agent_id);

    // Get agent by ID
    agent_interface* get_agent(const std::string& agent_id);

    // Find agents by capability
    std::vector<agent_info> find_agents(const agent_query& query);

    // Get all agents
    std::vector<agent_info> list_agents();

    // Send message to agent
    agent_response send_message(const agent_message& message);

    // Send request to agent
    agent_response send_request(const std::string& agent_id,
                               const agent_request& request);

    // Send request with failure policy
    agent_response send_request_with_policy(
        const std::string& agent_id,
        const agent_request& request,
        const failure_policy& policy
    );

    // Broadcast message to all agents
    std::vector<agent_response> broadcast_message(const agent_message& message);

    // Multi-agent consensus
    // Sends same request to multiple agents and collects responses
    struct consensus_result {
        std::vector<agent_response> responses;
        std::string synthesized_response;  // Optional synthesis
        std::map<std::string, int> response_similarity;  // Similarity scores
    };

    consensus_result consensus_request(
        const std::vector<std::string>& agent_ids,
        const agent_request& request,
        bool synthesize = false
    );

    // Route request to best agent based on capabilities
    std::optional<std::string> route_request(const agent_request& request);

    // Health check for all agents
    void health_check();

    // Get agent statistics
    agent_stats get_agent_stats(const std::string& agent_id);

    // Get registry statistics
    struct registry_stats {
        int total_agents;
        int active_agents;
        int busy_agents;
        int error_agents;
        int offline_agents;
        int64_t total_messages;
        int64_t total_requests;
        int64_t total_failures;
        std::map<std::string, agent_stats> agent_stats_map;

        std::string to_json() const;
    };
    registry_stats get_stats();

    // Set conversation memory (shared across agents)
    void set_conversation_memory(conversation_memory* memory);

    // Get conversation memory
    conversation_memory* get_conversation_memory();

    // Set message queue
    void set_message_queue(message_queue* queue);

    // Get message queue
    message_queue* get_message_queue();

    // Enable/disable async message processing
    void set_async_mode(bool enabled);

    // Start message processing thread
    void start_message_processor();

    // Stop message processing thread
    void stop_message_processor();

    // Set message handler callback
    using message_handler = std::function<void(const agent_message&, const agent_response&)>;
    void set_message_handler(message_handler handler);

    // Get last failure for agent
    std::optional<failure_record> get_last_failure(const std::string& agent_id);

    // Clear failure history
    void clear_failures();

    // Export registry state to JSON
    std::string export_state() const;

    // Import registry state from JSON
    bool import_state(const std::string& json_str);

private:
    agent_registry();
    ~agent_registry();

    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace agent
