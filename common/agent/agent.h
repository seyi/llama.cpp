#pragma once

#include "message.h"
#include "conversation.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace agent {

// Agent status
enum agent_status {
    AGENT_STATUS_ACTIVE,     // Agent is active and can accept requests
    AGENT_STATUS_IDLE,       // Agent is idle (no active requests)
    AGENT_STATUS_BUSY,       // Agent is processing requests
    AGENT_STATUS_ERROR,      // Agent encountered an error
    AGENT_STATUS_OFFLINE,    // Agent is offline/unreachable
    AGENT_STATUS_UNKNOWN     // Status unknown
};

// Convert agent status to string
const char* agent_status_to_string(agent_status status);

// Agent information structure
struct agent_info {
    std::string id;              // Unique agent identifier (UUID)
    std::string name;            // Human-readable name
    std::string description;     // Agent purpose/capabilities
    std::vector<std::string> capabilities;  // Agent capabilities/tags
    std::string endpoint;        // Connection endpoint (URL, socket, etc.)
    agent_status status;         // Current agent status
    int64_t last_heartbeat;      // Last heartbeat timestamp (ms)
    int64_t created_at;          // Agent registration timestamp (ms)
    std::map<std::string, std::string> metadata;  // Custom metadata

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static agent_info from_json(const std::string& json_str);

    // Check if agent has capability
    bool has_capability(const std::string& capability) const;

    // Check if agent is healthy (based on heartbeat)
    bool is_healthy(int64_t timeout_ms = 60000) const;
};

// Agent statistics
struct agent_stats {
    std::string agent_id;        // Agent identifier
    int64_t total_requests;      // Total requests processed
    int64_t successful_requests; // Successful requests
    int64_t failed_requests;     // Failed requests
    int64_t total_tokens;        // Total tokens processed
    double avg_response_time_ms; // Average response time
    int64_t last_request_time;   // Last request timestamp
    int active_threads;          // Active conversation threads

    // Serialize to JSON
    std::string to_json() const;
};

// Agent interface (abstract base class)
class agent_interface {
public:
    virtual ~agent_interface() = default;

    // Get agent info
    virtual agent_info get_info() const = 0;

    // Process request
    virtual agent_response process_request(const agent_request& request) = 0;

    // Handle message
    virtual agent_response handle_message(const agent_message& message) = 0;

    // Update agent status
    virtual void set_status(agent_status status) = 0;

    // Send heartbeat
    virtual void heartbeat() = 0;

    // Get agent statistics
    virtual agent_stats get_stats() const = 0;

    // Shutdown agent
    virtual void shutdown() = 0;
};

// Local agent implementation (wraps llama.cpp model)
class local_agent : public agent_interface {
public:
    // Constructor
    local_agent(const agent_info& info,
               conversation_memory* memory = nullptr);
    ~local_agent();

    // Agent interface implementation
    agent_info get_info() const override;
    agent_response process_request(const agent_request& request) override;
    agent_response handle_message(const agent_message& message) override;
    void set_status(agent_status status) override;
    void heartbeat() override;
    agent_stats get_stats() const override;
    void shutdown() override;

    // Set model context (for inference)
    void set_model_context(void* ctx);  // llama_context*

    // Set inference callback
    using inference_callback = std::function<std::string(
        const std::string& prompt,
        const std::map<std::string, std::string>& params
    )>;
    void set_inference_callback(inference_callback callback);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Remote agent implementation (connects to external agent)
class remote_agent : public agent_interface {
public:
    // Constructor
    remote_agent(const agent_info& info);
    ~remote_agent();

    // Agent interface implementation
    agent_info get_info() const override;
    agent_response process_request(const agent_request& request) override;
    agent_response handle_message(const agent_message& message) override;
    void set_status(agent_status status) override;
    void heartbeat() override;
    agent_stats get_stats() const override;
    void shutdown() override;

    // Set HTTP client configuration
    void set_timeout(int64_t timeout_ms);
    void set_retry_policy(int max_retries, int64_t retry_delay_ms);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Agent factory
class agent_factory {
public:
    // Create local agent
    static std::unique_ptr<agent_interface> create_local_agent(
        const std::string& name,
        const std::string& description,
        const std::vector<std::string>& capabilities,
        conversation_memory* memory = nullptr
    );

    // Create remote agent
    static std::unique_ptr<agent_interface> create_remote_agent(
        const std::string& endpoint,
        const std::string& name = "",
        const std::string& description = "",
        const std::vector<std::string>& capabilities = {}
    );
};

} // namespace agent
