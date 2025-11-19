#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <memory>

namespace agent {

// Message types for agent communication
enum message_type {
    MESSAGE_TYPE_REQUEST,
    MESSAGE_TYPE_RESPONSE,
    MESSAGE_TYPE_NOTIFICATION,
    MESSAGE_TYPE_ERROR,
    MESSAGE_TYPE_HEARTBEAT,
    MESSAGE_TYPE_BROADCAST
};

// Response status codes
enum response_status {
    RESPONSE_STATUS_SUCCESS,
    RESPONSE_STATUS_ERROR,
    RESPONSE_STATUS_CONTINUATION_REQUIRED,
    RESPONSE_STATUS_TIMEOUT,
    RESPONSE_STATUS_NOT_FOUND,
    RESPONSE_STATUS_UNAVAILABLE
};

// Convert message type to string
const char* message_type_to_string(message_type type);

// Convert response status to string
const char* response_status_to_string(response_status status);

// Agent message structure
struct agent_message {
    std::string message_id;      // UUID
    std::string from_agent;      // Source agent ID
    std::string to_agent;        // Destination agent ID (empty for broadcast)
    message_type type;           // Message type
    std::string payload;         // JSON payload
    std::string thread_id;       // Associated conversation thread
    int64_t timestamp;           // Message timestamp (Unix epoch ms)
    int priority;                // Message priority (0-10, higher = more urgent)
    std::map<std::string, std::string> metadata;  // Custom metadata

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static agent_message from_json(const std::string& json_str);
};

// Agent request structure
struct agent_request {
    std::string prompt;          // User prompt
    std::string thread_id;       // Continuation thread ID (optional)
    std::vector<std::string> files;  // File references
    std::vector<std::string> images; // Image references
    std::map<std::string, std::string> params;  // Request parameters
    int max_tokens;              // Token limit (0 = no limit)
    float temperature;           // Sampling temperature
    std::string system_prompt;   // System prompt override

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static agent_request from_json(const std::string& json_str);
};

// Agent response structure
struct agent_response {
    response_status status;      // Response status
    std::string content;         // Response content
    std::string thread_id;       // Thread ID for continuation
    int tokens_used;             // Tokens consumed
    std::string error_message;   // Error details (if failed)
    std::string error_type;      // Error classification
    std::map<std::string, std::string> metadata;  // Custom metadata

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static agent_response from_json(const std::string& json_str);
};

// Continuation offer structure (for multi-turn conversations)
struct continuation_offer {
    std::string continuation_id; // Thread UUID for continuation
    std::string note;            // Instructions for agent
    int remaining_turns;         // Turn limit tracking
    int64_t expires_at;          // Expiration timestamp

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static continuation_offer from_json(const std::string& json_str);
};

// Message queue for async processing
class message_queue {
public:
    message_queue(size_t max_size = 10000);
    ~message_queue();

    // Push message to queue
    bool push(const agent_message& msg);

    // Pop message from queue (blocking with timeout)
    bool pop(agent_message& msg, int64_t timeout_ms = 0);

    // Get queue size
    size_t size() const;

    // Check if queue is empty
    bool empty() const;

    // Clear all messages
    void clear();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Generate UUID for messages and threads
std::string generate_uuid();

// Get current timestamp in milliseconds
int64_t get_timestamp_ms();

} // namespace agent
