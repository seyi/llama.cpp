#pragma once

#include "message.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>

namespace agent {

// Conversation turn structure
struct conversation_turn {
    std::string role;            // "user", "assistant", "system"
    std::string content;         // Turn content
    int64_t timestamp;           // Unix timestamp (ms)
    std::vector<std::string> files;  // Referenced files
    std::vector<std::string> images; // Referenced images
    std::string agent_id;        // Agent that created turn
    std::string model;           // Model used (if applicable)
    std::map<std::string, std::string> metadata;  // Custom metadata

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static conversation_turn from_json(const std::string& json_str);

    // Estimate token count for this turn
    int estimate_tokens() const;
};

// Conversation thread structure
struct conversation_thread {
    std::string thread_id;       // UUID
    std::string parent_id;       // Parent thread (for branching)
    int64_t created_at;          // Creation timestamp (ms)
    int64_t updated_at;          // Last update timestamp (ms)
    std::string initiating_agent;  // Agent that created thread
    std::vector<conversation_turn> turns;  // Conversation history
    std::map<std::string, std::string> context;  // Initial context
    int64_t expires_at;          // Expiration timestamp (ms)

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static conversation_thread from_json(const std::string& json_str);

    // Get total token count
    int estimate_total_tokens() const;

    // Get turn count
    size_t turn_count() const { return turns.size(); }
};

// Context reconstruction result
struct reconstructed_context {
    std::string full_context;    // Complete conversation history as text
    int tokens_used;             // Total tokens in context
    int turns_included;          // Number of turns included
    std::vector<std::string> files_included;  // Files referenced
    bool truncated;              // Whether context was truncated
};

// Conversation memory manager
class conversation_memory {
public:
    // Constructor with TTL configuration
    conversation_memory(int64_t ttl_hours = 3, size_t max_threads = 10000);
    ~conversation_memory();

    // Create new conversation thread
    std::string create_thread(const std::string& agent_id,
                            const agent_request& initial_request);

    // Add turn to existing thread
    bool add_turn(const std::string& thread_id,
                 const std::string& role,
                 const std::string& content,
                 const std::vector<std::string>& files = {},
                 const std::vector<std::string>& images = {},
                 const std::string& agent_id = "",
                 const std::string& model = "");

    // Get thread by ID
    std::optional<conversation_thread> get_thread(const std::string& thread_id);

    // Update thread timestamp (keeps it alive)
    bool touch_thread(const std::string& thread_id);

    // Delete thread
    bool delete_thread(const std::string& thread_id);

    // Check if thread exists
    bool has_thread(const std::string& thread_id) const;

    // Build conversation history for context
    // Returns formatted conversation history suitable for LLM context
    reconstructed_context build_conversation_history(
        const std::string& thread_id,
        int max_tokens = 0,  // 0 = no limit
        bool include_files = true
    );

    // Reconstruct request with thread context
    // Used for continuation requests
    agent_request reconstruct_request(const agent_request& continuation_request);

    // Clean up expired threads
    size_t cleanup_expired();

    // Get active thread count
    size_t thread_count() const;

    // Get all thread IDs for an agent
    std::vector<std::string> get_agent_threads(const std::string& agent_id) const;

    // Create child thread (branching)
    std::string branch_thread(const std::string& parent_id,
                             const std::string& agent_id);

    // Export thread to JSON
    std::string export_thread(const std::string& thread_id) const;

    // Import thread from JSON
    bool import_thread(const std::string& json_str);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Token estimation utilities
class token_estimator {
public:
    // Estimate tokens for text (rough approximation)
    static int estimate_tokens(const std::string& text);

    // Estimate tokens for file content
    static int estimate_file_tokens(const std::string& file_path);

    // Estimate tokens for conversation turn
    static int estimate_turn_tokens(const conversation_turn& turn);
};

} // namespace agent
