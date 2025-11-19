#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <functional>
#include <memory>

namespace agent {

// Error types
enum error_type {
    ERROR_TYPE_NONE,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_CONNECTION,
    ERROR_TYPE_UNAVAILABLE,
    ERROR_TYPE_OVERLOAD,
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_INVALID_RESPONSE,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_AUTHORIZATION,
    ERROR_TYPE_RATE_LIMIT,
    ERROR_TYPE_CONTEXT_EXPIRED,
    ERROR_TYPE_THREAD_NOT_FOUND,
    ERROR_TYPE_AGENT_NOT_FOUND,
    ERROR_TYPE_OFFLINE,
    ERROR_TYPE_INTERNAL_ERROR,
    ERROR_TYPE_UNKNOWN
};

// Convert error type to string
const char* error_type_to_string(error_type type);

// Failure policy configuration
struct failure_policy {
    int max_retries;             // Maximum retry attempts
    int64_t retry_delay_ms;      // Initial retry delay
    float backoff_multiplier;    // Exponential backoff factor (e.g., 2.0)
    int64_t max_retry_delay_ms;  // Maximum retry delay cap
    int64_t timeout_ms;          // Request timeout
    bool enable_failover;        // Auto-failover on failure
    std::vector<std::string> fallback_agents;  // Failover agents
    bool log_failures;           // Log failures to history

    // Default policy
    static failure_policy default_policy();

    // Aggressive retry policy
    static failure_policy aggressive_policy();

    // Conservative policy (fewer retries)
    static failure_policy conservative_policy();
};

// Failure record
struct failure_record {
    std::string agent_id;        // Failed agent
    error_type error;            // Error type
    std::string error_message;   // Error details
    int64_t timestamp;           // Failure timestamp (ms)
    std::string thread_id;       // Associated thread (if any)
    std::string message_id;      // Failed message ID (if any)
    int retry_count;             // Retry attempts made
    bool recovered;              // Whether failure was recovered
    std::string recovery_agent;  // Agent that handled recovery (if any)

    // Serialize to JSON
    std::string to_json() const;

    // Deserialize from JSON
    static failure_record from_json(const std::string& json_str);
};

// Circuit breaker state
enum circuit_state {
    CIRCUIT_CLOSED,   // Normal operation
    CIRCUIT_OPEN,     // Too many failures, reject requests
    CIRCUIT_HALF_OPEN // Testing if service recovered
};

// Circuit breaker for agent failure detection
class circuit_breaker {
public:
    // Constructor
    circuit_breaker(int failure_threshold = 5,
                   int64_t timeout_ms = 60000,
                   int success_threshold = 2);
    ~circuit_breaker();

    // Record success
    void record_success();

    // Record failure
    void record_failure();

    // Check if request is allowed
    bool allow_request();

    // Get current state
    circuit_state get_state() const;

    // Reset circuit breaker
    void reset();

    // Get statistics
    struct stats {
        circuit_state state;
        int failure_count;
        int success_count;
        int64_t last_failure_time;
        int64_t last_state_change;
    };
    stats get_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Failure handler interface
class failure_handler {
public:
    virtual ~failure_handler() = default;

    // Handle failure
    // Returns true if failure was handled/recovered
    virtual bool handle_failure(const failure_record& record) = 0;

    // Check if this handler can handle the error type
    virtual bool can_handle(error_type type) const = 0;
};

// Retry handler with exponential backoff
class retry_handler : public failure_handler {
public:
    retry_handler(const failure_policy& policy);
    ~retry_handler();

    bool handle_failure(const failure_record& record) override;
    bool can_handle(error_type type) const override;

    // Execute function with retry logic
    template<typename Func>
    bool execute_with_retry(Func func, error_type& out_error);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Failover handler (switches to backup agent)
class failover_handler : public failure_handler {
public:
    failover_handler(const std::vector<std::string>& fallback_agents);
    ~failover_handler();

    bool handle_failure(const failure_record& record) override;
    bool can_handle(error_type type) const override;

    // Get next fallback agent
    std::string get_next_fallback();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Dead letter queue for failed messages
class dead_letter_queue {
public:
    dead_letter_queue(size_t max_size = 1000);
    ~dead_letter_queue();

    // Add failed message
    void add_message(const std::string& message_id,
                    const std::string& payload,
                    const failure_record& failure);

    // Get failed messages
    struct dead_letter {
        std::string message_id;
        std::string payload;
        failure_record failure;
        int64_t queued_at;
    };
    std::vector<dead_letter> get_messages(int limit = 100);

    // Retry failed message
    bool retry_message(const std::string& message_id);

    // Remove message from queue
    bool remove_message(const std::string& message_id);

    // Get queue size
    size_t size() const;

    // Clear queue
    void clear();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Failure manager (coordinates all failure handling)
class failure_manager {
public:
    failure_manager();
    ~failure_manager();

    // Add failure handler
    void add_handler(std::unique_ptr<failure_handler> handler);

    // Record failure
    void record_failure(const failure_record& record);

    // Handle failure with registered handlers
    bool handle_failure(failure_record& record);

    // Get failure history for agent
    std::vector<failure_record> get_history(const std::string& agent_id,
                                           int limit = 100);

    // Get circuit breaker for agent
    circuit_breaker* get_circuit_breaker(const std::string& agent_id);

    // Get dead letter queue
    dead_letter_queue* get_dead_letter_queue();

    // Clear failure history
    void clear_history();

    // Get statistics
    struct stats {
        int64_t total_failures;
        int64_t recovered_failures;
        std::map<error_type, int64_t> failures_by_type;
        std::map<std::string, int64_t> failures_by_agent;
        int dead_letters;
    };
    stats get_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace agent
