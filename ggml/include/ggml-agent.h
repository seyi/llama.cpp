#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward declarations
struct ggml_agent;
struct ggml_agent_msg;
struct ggml_agent_supervisor;
struct ggml_agent_coordinator;

// ============================================================================
// Message Types and Structures
// ============================================================================

enum ggml_agent_msg_type {
    GGML_AGENT_MSG_USER,        // User-defined message
    GGML_AGENT_MSG_HEARTBEAT,   // Health check ping
    GGML_AGENT_MSG_HEARTBEAT_ACK, // Health check response
    GGML_AGENT_MSG_SHUTDOWN,    // Graceful shutdown request
    GGML_AGENT_MSG_ERROR,       // Error notification
    GGML_AGENT_MSG_TASK,        // Task assignment
    GGML_AGENT_MSG_TASK_RESULT, // Task completion result
    GGML_AGENT_MSG_DOC_EDIT,    // Document edit request
    GGML_AGENT_MSG_DOC_UPDATE,  // Document update notification
    GGML_AGENT_MSG_LOCK_REQUEST,  // Request lock on resource
    GGML_AGENT_MSG_LOCK_RELEASE,  // Release lock on resource
    GGML_AGENT_MSG_LOCK_ACQUIRED, // Lock acquisition confirmation
    GGML_AGENT_MSG_LOCK_DENIED,   // Lock acquisition denied
};

struct ggml_agent_msg {
    std::string from_id;
    std::string to_id;
    ggml_agent_msg_type type;
    std::vector<uint8_t> payload;
    uint64_t timestamp_ms;
    std::string msg_id;
    std::string correlation_id; // For request-response tracking

    ggml_agent_msg(const std::string& from, const std::string& to,
                   ggml_agent_msg_type t, const std::vector<uint8_t>& data = {})
        : from_id(from), to_id(to), type(t), payload(data),
          timestamp_ms(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()),
          msg_id(generate_msg_id()) {}

    static std::string generate_msg_id();
};

// ============================================================================
// Failure Handling
// ============================================================================

struct ggml_agent_retry_policy {
    uint32_t max_attempts = 3;
    uint64_t initial_backoff_ms = 100;
    float backoff_multiplier = 2.0f;
    uint64_t max_backoff_ms = 10000;

    uint64_t get_backoff(uint32_t attempt) const {
        uint64_t backoff = initial_backoff_ms * std::pow(backoff_multiplier, attempt);
        return std::min(backoff, max_backoff_ms);
    }
};

enum ggml_agent_circuit_state {
    GGML_AGENT_CIRCUIT_CLOSED,     // Normal operation
    GGML_AGENT_CIRCUIT_OPEN,       // Failing, reject requests
    GGML_AGENT_CIRCUIT_HALF_OPEN,  // Testing recovery
};

struct ggml_agent_circuit_breaker {
    std::atomic<ggml_agent_circuit_state> state{GGML_AGENT_CIRCUIT_CLOSED};
    std::atomic<uint32_t> failure_count{0};
    std::atomic<uint32_t> success_count{0};
    uint32_t failure_threshold = 5;
    uint32_t success_threshold = 2;
    uint64_t open_timeout_ms = 30000; // 30 seconds
    std::atomic<uint64_t> last_failure_time_ms{0};

    bool allow_request();
    void record_success();
    void record_failure();
    void reset();
};

// ============================================================================
// Health Monitoring
// ============================================================================

struct ggml_agent_health {
    std::string agent_id;
    std::atomic<uint64_t> last_heartbeat_ms{0};
    uint64_t timeout_ms = 5000; // 5 second timeout
    std::atomic<bool> is_healthy{true};

    bool check_health() const {
        uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return (now - last_heartbeat_ms.load()) < timeout_ms;
    }

    void update_heartbeat() {
        last_heartbeat_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        is_healthy = true;
    }
};

// ============================================================================
// Actor Base Class
// ============================================================================

enum ggml_agent_state {
    GGML_AGENT_STATE_CREATED,
    GGML_AGENT_STATE_STARTING,
    GGML_AGENT_STATE_RUNNING,
    GGML_AGENT_STATE_STOPPING,
    GGML_AGENT_STATE_STOPPED,
    GGML_AGENT_STATE_FAILED,
};

using ggml_agent_msg_handler = std::function<void(const ggml_agent_msg&)>;

struct ggml_agent {
    std::string id;
    std::atomic<ggml_agent_state> state{GGML_AGENT_STATE_CREATED};

    // Message queue
    std::queue<ggml_agent_msg> msg_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // Processing thread
    std::thread worker_thread;
    std::atomic<bool> should_stop{false};

    // Message handlers
    std::unordered_map<ggml_agent_msg_type, ggml_agent_msg_handler> handlers;

    // Health monitoring
    ggml_agent_health health;

    // Failure handling
    ggml_agent_circuit_breaker circuit_breaker;
    ggml_agent_retry_policy retry_policy;

    // Supervisor reference
    ggml_agent_supervisor* supervisor = nullptr;

    // Constructor
    ggml_agent(const std::string& agent_id);
    virtual ~ggml_agent();

    // Lifecycle
    virtual void start();
    virtual void stop();
    void join();

    // Message passing
    void send(const ggml_agent_msg& msg);
    void send_to(const std::string& to_id, ggml_agent_msg_type type,
                 const std::vector<uint8_t>& payload = {});

    // Message handling
    void register_handler(ggml_agent_msg_type type, ggml_agent_msg_handler handler);

    // Health
    void send_heartbeat(const std::string& to_id);

protected:
    virtual void run();
    virtual void process_message(const ggml_agent_msg& msg);
    virtual void on_start() {}
    virtual void on_stop() {}
    virtual void on_message(const ggml_agent_msg& /* msg */) {}

private:
    void default_heartbeat_handler(const ggml_agent_msg& msg);
    void default_shutdown_handler(const ggml_agent_msg& msg);
};

// ============================================================================
// Agent Registry
// ============================================================================

struct ggml_agent_registry {
    std::unordered_map<std::string, std::shared_ptr<ggml_agent>> agents;
    std::mutex registry_mutex;

    static ggml_agent_registry& instance() {
        static ggml_agent_registry inst;
        return inst;
    }

    void register_agent(std::shared_ptr<ggml_agent> agent);
    void unregister_agent(const std::string& id);
    std::shared_ptr<ggml_agent> get_agent(const std::string& id);
    std::vector<std::string> list_agents();

    // Message routing
    bool route_message(const ggml_agent_msg& msg);
    void broadcast(const ggml_agent_msg& msg, const std::string& except_id = "");
};

// ============================================================================
// Supervisor Actor (Failure Recovery)
// ============================================================================

enum ggml_agent_restart_strategy {
    GGML_AGENT_RESTART_ONE_FOR_ONE,  // Restart only failed agent
    GGML_AGENT_RESTART_ONE_FOR_ALL,  // Restart all agents
    GGML_AGENT_RESTART_REST_FOR_ONE, // Restart failed and all started after it
};

struct ggml_agent_supervisor : public ggml_agent {
    std::vector<std::shared_ptr<ggml_agent>> children;
    std::mutex children_mutex;
    ggml_agent_restart_strategy strategy = GGML_AGENT_RESTART_ONE_FOR_ONE;
    uint32_t max_restarts = 3;
    uint64_t max_restart_window_ms = 60000; // 1 minute

    // Restart tracking
    std::unordered_map<std::string, std::vector<uint64_t>> restart_history;

    // Health monitoring thread
    std::thread health_monitor_thread;
    uint64_t health_check_interval_ms = 1000; // 1 second

    ggml_agent_supervisor(const std::string& id);
    virtual ~ggml_agent_supervisor();

    void start() override;
    void stop() override;

    // Child management
    void add_child(std::shared_ptr<ggml_agent> child);
    void remove_child(const std::string& child_id);

    // Failure handling
    void handle_child_failure(const std::string& child_id);
    bool should_restart(const std::string& child_id);
    void restart_child(const std::string& child_id);
    void restart_all_children();

protected:
    void run() override;
    void monitor_health();
};

// ============================================================================
// Document Coordinator Actor
// ============================================================================

struct ggml_doc_section {
    size_t start_pos;
    size_t end_pos;
    std::string locked_by;
    bool is_locked() const { return !locked_by.empty(); }
};

struct ggml_agent_coordinator : public ggml_agent {
    // Document state
    std::vector<uint8_t> document;
    std::vector<ggml_doc_section> sections;
    std::mutex doc_mutex;

    // Lock tracking
    std::unordered_map<std::string, std::vector<size_t>> agent_locks; // agent_id -> section indices

    // Edit queue for serialization
    std::queue<ggml_agent_msg> edit_queue;
    std::mutex edit_mutex;

    ggml_agent_coordinator(const std::string& id, size_t num_sections = 10);

    void start() override;

    // Document operations
    bool try_lock_section(const std::string& agent_id, size_t section_idx);
    bool release_section(const std::string& agent_id, size_t section_idx);
    void apply_edit(const std::string& agent_id, size_t section_idx,
                   const std::vector<uint8_t>& new_content);

    // Broadcast updates
    void broadcast_update(size_t section_idx);

protected:
    void on_start() override;

private:
    void handle_lock_request(const ggml_agent_msg& msg);
    void handle_lock_release(const ggml_agent_msg& msg);
    void handle_doc_edit(const ggml_agent_msg& msg);
};

// ============================================================================
// C API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Agent lifecycle
struct ggml_agent* ggml_agent_create(const char* id);
void ggml_agent_free(struct ggml_agent* agent);
void ggml_agent_start(struct ggml_agent* agent);
void ggml_agent_stop(struct ggml_agent* agent);

// Message passing
void ggml_agent_send_msg(struct ggml_agent* agent, const struct ggml_agent_msg* msg);

// Supervisor
struct ggml_agent_supervisor* ggml_agent_supervisor_create(const char* id);
void ggml_agent_supervisor_add_child(struct ggml_agent_supervisor* supervisor,
                                     struct ggml_agent* child);

// Coordinator
struct ggml_agent_coordinator* ggml_agent_coordinator_create(const char* id, size_t num_sections);

#ifdef __cplusplus
}
#endif
