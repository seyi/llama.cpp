#pragma once

#include "llama.h"
#include "log.h"

#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

//
// Agent Collaboration System for llama.cpp
// Inspired by Apache Spark's distributed computing architecture
//

namespace agent_collab {

// ============================================================================
// Core Types and Enums
// ============================================================================

enum agent_task_type {
    AGENT_TASK_TYPE_ANALYZE,      // Analyze code/data
    AGENT_TASK_TYPE_GENERATE,     // Generate code/content
    AGENT_TASK_TYPE_TEST,         // Run tests
    AGENT_TASK_TYPE_REVIEW,       // Review work
    AGENT_TASK_TYPE_REFACTOR,     // Refactor code
    AGENT_TASK_TYPE_DOCUMENT,     // Generate documentation
    AGENT_TASK_TYPE_CONSENSUS,    // Participate in voting
    AGENT_TASK_TYPE_CUSTOM        // User-defined
};

enum agent_state {
    AGENT_STATE_INITIALIZING,
    AGENT_STATE_IDLE,
    AGENT_STATE_ASSIGNED,
    AGENT_STATE_EXECUTING,
    AGENT_STATE_WAITING,
    AGENT_STATE_REPORTING,
    AGENT_STATE_FAILED,
    AGENT_STATE_RECOVERING,
    AGENT_STATE_TERMINATED
};

enum message_type {
    MSG_TYPE_REQUEST,       // Agent requests assistance
    MSG_TYPE_RESPONSE,      // Reply to request
    MSG_TYPE_BROADCAST,     // Message to all agents
    MSG_TYPE_DIRECT,        // Point-to-point message
    MSG_TYPE_EVENT,         // Event notification
    MSG_TYPE_CONSENSUS      // Voting message
};

enum consensus_type {
    CONSENSUS_SIMPLE_MAJORITY,    // >50% agreement
    CONSENSUS_SUPERMAJORITY,      // >=66% agreement
    CONSENSUS_UNANIMOUS,          // 100% agreement
    CONSENSUS_WEIGHTED            // Weighted by agent expertise
};

enum task_status {
    TASK_STATUS_PENDING,
    TASK_STATUS_ASSIGNED,
    TASK_STATUS_EXECUTING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_CANCELLED
};

// ============================================================================
// Core Data Structures
// ============================================================================

struct knowledge_entry {
    std::string key;
    std::string value;
    std::string contributor_id;
    int64_t timestamp;
    int version;
    std::vector<std::string> tags;

    json to_json() const {
        return json{
            {"key", key},
            {"value", value},
            {"contributor_id", contributor_id},
            {"timestamp", timestamp},
            {"version", version},
            {"tags", tags}
        };
    }
};

struct task_result {
    std::string task_id;
    std::string agent_id;
    std::string result;
    bool success;
    std::string error_message;
    int64_t duration_ms;

    json to_json() const {
        return json{
            {"task_id", task_id},
            {"agent_id", agent_id},
            {"result", result},
            {"success", success},
            {"error_message", error_message},
            {"duration_ms", duration_ms}
        };
    }
};

struct agent_task {
    std::string task_id;
    agent_task_type type;
    std::string description;
    json parameters;
    std::vector<std::string> dependencies;
    std::vector<std::string> required_roles;
    int priority;  // 0-10 (10 = highest)
    std::string parent_task_id;
    int64_t created_at;
    int64_t deadline;  // 0 = no deadline
    task_status status;
    std::string assigned_agent_id;

    json to_json() const {
        return json{
            {"task_id", task_id},
            {"type", type},
            {"description", description},
            {"parameters", parameters},
            {"dependencies", dependencies},
            {"required_roles", required_roles},
            {"priority", priority},
            {"parent_task_id", parent_task_id},
            {"created_at", created_at},
            {"deadline", deadline},
            {"status", status},
            {"assigned_agent_id", assigned_agent_id}
        };
    }
};

struct agent_message {
    std::string message_id;
    std::string from_agent_id;
    std::string to_agent_id;  // Empty for broadcast
    message_type type;
    std::string subject;
    json payload;
    int64_t timestamp;
    std::string conversation_id;

    json to_json() const {
        return json{
            {"message_id", message_id},
            {"from_agent_id", from_agent_id},
            {"to_agent_id", to_agent_id},
            {"type", type},
            {"subject", subject},
            {"payload", payload},
            {"timestamp", timestamp},
            {"conversation_id", conversation_id}
        };
    }
};

struct consensus_vote {
    std::string vote_id;
    std::string question;
    std::vector<std::string> options;
    consensus_type type;
    std::map<std::string, std::string> votes;  // agent_id -> option
    std::map<std::string, float> weights;      // agent_id -> weight
    int64_t deadline;
    std::string result;
    bool finalized;

    json to_json() const {
        return json{
            {"vote_id", vote_id},
            {"question", question},
            {"options", options},
            {"type", type},
            {"votes", votes},
            {"weights", weights},
            {"deadline", deadline},
            {"result", result},
            {"finalized", finalized}
        };
    }
};

struct agent_info {
    std::string agent_id;
    std::string role;
    int slot_id;
    std::vector<std::string> capabilities;
    agent_state state;
    std::string current_task_id;
    int64_t created_at;
    int64_t last_activity;
    json config;

    json to_json() const {
        return json{
            {"agent_id", agent_id},
            {"role", role},
            {"slot_id", slot_id},
            {"capabilities", capabilities},
            {"state", state},
            {"current_task_id", current_task_id},
            {"created_at", created_at},
            {"last_activity", last_activity},
            {"config", config}
        };
    }
};

// ============================================================================
// Knowledge Base
// ============================================================================

class knowledge_base {
private:
    std::unordered_map<std::string, std::vector<knowledge_entry>> entries;
    std::unordered_map<std::string, std::unordered_set<std::string>> subscribers;
    mutable std::shared_mutex mutex;

    std::function<void(const std::string&, const knowledge_entry&)> on_update_callback;

public:
    knowledge_base() = default;

    // Store knowledge
    void put(const std::string & key, const std::string & value,
             const std::string & contributor_id, const std::vector<std::string> & tags = {});

    // Retrieve latest version
    bool get(const std::string & key, knowledge_entry & entry) const;

    // Get all versions
    std::vector<knowledge_entry> get_history(const std::string & key) const;

    // Query by tags
    std::vector<knowledge_entry> query(const std::vector<std::string> & tags) const;

    // Subscribe to updates
    void subscribe(const std::string & key, const std::string & agent_id);
    void unsubscribe(const std::string & key, const std::string & agent_id);

    // Update callback
    void set_update_callback(std::function<void(const std::string&, const knowledge_entry&)> callback) {
        on_update_callback = callback;
    }

    // Get all keys
    std::vector<std::string> get_all_keys() const;

    // Clear all entries
    void clear();

    // Export/import for persistence
    json to_json() const;
    void from_json(const json & j);
};

// ============================================================================
// Message Queue
// ============================================================================

class message_queue {
private:
    std::deque<agent_message> messages;
    std::unordered_map<std::string, std::deque<agent_message>> agent_mailboxes;
    mutable std::mutex mutex;
    std::condition_variable cv;

    size_t max_queue_size = 10000;
    int64_t message_retention_ms = 86400000;  // 24 hours

public:
    message_queue() = default;

    // Send message
    void send(const agent_message & msg);

    // Receive messages for agent (non-blocking)
    std::vector<agent_message> receive(const std::string & agent_id, size_t max_count = 100);

    // Receive with timeout (blocking)
    std::vector<agent_message> receive_wait(const std::string & agent_id,
                                            int timeout_ms = 1000,
                                            size_t max_count = 100);

    // Broadcast to all agents
    void broadcast(const agent_message & msg, const std::vector<std::string> & agent_ids);

    // Get message count for agent
    size_t get_count(const std::string & agent_id) const;

    // Clean old messages
    void cleanup_old_messages();
};

// ============================================================================
// Task Scheduler
// ============================================================================

class task_scheduler {
private:
    struct task_compare {
        bool operator()(const agent_task & a, const agent_task & b) const {
            return a.priority < b.priority;  // Higher priority first
        }
    };

    std::vector<agent_task> task_queue;
    std::unordered_map<std::string, agent_task> task_map;
    std::unordered_map<std::string, task_result> results;
    mutable std::mutex mutex;
    std::condition_variable cv;

    // Dependency graph
    std::unordered_map<std::string, std::vector<std::string>> dependencies;
    std::unordered_map<std::string, std::unordered_set<std::string>> dependents;

public:
    task_scheduler() = default;

    // Submit task
    void submit(const agent_task & task);

    // Get next available task for agent with role
    bool get_next_task(const std::vector<std::string> & agent_roles, agent_task & task);

    // Update task status
    void update_status(const std::string & task_id, task_status status,
                      const std::string & agent_id = "");

    // Complete task with result
    void complete_task(const std::string & task_id, const task_result & result);

    // Fail task
    void fail_task(const std::string & task_id, const std::string & error);

    // Get task status
    bool get_task(const std::string & task_id, agent_task & task) const;

    // Get task result
    bool get_result(const std::string & task_id, task_result & result) const;

    // Cancel task
    void cancel_task(const std::string & task_id);

    // Get pending task count
    size_t get_pending_count() const;

    // Get all tasks
    std::vector<agent_task> get_all_tasks() const;

private:
    bool can_execute(const agent_task & task) const;
    void notify_dependents(const std::string & task_id);
};

// ============================================================================
// Consensus Manager
// ============================================================================

class consensus_manager {
private:
    std::unordered_map<std::string, consensus_vote> votes;
    mutable std::mutex mutex;

    std::function<void(const std::string&, const consensus_vote&)> on_finalize_callback;

public:
    consensus_manager() = default;

    // Create vote
    std::string create_vote(const std::string & question,
                           const std::vector<std::string> & options,
                           consensus_type type,
                           int64_t deadline_ms = 0);

    // Cast vote
    bool cast_vote(const std::string & vote_id,
                   const std::string & agent_id,
                   const std::string & option,
                   float weight = 1.0f);

    // Get vote status
    bool get_vote(const std::string & vote_id, consensus_vote & vote) const;

    // Check if finalized
    bool is_finalized(const std::string & vote_id) const;

    // Finalize vote (calculate result)
    bool finalize_vote(const std::string & vote_id, const std::vector<std::string> & eligible_agents);

    // Set finalize callback
    void set_finalize_callback(std::function<void(const std::string&, const consensus_vote&)> callback) {
        on_finalize_callback = callback;
    }

    // Get all votes
    std::vector<consensus_vote> get_all_votes() const;

private:
    std::string calculate_result(const consensus_vote & vote) const;
};

// ============================================================================
// Agent Registry
// ============================================================================

class agent_registry {
private:
    std::unordered_map<std::string, agent_info> agents;
    std::unordered_map<int, std::string> slot_to_agent;  // slot_id -> agent_id
    mutable std::shared_mutex mutex;

public:
    agent_registry() = default;

    // Register agent
    bool register_agent(const agent_info & agent);

    // Unregister agent
    bool unregister_agent(const std::string & agent_id);

    // Get agent info
    bool get_agent(const std::string & agent_id, agent_info & agent) const;

    // Update agent state
    bool update_state(const std::string & agent_id, agent_state state);

    // Update current task
    bool update_current_task(const std::string & agent_id, const std::string & task_id);

    // Get agents by role
    std::vector<agent_info> get_agents_by_role(const std::string & role) const;

    // Get agents by state
    std::vector<agent_info> get_agents_by_state(agent_state state) const;

    // Get all agents
    std::vector<agent_info> get_all_agents() const;

    // Get agent by slot
    bool get_agent_by_slot(int slot_id, agent_info & agent) const;

    // Check if slot is used by agent
    bool is_slot_agent(int slot_id) const;
};

// ============================================================================
// Agent Orchestrator (Main Controller)
// ============================================================================

class agent_orchestrator {
private:
    knowledge_base kb;
    message_queue msg_queue;
    task_scheduler scheduler;
    consensus_manager consensus;
    agent_registry registry;

    std::atomic<bool> running;
    std::thread worker_thread;

    // Configuration
    int max_agents = 10;
    int default_agent_timeout = 300000;  // 5 minutes

    // Callbacks
    std::function<void(const agent_message&)> on_message_callback;
    std::function<void(const std::string&, const task_result&)> on_task_complete_callback;

public:
    agent_orchestrator() : running(false) {}
    ~agent_orchestrator();

    // Lifecycle
    void start();
    void stop();
    bool is_running() const { return running.load(); }

    // Agent management
    std::string spawn_agent(const std::string & role,
                           const std::vector<std::string> & capabilities,
                           int slot_id,
                           const json & config = {});
    bool terminate_agent(const std::string & agent_id);
    std::vector<agent_info> list_agents() const;
    bool get_agent_info(const std::string & agent_id, agent_info & agent) const;

    // Task management
    std::string submit_task(const agent_task & task);
    bool get_task_status(const std::string & task_id, agent_task & task) const;
    bool get_task_result(const std::string & task_id, task_result & result) const;
    bool cancel_task(const std::string & task_id);
    std::vector<agent_task> list_tasks() const;

    // Message passing
    void send_message(const agent_message & msg);
    std::vector<agent_message> receive_messages(const std::string & agent_id, size_t max_count = 100);
    void broadcast_message(const agent_message & msg);

    // Knowledge base access
    void store_knowledge(const std::string & key, const std::string & value,
                        const std::string & agent_id, const std::vector<std::string> & tags = {});
    bool retrieve_knowledge(const std::string & key, knowledge_entry & entry) const;
    std::vector<knowledge_entry> query_knowledge(const std::vector<std::string> & tags) const;

    // Consensus
    std::string create_vote(const std::string & question,
                           const std::vector<std::string> & options,
                           consensus_type type,
                           int64_t deadline_ms = 0);
    bool cast_vote(const std::string & vote_id, const std::string & agent_id,
                   const std::string & option, float weight = 1.0f);
    bool get_vote_result(const std::string & vote_id, consensus_vote & vote) const;

    // Callbacks
    void set_message_callback(std::function<void(const agent_message&)> callback) {
        on_message_callback = callback;
    }
    void set_task_complete_callback(std::function<void(const std::string&, const task_result&)> callback) {
        on_task_complete_callback = callback;
    }

    // Configuration
    void set_max_agents(int max) { max_agents = max; }
    int get_max_agents() const { return max_agents; }

    // Stats
    json get_stats() const;

private:
    void worker_loop();
    std::string generate_id(const std::string & prefix) const;
    int64_t current_timestamp() const;
};

// ============================================================================
// Utility Functions
// ============================================================================

std::string agent_task_type_to_str(agent_task_type type);
std::string agent_state_to_str(agent_state state);
std::string message_type_to_str(message_type type);
std::string consensus_type_to_str(consensus_type type);
std::string task_status_to_str(task_status status);

agent_task_type str_to_agent_task_type(const std::string & str);
agent_state str_to_agent_state(const std::string & str);
message_type str_to_message_type(const std::string & str);
consensus_type str_to_consensus_type(const std::string & str);
task_status str_to_task_status(const std::string & str);

}  // namespace agent_collab
