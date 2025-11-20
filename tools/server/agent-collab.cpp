#include "agent-collab.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>

namespace agent_collab {

// ============================================================================
// Utility Functions Implementation
// ============================================================================

static int64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex = "0123456789abcdef";

    std::stringstream ss;
    for (int i = 0; i < 8; i++) ss << hex[dis(gen)];
    ss << "-";
    for (int i = 0; i < 4; i++) ss << hex[dis(gen)];
    ss << "-4";
    for (int i = 0; i < 3; i++) ss << hex[dis(gen)];
    ss << "-";
    ss << hex[dis(gen) & 0x3 | 0x8];
    for (int i = 0; i < 3; i++) ss << hex[dis(gen)];
    ss << "-";
    for (int i = 0; i < 12; i++) ss << hex[dis(gen)];

    return ss.str();
}

std::string agent_task_type_to_str(agent_task_type type) {
    switch (type) {
        case AGENT_TASK_TYPE_ANALYZE:   return "analyze";
        case AGENT_TASK_TYPE_GENERATE:  return "generate";
        case AGENT_TASK_TYPE_TEST:      return "test";
        case AGENT_TASK_TYPE_REVIEW:    return "review";
        case AGENT_TASK_TYPE_REFACTOR:  return "refactor";
        case AGENT_TASK_TYPE_DOCUMENT:  return "document";
        case AGENT_TASK_TYPE_CONSENSUS: return "consensus";
        case AGENT_TASK_TYPE_CUSTOM:    return "custom";
        default:                        return "unknown";
    }
}

std::string agent_state_to_str(agent_state state) {
    switch (state) {
        case AGENT_STATE_INITIALIZING: return "initializing";
        case AGENT_STATE_IDLE:         return "idle";
        case AGENT_STATE_ASSIGNED:     return "assigned";
        case AGENT_STATE_EXECUTING:    return "executing";
        case AGENT_STATE_WAITING:      return "waiting";
        case AGENT_STATE_REPORTING:    return "reporting";
        case AGENT_STATE_FAILED:       return "failed";
        case AGENT_STATE_RECOVERING:   return "recovering";
        case AGENT_STATE_TERMINATED:   return "terminated";
        default:                       return "unknown";
    }
}

std::string message_type_to_str(message_type type) {
    switch (type) {
        case MSG_TYPE_REQUEST:    return "request";
        case MSG_TYPE_RESPONSE:   return "response";
        case MSG_TYPE_BROADCAST:  return "broadcast";
        case MSG_TYPE_DIRECT:     return "direct";
        case MSG_TYPE_EVENT:      return "event";
        case MSG_TYPE_CONSENSUS:  return "consensus";
        default:                  return "unknown";
    }
}

std::string consensus_type_to_str(consensus_type type) {
    switch (type) {
        case CONSENSUS_SIMPLE_MAJORITY:  return "simple_majority";
        case CONSENSUS_SUPERMAJORITY:    return "supermajority";
        case CONSENSUS_UNANIMOUS:        return "unanimous";
        case CONSENSUS_WEIGHTED:         return "weighted";
        default:                         return "unknown";
    }
}

std::string task_status_to_str(task_status status) {
    switch (status) {
        case TASK_STATUS_PENDING:    return "pending";
        case TASK_STATUS_ASSIGNED:   return "assigned";
        case TASK_STATUS_EXECUTING:  return "executing";
        case TASK_STATUS_COMPLETED:  return "completed";
        case TASK_STATUS_FAILED:     return "failed";
        case TASK_STATUS_CANCELLED:  return "cancelled";
        default:                     return "unknown";
    }
}

agent_task_type str_to_agent_task_type(const std::string & str) {
    if (str == "analyze")   return AGENT_TASK_TYPE_ANALYZE;
    if (str == "generate")  return AGENT_TASK_TYPE_GENERATE;
    if (str == "test")      return AGENT_TASK_TYPE_TEST;
    if (str == "review")    return AGENT_TASK_TYPE_REVIEW;
    if (str == "refactor")  return AGENT_TASK_TYPE_REFACTOR;
    if (str == "document")  return AGENT_TASK_TYPE_DOCUMENT;
    if (str == "consensus") return AGENT_TASK_TYPE_CONSENSUS;
    return AGENT_TASK_TYPE_CUSTOM;
}

agent_state str_to_agent_state(const std::string & str) {
    if (str == "initializing") return AGENT_STATE_INITIALIZING;
    if (str == "idle")         return AGENT_STATE_IDLE;
    if (str == "assigned")     return AGENT_STATE_ASSIGNED;
    if (str == "executing")    return AGENT_STATE_EXECUTING;
    if (str == "waiting")      return AGENT_STATE_WAITING;
    if (str == "reporting")    return AGENT_STATE_REPORTING;
    if (str == "failed")       return AGENT_STATE_FAILED;
    if (str == "recovering")   return AGENT_STATE_RECOVERING;
    if (str == "terminated")   return AGENT_STATE_TERMINATED;
    return AGENT_STATE_IDLE;
}

message_type str_to_message_type(const std::string & str) {
    if (str == "request")    return MSG_TYPE_REQUEST;
    if (str == "response")   return MSG_TYPE_RESPONSE;
    if (str == "broadcast")  return MSG_TYPE_BROADCAST;
    if (str == "direct")     return MSG_TYPE_DIRECT;
    if (str == "event")      return MSG_TYPE_EVENT;
    if (str == "consensus")  return MSG_TYPE_CONSENSUS;
    return MSG_TYPE_DIRECT;
}

consensus_type str_to_consensus_type(const std::string & str) {
    if (str == "simple_majority") return CONSENSUS_SIMPLE_MAJORITY;
    if (str == "supermajority")   return CONSENSUS_SUPERMAJORITY;
    if (str == "unanimous")       return CONSENSUS_UNANIMOUS;
    if (str == "weighted")        return CONSENSUS_WEIGHTED;
    return CONSENSUS_SIMPLE_MAJORITY;
}

task_status str_to_task_status(const std::string & str) {
    if (str == "pending")    return TASK_STATUS_PENDING;
    if (str == "assigned")   return TASK_STATUS_ASSIGNED;
    if (str == "executing")  return TASK_STATUS_EXECUTING;
    if (str == "completed")  return TASK_STATUS_COMPLETED;
    if (str == "failed")     return TASK_STATUS_FAILED;
    if (str == "cancelled")  return TASK_STATUS_CANCELLED;
    return TASK_STATUS_PENDING;
}

// ============================================================================
// Knowledge Base Implementation
// ============================================================================

void knowledge_base::put(const std::string & key, const std::string & value,
                        const std::string & contributor_id,
                        const std::vector<std::string> & tags) {
    std::unique_lock<std::shared_mutex> lock(mutex);

    knowledge_entry entry;
    entry.key = key;
    entry.value = value;
    entry.contributor_id = contributor_id;
    entry.timestamp = get_timestamp_ms();
    entry.tags = tags;

    // Get version number
    if (entries.find(key) != entries.end()) {
        entry.version = entries[key].back().version + 1;
    } else {
        entry.version = 1;
    }

    entries[key].push_back(entry);

    // Notify subscribers
    if (on_update_callback && subscribers.find(key) != subscribers.end()) {
        for (const auto & agent_id : subscribers[key]) {
            on_update_callback(agent_id, entry);
        }
    }
}

bool knowledge_base::get(const std::string & key, knowledge_entry & entry) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    auto it = entries.find(key);
    if (it != entries.end() && !it->second.empty()) {
        entry = it->second.back();
        return true;
    }
    return false;
}

std::vector<knowledge_entry> knowledge_base::get_history(const std::string & key) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    auto it = entries.find(key);
    if (it != entries.end()) {
        return it->second;
    }
    return {};
}

std::vector<knowledge_entry> knowledge_base::query(const std::vector<std::string> & tags) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    std::vector<knowledge_entry> results;

    for (const auto & [key, versions] : entries) {
        if (!versions.empty()) {
            const auto & latest = versions.back();

            // Check if entry has all requested tags
            bool has_all_tags = true;
            for (const auto & tag : tags) {
                if (std::find(latest.tags.begin(), latest.tags.end(), tag) == latest.tags.end()) {
                    has_all_tags = false;
                    break;
                }
            }

            if (has_all_tags) {
                results.push_back(latest);
            }
        }
    }

    return results;
}

void knowledge_base::subscribe(const std::string & key, const std::string & agent_id) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    subscribers[key].insert(agent_id);
}

void knowledge_base::unsubscribe(const std::string & key, const std::string & agent_id) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    auto it = subscribers.find(key);
    if (it != subscribers.end()) {
        it->second.erase(agent_id);
    }
}

std::vector<std::string> knowledge_base::get_all_keys() const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    std::vector<std::string> keys;
    keys.reserve(entries.size());
    for (const auto & [key, _] : entries) {
        keys.push_back(key);
    }
    return keys;
}

void knowledge_base::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    entries.clear();
    subscribers.clear();
}

json knowledge_base::to_json() const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    json j = json::array();
    for (const auto & [key, versions] : entries) {
        for (const auto & entry : versions) {
            j.push_back(entry.to_json());
        }
    }
    return j;
}

void knowledge_base::from_json(const json & j) {
    std::unique_lock<std::shared_mutex> lock(mutex);

    entries.clear();

    for (const auto & item : j) {
        knowledge_entry entry;
        entry.key = item["key"];
        entry.value = item["value"];
        entry.contributor_id = item["contributor_id"];
        entry.timestamp = item["timestamp"];
        entry.version = item["version"];
        entry.tags = item["tags"].get<std::vector<std::string>>();

        entries[entry.key].push_back(entry);
    }
}

// ============================================================================
// Message Queue Implementation
// ============================================================================

void message_queue::send(const agent_message & msg) {
    std::lock_guard<std::mutex> lock(mutex);

    messages.push_back(msg);

    // Route to agent mailbox
    if (!msg.to_agent_id.empty()) {
        agent_mailboxes[msg.to_agent_id].push_back(msg);
    }

    // Check size limits
    if (messages.size() > max_queue_size) {
        messages.pop_front();
    }

    cv.notify_all();
}

std::vector<agent_message> message_queue::receive(const std::string & agent_id, size_t max_count) {
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<agent_message> result;

    auto it = agent_mailboxes.find(agent_id);
    if (it != agent_mailboxes.end()) {
        size_t count = std::min(max_count, it->second.size());
        for (size_t i = 0; i < count; i++) {
            result.push_back(it->second.front());
            it->second.pop_front();
        }
    }

    return result;
}

std::vector<agent_message> message_queue::receive_wait(const std::string & agent_id,
                                                       int timeout_ms,
                                                       size_t max_count) {
    std::unique_lock<std::mutex> lock(mutex);

    cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, &agent_id] {
        auto it = agent_mailboxes.find(agent_id);
        return it != agent_mailboxes.end() && !it->second.empty();
    });

    std::vector<agent_message> result;

    auto it = agent_mailboxes.find(agent_id);
    if (it != agent_mailboxes.end()) {
        size_t count = std::min(max_count, it->second.size());
        for (size_t i = 0; i < count; i++) {
            result.push_back(it->second.front());
            it->second.pop_front();
        }
    }

    return result;
}

void message_queue::broadcast(const agent_message & msg, const std::vector<std::string> & agent_ids) {
    std::lock_guard<std::mutex> lock(mutex);

    for (const auto & agent_id : agent_ids) {
        agent_message copy = msg;
        copy.to_agent_id = agent_id;
        agent_mailboxes[agent_id].push_back(copy);
    }

    messages.push_back(msg);

    cv.notify_all();
}

size_t message_queue::get_count(const std::string & agent_id) const {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = agent_mailboxes.find(agent_id);
    if (it != agent_mailboxes.end()) {
        return it->second.size();
    }
    return 0;
}

void message_queue::cleanup_old_messages() {
    std::lock_guard<std::mutex> lock(mutex);

    int64_t cutoff = get_timestamp_ms() - message_retention_ms;

    // Clean main queue
    while (!messages.empty() && messages.front().timestamp < cutoff) {
        messages.pop_front();
    }

    // Clean mailboxes
    for (auto & [agent_id, mailbox] : agent_mailboxes) {
        while (!mailbox.empty() && mailbox.front().timestamp < cutoff) {
            mailbox.pop_front();
        }
    }
}

// ============================================================================
// Task Scheduler Implementation
// ============================================================================

void task_scheduler::submit(const agent_task & task) {
    std::lock_guard<std::mutex> lock(mutex);

    task_map[task.task_id] = task;

    // Add to dependency graph
    for (const auto & dep : task.dependencies) {
        dependencies[task.task_id].push_back(dep);
        dependents[dep].insert(task.task_id);
    }

    // Add to queue if no dependencies or all dependencies completed
    if (can_execute(task)) {
        task_queue.push_back(task);
        std::push_heap(task_queue.begin(), task_queue.end(), task_compare());
    }

    cv.notify_all();
}

bool task_scheduler::get_next_task(const std::vector<std::string> & agent_roles, agent_task & task) {
    std::lock_guard<std::mutex> lock(mutex);

    if (task_queue.empty()) {
        return false;
    }

    // Find task matching agent roles
    for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
        if (it->required_roles.empty()) {
            // No role requirement
            task = *it;
            task_queue.erase(it);
            std::make_heap(task_queue.begin(), task_queue.end(), task_compare());
            return true;
        }

        // Check if agent has any required role
        for (const auto & role : agent_roles) {
            if (std::find(it->required_roles.begin(), it->required_roles.end(), role)
                != it->required_roles.end()) {
                task = *it;
                task_queue.erase(it);
                std::make_heap(task_queue.begin(), task_queue.end(), task_compare());
                return true;
            }
        }
    }

    return false;
}

void task_scheduler::update_status(const std::string & task_id, task_status status,
                                   const std::string & agent_id) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = task_map.find(task_id);
    if (it != task_map.end()) {
        it->second.status = status;
        if (!agent_id.empty()) {
            it->second.assigned_agent_id = agent_id;
        }
    }
}

void task_scheduler::complete_task(const std::string & task_id, const task_result & result) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = task_map.find(task_id);
    if (it != task_map.end()) {
        it->second.status = TASK_STATUS_COMPLETED;
        results[task_id] = result;

        // Notify dependents
        notify_dependents(task_id);
    }
}

void task_scheduler::fail_task(const std::string & task_id, const std::string & error) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = task_map.find(task_id);
    if (it != task_map.end()) {
        it->second.status = TASK_STATUS_FAILED;

        task_result result;
        result.task_id = task_id;
        result.success = false;
        result.error_message = error;
        results[task_id] = result;
    }
}

bool task_scheduler::get_task(const std::string & task_id, agent_task & task) const {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = task_map.find(task_id);
    if (it != task_map.end()) {
        task = it->second;
        return true;
    }
    return false;
}

bool task_scheduler::get_result(const std::string & task_id, task_result & result) const {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = results.find(task_id);
    if (it != results.end()) {
        result = it->second;
        return true;
    }
    return false;
}

void task_scheduler::cancel_task(const std::string & task_id) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = task_map.find(task_id);
    if (it != task_map.end()) {
        it->second.status = TASK_STATUS_CANCELLED;

        // Remove from queue if present
        task_queue.erase(
            std::remove_if(task_queue.begin(), task_queue.end(),
                          [&task_id](const agent_task & t) { return t.task_id == task_id; }),
            task_queue.end());
    }
}

size_t task_scheduler::get_pending_count() const {
    std::lock_guard<std::mutex> lock(mutex);
    return task_queue.size();
}

std::vector<agent_task> task_scheduler::get_all_tasks() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<agent_task> tasks;
    tasks.reserve(task_map.size());
    for (const auto & [id, task] : task_map) {
        tasks.push_back(task);
    }
    return tasks;
}

bool task_scheduler::can_execute(const agent_task & task) const {
    // Check if all dependencies are completed
    for (const auto & dep_id : task.dependencies) {
        auto it = task_map.find(dep_id);
        if (it == task_map.end() || it->second.status != TASK_STATUS_COMPLETED) {
            return false;
        }
    }
    return true;
}

void task_scheduler::notify_dependents(const std::string & task_id) {
    auto it = dependents.find(task_id);
    if (it != dependents.end()) {
        for (const auto & dependent_id : it->second) {
            auto task_it = task_map.find(dependent_id);
            if (task_it != task_map.end() && can_execute(task_it->second)) {
                // Add to queue
                task_queue.push_back(task_it->second);
                std::push_heap(task_queue.begin(), task_queue.end(), task_compare());
            }
        }
    }
}

// ============================================================================
// Consensus Manager Implementation
// ============================================================================

std::string consensus_manager::create_vote(const std::string & question,
                                          const std::vector<std::string> & options,
                                          consensus_type type,
                                          int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex);

    std::string vote_id = "vote-" + generate_uuid();

    consensus_vote vote;
    vote.vote_id = vote_id;
    vote.question = question;
    vote.options = options;
    vote.type = type;
    vote.deadline = deadline_ms > 0 ? get_timestamp_ms() + deadline_ms : 0;
    vote.finalized = false;

    votes[vote_id] = vote;

    return vote_id;
}

bool consensus_manager::cast_vote(const std::string & vote_id,
                                 const std::string & agent_id,
                                 const std::string & option,
                                 float weight) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = votes.find(vote_id);
    if (it == votes.end() || it->second.finalized) {
        return false;
    }

    // Check if option is valid
    if (std::find(it->second.options.begin(), it->second.options.end(), option)
        == it->second.options.end()) {
        return false;
    }

    it->second.votes[agent_id] = option;
    it->second.weights[agent_id] = weight;

    return true;
}

bool consensus_manager::get_vote(const std::string & vote_id, consensus_vote & vote) const {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = votes.find(vote_id);
    if (it != votes.end()) {
        vote = it->second;
        return true;
    }
    return false;
}

bool consensus_manager::is_finalized(const std::string & vote_id) const {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = votes.find(vote_id);
    return it != votes.end() && it->second.finalized;
}

bool consensus_manager::finalize_vote(const std::string & vote_id,
                                     const std::vector<std::string> & eligible_agents) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = votes.find(vote_id);
    if (it == votes.end() || it->second.finalized) {
        return false;
    }

    it->second.result = calculate_result(it->second);
    it->second.finalized = true;

    if (on_finalize_callback) {
        on_finalize_callback(vote_id, it->second);
    }

    return true;
}

std::vector<consensus_vote> consensus_manager::get_all_votes() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<consensus_vote> all_votes;
    all_votes.reserve(votes.size());
    for (const auto & [id, vote] : votes) {
        all_votes.push_back(vote);
    }
    return all_votes;
}

std::string consensus_manager::calculate_result(const consensus_vote & vote) const {
    if (vote.votes.empty()) {
        return "";
    }

    // Count votes per option
    std::unordered_map<std::string, float> counts;
    float total_weight = 0.0f;

    for (const auto & [agent_id, option] : vote.votes) {
        float weight = 1.0f;
        if (vote.type == CONSENSUS_WEIGHTED) {
            auto w_it = vote.weights.find(agent_id);
            if (w_it != vote.weights.end()) {
                weight = w_it->second;
            }
        }
        counts[option] += weight;
        total_weight += weight;
    }

    // Find winner based on consensus type
    std::string winner;
    float max_count = 0.0f;

    for (const auto & [option, count] : counts) {
        if (count > max_count) {
            max_count = count;
            winner = option;
        }
    }

    // Check if threshold is met
    float percentage = total_weight > 0 ? (max_count / total_weight) : 0;

    switch (vote.type) {
        case CONSENSUS_SIMPLE_MAJORITY:
            return percentage > 0.5f ? winner : "";
        case CONSENSUS_SUPERMAJORITY:
            return percentage >= 0.66f ? winner : "";
        case CONSENSUS_UNANIMOUS:
            return percentage >= 1.0f ? winner : "";
        case CONSENSUS_WEIGHTED:
            return winner;  // Already handled by weighted counting
    }

    return winner;
}

// ============================================================================
// Agent Registry Implementation
// ============================================================================

bool agent_registry::register_agent(const agent_info & agent) {
    std::unique_lock<std::shared_mutex> lock(mutex);

    if (agents.find(agent.agent_id) != agents.end()) {
        return false;  // Already exists
    }

    agents[agent.agent_id] = agent;
    slot_to_agent[agent.slot_id] = agent.agent_id;

    return true;
}

bool agent_registry::unregister_agent(const std::string & agent_id) {
    std::unique_lock<std::shared_mutex> lock(mutex);

    auto it = agents.find(agent_id);
    if (it == agents.end()) {
        return false;
    }

    slot_to_agent.erase(it->second.slot_id);
    agents.erase(it);

    return true;
}

bool agent_registry::get_agent(const std::string & agent_id, agent_info & agent) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    auto it = agents.find(agent_id);
    if (it != agents.end()) {
        agent = it->second;
        return true;
    }
    return false;
}

bool agent_registry::update_state(const std::string & agent_id, agent_state state) {
    std::unique_lock<std::shared_mutex> lock(mutex);

    auto it = agents.find(agent_id);
    if (it != agents.end()) {
        it->second.state = state;
        it->second.last_activity = get_timestamp_ms();
        return true;
    }
    return false;
}

bool agent_registry::update_current_task(const std::string & agent_id, const std::string & task_id) {
    std::unique_lock<std::shared_mutex> lock(mutex);

    auto it = agents.find(agent_id);
    if (it != agents.end()) {
        it->second.current_task_id = task_id;
        it->second.last_activity = get_timestamp_ms();
        return true;
    }
    return false;
}

std::vector<agent_info> agent_registry::get_agents_by_role(const std::string & role) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    std::vector<agent_info> result;
    for (const auto & [id, agent] : agents) {
        if (agent.role == role) {
            result.push_back(agent);
        }
    }
    return result;
}

std::vector<agent_info> agent_registry::get_agents_by_state(agent_state state) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    std::vector<agent_info> result;
    for (const auto & [id, agent] : agents) {
        if (agent.state == state) {
            result.push_back(agent);
        }
    }
    return result;
}

std::vector<agent_info> agent_registry::get_all_agents() const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    std::vector<agent_info> result;
    result.reserve(agents.size());
    for (const auto & [id, agent] : agents) {
        result.push_back(agent);
    }
    return result;
}

bool agent_registry::get_agent_by_slot(int slot_id, agent_info & agent) const {
    std::shared_lock<std::shared_mutex> lock(mutex);

    auto it = slot_to_agent.find(slot_id);
    if (it != slot_to_agent.end()) {
        return get_agent(it->second, agent);
    }
    return false;
}

bool agent_registry::is_slot_agent(int slot_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return slot_to_agent.find(slot_id) != slot_to_agent.end();
}

// ============================================================================
// Agent Orchestrator Implementation
// ============================================================================

agent_orchestrator::~agent_orchestrator() {
    stop();
}

void agent_orchestrator::start() {
    if (running.load()) {
        return;
    }

    running.store(true);
    worker_thread = std::thread(&agent_orchestrator::worker_loop, this);

    LOG_INF("Agent orchestrator started\n");
}

void agent_orchestrator::stop() {
    if (!running.load()) {
        return;
    }

    running.store(false);
    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    LOG_INF("Agent orchestrator stopped\n");
}

std::string agent_orchestrator::spawn_agent(const std::string & role,
                                           const std::vector<std::string> & capabilities,
                                           int slot_id,
                                           const json & config) {
    agent_info agent;
    agent.agent_id = generate_id("agent");
    agent.role = role;
    agent.slot_id = slot_id;
    agent.capabilities = capabilities;
    agent.state = AGENT_STATE_IDLE;
    agent.created_at = current_timestamp();
    agent.last_activity = agent.created_at;
    agent.config = config;

    if (registry.register_agent(agent)) {
        LOG_INF("Agent spawned: %s (role: %s, slot: %d)\n",
                agent.agent_id.c_str(), role.c_str(), slot_id);
        return agent.agent_id;
    }

    return "";
}

bool agent_orchestrator::terminate_agent(const std::string & agent_id) {
    agent_info agent;
    if (registry.get_agent(agent_id, agent)) {
        registry.update_state(agent_id, AGENT_STATE_TERMINATED);
        // Note: Actual slot cleanup should be handled by server
        LOG_INF("Agent terminated: %s\n", agent_id.c_str());
        return true;
    }
    return false;
}

std::vector<agent_info> agent_orchestrator::list_agents() const {
    return registry.get_all_agents();
}

bool agent_orchestrator::get_agent_info(const std::string & agent_id, agent_info & agent) const {
    return registry.get_agent(agent_id, agent);
}

std::string agent_orchestrator::submit_task(const agent_task & task) {
    scheduler.submit(task);
    LOG_INF("Task submitted: %s (%s)\n",
            task.task_id.c_str(), agent_task_type_to_str(task.type).c_str());
    return task.task_id;
}

bool agent_orchestrator::get_task_status(const std::string & task_id, agent_task & task) const {
    return scheduler.get_task(task_id, task);
}

bool agent_orchestrator::get_task_result(const std::string & task_id, task_result & result) const {
    return scheduler.get_result(task_id, result);
}

bool agent_orchestrator::cancel_task(const std::string & task_id) {
    scheduler.cancel_task(task_id);
    return true;
}

std::vector<agent_task> agent_orchestrator::list_tasks() const {
    return scheduler.get_all_tasks();
}

void agent_orchestrator::send_message(const agent_message & msg) {
    msg_queue.send(msg);

    if (on_message_callback) {
        on_message_callback(msg);
    }
}

std::vector<agent_message> agent_orchestrator::receive_messages(const std::string & agent_id,
                                                                size_t max_count) {
    return msg_queue.receive(agent_id, max_count);
}

void agent_orchestrator::broadcast_message(const agent_message & msg) {
    auto agents = registry.get_all_agents();
    std::vector<std::string> agent_ids;
    for (const auto & agent : agents) {
        agent_ids.push_back(agent.agent_id);
    }
    msg_queue.broadcast(msg, agent_ids);
}

void agent_orchestrator::store_knowledge(const std::string & key, const std::string & value,
                                        const std::string & agent_id,
                                        const std::vector<std::string> & tags) {
    kb.put(key, value, agent_id, tags);
}

bool agent_orchestrator::retrieve_knowledge(const std::string & key, knowledge_entry & entry) const {
    return kb.get(key, entry);
}

std::vector<knowledge_entry> agent_orchestrator::query_knowledge(const std::vector<std::string> & tags) const {
    return kb.query(tags);
}

std::string agent_orchestrator::create_vote(const std::string & question,
                                           const std::vector<std::string> & options,
                                           consensus_type type,
                                           int64_t deadline_ms) {
    return consensus.create_vote(question, options, type, deadline_ms);
}

bool agent_orchestrator::cast_vote(const std::string & vote_id, const std::string & agent_id,
                                  const std::string & option, float weight) {
    return consensus.cast_vote(vote_id, agent_id, option, weight);
}

bool agent_orchestrator::get_vote_result(const std::string & vote_id, consensus_vote & vote) const {
    return consensus.get_vote(vote_id, vote);
}

json agent_orchestrator::get_stats() const {
    auto agents = registry.get_all_agents();
    auto tasks = scheduler.get_all_tasks();

    int idle_agents = 0, busy_agents = 0;
    int pending_tasks = 0, completed_tasks = 0, failed_tasks = 0;

    for (const auto & agent : agents) {
        if (agent.state == AGENT_STATE_IDLE) idle_agents++;
        else if (agent.state == AGENT_STATE_EXECUTING) busy_agents++;
    }

    for (const auto & task : tasks) {
        if (task.status == TASK_STATUS_PENDING) pending_tasks++;
        else if (task.status == TASK_STATUS_COMPLETED) completed_tasks++;
        else if (task.status == TASK_STATUS_FAILED) failed_tasks++;
    }

    return json{
        {"agents", {
            {"total", agents.size()},
            {"idle", idle_agents},
            {"busy", busy_agents}
        }},
        {"tasks", {
            {"total", tasks.size()},
            {"pending", pending_tasks},
            {"completed", completed_tasks},
            {"failed", failed_tasks}
        }},
        {"knowledge_base", {
            {"entries", kb.get_all_keys().size()}
        }}
    };
}

void agent_orchestrator::worker_loop() {
    while (running.load()) {
        // Cleanup old messages periodically
        msg_queue.cleanup_old_messages();

        // Sleep for a bit
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

std::string agent_orchestrator::generate_id(const std::string & prefix) const {
    return prefix + "-" + generate_uuid();
}

int64_t agent_orchestrator::current_timestamp() const {
    return get_timestamp_ms();
}

}  // namespace agent_collab
