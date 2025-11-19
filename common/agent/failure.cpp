#include "failure.h"
#include "message.h"
#include "../../vendor/nlohmann/json.hpp"
#include <deque>
#include <mutex>
#include <algorithm>

using json = nlohmann::json;

namespace agent {

// Convert error type to string
const char* error_type_to_string(error_type type) {
    switch (type) {
        case ERROR_TYPE_NONE: return "none";
        case ERROR_TYPE_TIMEOUT: return "timeout";
        case ERROR_TYPE_CONNECTION: return "connection";
        case ERROR_TYPE_UNAVAILABLE: return "unavailable";
        case ERROR_TYPE_OVERLOAD: return "overload";
        case ERROR_TYPE_INVALID_REQUEST: return "invalid_request";
        case ERROR_TYPE_INVALID_RESPONSE: return "invalid_response";
        case ERROR_TYPE_AUTHENTICATION: return "authentication";
        case ERROR_TYPE_AUTHORIZATION: return "authorization";
        case ERROR_TYPE_RATE_LIMIT: return "rate_limit";
        case ERROR_TYPE_CONTEXT_EXPIRED: return "context_expired";
        case ERROR_TYPE_THREAD_NOT_FOUND: return "thread_not_found";
        case ERROR_TYPE_AGENT_NOT_FOUND: return "agent_not_found";
        case ERROR_TYPE_OFFLINE: return "offline";
        case ERROR_TYPE_INTERNAL_ERROR: return "internal_error";
        case ERROR_TYPE_UNKNOWN: return "unknown";
        default: return "unknown";
    }
}

// failure_policy implementation
failure_policy failure_policy::default_policy() {
    failure_policy p;
    p.max_retries = 3;
    p.retry_delay_ms = 1000;
    p.backoff_multiplier = 2.0f;
    p.max_retry_delay_ms = 30000;
    p.timeout_ms = 30000;
    p.enable_failover = false;
    p.fallback_agents = {};
    p.log_failures = true;
    return p;
}

failure_policy failure_policy::aggressive_policy() {
    failure_policy p;
    p.max_retries = 5;
    p.retry_delay_ms = 500;
    p.backoff_multiplier = 1.5f;
    p.max_retry_delay_ms = 10000;
    p.timeout_ms = 60000;
    p.enable_failover = true;
    p.fallback_agents = {};
    p.log_failures = true;
    return p;
}

failure_policy failure_policy::conservative_policy() {
    failure_policy p;
    p.max_retries = 1;
    p.retry_delay_ms = 2000;
    p.backoff_multiplier = 2.0f;
    p.max_retry_delay_ms = 60000;
    p.timeout_ms = 15000;
    p.enable_failover = false;
    p.fallback_agents = {};
    p.log_failures = true;
    return p;
}

// failure_record implementation
std::string failure_record::to_json() const {
    json j;
    j["agent_id"] = agent_id;
    j["error"] = error_type_to_string(error);
    j["error_message"] = error_message;
    j["timestamp"] = timestamp;
    j["thread_id"] = thread_id;
    j["message_id"] = message_id;
    j["retry_count"] = retry_count;
    j["recovered"] = recovered;
    j["recovery_agent"] = recovery_agent;
    return j.dump();
}

failure_record failure_record::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    failure_record record;
    record.agent_id = j.value("agent_id", "");
    record.error_message = j.value("error_message", "");
    record.timestamp = j.value("timestamp", get_timestamp_ms());
    record.thread_id = j.value("thread_id", "");
    record.message_id = j.value("message_id", "");
    record.retry_count = j.value("retry_count", 0);
    record.recovered = j.value("recovered", false);
    record.recovery_agent = j.value("recovery_agent", "");

    std::string error_str = j.value("error", "unknown");
    record.error = ERROR_TYPE_UNKNOWN;  // Default, parse if needed

    return record;
}

// circuit_breaker implementation
struct circuit_breaker::impl {
    int failure_threshold;
    int64_t timeout_ms;
    int success_threshold;
    circuit_state state;
    int failure_count;
    int success_count;
    int64_t last_failure_time;
    int64_t last_state_change;
    std::mutex mutex;

    impl(int fail_thresh, int64_t timeout, int success_thresh)
        : failure_threshold(fail_thresh),
          timeout_ms(timeout),
          success_threshold(success_thresh),
          state(CIRCUIT_CLOSED),
          failure_count(0),
          success_count(0),
          last_failure_time(0),
          last_state_change(get_timestamp_ms()) {}
};

circuit_breaker::circuit_breaker(int failure_threshold, int64_t timeout_ms, int success_threshold)
    : pimpl(std::make_unique<impl>(failure_threshold, timeout_ms, success_threshold)) {}

circuit_breaker::~circuit_breaker() = default;

void circuit_breaker::record_success() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    if (pimpl->state == CIRCUIT_HALF_OPEN) {
        pimpl->success_count++;
        if (pimpl->success_count >= pimpl->success_threshold) {
            pimpl->state = CIRCUIT_CLOSED;
            pimpl->failure_count = 0;
            pimpl->success_count = 0;
            pimpl->last_state_change = get_timestamp_ms();
        }
    } else if (pimpl->state == CIRCUIT_CLOSED) {
        pimpl->failure_count = 0;
    }
}

void circuit_breaker::record_failure() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    pimpl->last_failure_time = get_timestamp_ms();

    if (pimpl->state == CIRCUIT_CLOSED) {
        pimpl->failure_count++;
        if (pimpl->failure_count >= pimpl->failure_threshold) {
            pimpl->state = CIRCUIT_OPEN;
            pimpl->last_state_change = get_timestamp_ms();
        }
    } else if (pimpl->state == CIRCUIT_HALF_OPEN) {
        pimpl->state = CIRCUIT_OPEN;
        pimpl->success_count = 0;
        pimpl->last_state_change = get_timestamp_ms();
    }
}

bool circuit_breaker::allow_request() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    if (pimpl->state == CIRCUIT_CLOSED) {
        return true;
    }

    if (pimpl->state == CIRCUIT_OPEN) {
        int64_t now = get_timestamp_ms();
        if ((now - pimpl->last_state_change) >= pimpl->timeout_ms) {
            pimpl->state = CIRCUIT_HALF_OPEN;
            pimpl->success_count = 0;
            pimpl->last_state_change = now;
            return true;
        }
        return false;
    }

    // CIRCUIT_HALF_OPEN
    return true;
}

circuit_state circuit_breaker::get_state() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->state;
}

void circuit_breaker::reset() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->state = CIRCUIT_CLOSED;
    pimpl->failure_count = 0;
    pimpl->success_count = 0;
}

circuit_breaker::stats circuit_breaker::get_stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    stats s;
    s.state = pimpl->state;
    s.failure_count = pimpl->failure_count;
    s.success_count = pimpl->success_count;
    s.last_failure_time = pimpl->last_failure_time;
    s.last_state_change = pimpl->last_state_change;
    return s;
}

// retry_handler implementation
struct retry_handler::impl {
    failure_policy policy;
    impl(const failure_policy& p) : policy(p) {}
};

retry_handler::retry_handler(const failure_policy& policy)
    : pimpl(std::make_unique<impl>(policy)) {}

retry_handler::~retry_handler() = default;

bool retry_handler::handle_failure(const failure_record& record) {
    // Check if we should retry based on policy
    return record.retry_count < pimpl->policy.max_retries;
}

bool retry_handler::can_handle(error_type type) const {
    // Can handle most transient errors
    return type == ERROR_TYPE_TIMEOUT ||
           type == ERROR_TYPE_CONNECTION ||
           type == ERROR_TYPE_UNAVAILABLE ||
           type == ERROR_TYPE_OVERLOAD;
}

// failover_handler implementation
struct failover_handler::impl {
    std::vector<std::string> fallback_agents;
    size_t current_index;
    std::mutex mutex;

    impl(const std::vector<std::string>& agents) : fallback_agents(agents), current_index(0) {}
};

failover_handler::failover_handler(const std::vector<std::string>& fallback_agents)
    : pimpl(std::make_unique<impl>(fallback_agents)) {}

failover_handler::~failover_handler() = default;

bool failover_handler::handle_failure(const failure_record& record) {
    return !pimpl->fallback_agents.empty();
}

bool failover_handler::can_handle(error_type type) const {
    return type == ERROR_TYPE_UNAVAILABLE ||
           type == ERROR_TYPE_AGENT_NOT_FOUND ||
           type == ERROR_TYPE_OFFLINE;
}

std::string failover_handler::get_next_fallback() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    if (pimpl->fallback_agents.empty()) {
        return "";
    }

    std::string agent = pimpl->fallback_agents[pimpl->current_index];
    pimpl->current_index = (pimpl->current_index + 1) % pimpl->fallback_agents.size();
    return agent;
}

// dead_letter_queue implementation
struct dead_letter_queue::impl {
    std::deque<dead_letter> queue;
    size_t max_size;
    std::mutex mutex;

    impl(size_t max) : max_size(max) {}
};

dead_letter_queue::dead_letter_queue(size_t max_size)
    : pimpl(std::make_unique<impl>(max_size)) {}

dead_letter_queue::~dead_letter_queue() = default;

void dead_letter_queue::add_message(
    const std::string& message_id,
    const std::string& payload,
    const failure_record& failure) {

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    dead_letter letter;
    letter.message_id = message_id;
    letter.payload = payload;
    letter.failure = failure;
    letter.queued_at = get_timestamp_ms();

    pimpl->queue.push_back(letter);

    // Enforce max size
    while (pimpl->queue.size() > pimpl->max_size) {
        pimpl->queue.pop_front();
    }
}

std::vector<dead_letter_queue::dead_letter> dead_letter_queue::get_messages(int limit) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    std::vector<dead_letter> messages;
    int count = 0;

    for (const auto& letter : pimpl->queue) {
        if (limit > 0 && count >= limit) break;
        messages.push_back(letter);
        count++;
    }

    return messages;
}

bool dead_letter_queue::retry_message(const std::string& message_id) {
    // Stub: would need integration with message queue
    return false;
}

bool dead_letter_queue::remove_message(const std::string& message_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = std::find_if(pimpl->queue.begin(), pimpl->queue.end(),
        [&message_id](const dead_letter& letter) {
            return letter.message_id == message_id;
        });

    if (it != pimpl->queue.end()) {
        pimpl->queue.erase(it);
        return true;
    }

    return false;
}

size_t dead_letter_queue::size() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->queue.size();
}

void dead_letter_queue::clear() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->queue.clear();
}

// failure_manager implementation
struct failure_manager::impl {
    std::vector<std::unique_ptr<failure_handler>> handlers;
    std::map<std::string, std::deque<failure_record>> history;
    std::map<std::string, std::unique_ptr<circuit_breaker>> circuit_breakers;
    dead_letter_queue dlq;
    std::mutex mutex;

    impl() : dlq(1000) {}
};

failure_manager::failure_manager() : pimpl(std::make_unique<impl>()) {}

failure_manager::~failure_manager() = default;

void failure_manager::add_handler(std::unique_ptr<failure_handler> handler) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->handlers.push_back(std::move(handler));
}

void failure_manager::record_failure(const failure_record& record) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto& agent_history = pimpl->history[record.agent_id];
    agent_history.push_back(record);

    // Keep last 100 failures per agent
    if (agent_history.size() > 100) {
        agent_history.pop_front();
    }

    // Update circuit breaker
    auto cb_it = pimpl->circuit_breakers.find(record.agent_id);
    if (cb_it == pimpl->circuit_breakers.end()) {
        pimpl->circuit_breakers[record.agent_id] = std::make_unique<circuit_breaker>();
        cb_it = pimpl->circuit_breakers.find(record.agent_id);
    }

    cb_it->second->record_failure();
}

bool failure_manager::handle_failure(failure_record& record) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    for (const auto& handler : pimpl->handlers) {
        if (handler->can_handle(record.error) && handler->handle_failure(record)) {
            record.recovered = true;
            return true;
        }
    }

    return false;
}

std::vector<failure_record> failure_manager::get_history(
    const std::string& agent_id,
    int limit) {

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = pimpl->history.find(agent_id);
    if (it == pimpl->history.end()) {
        return {};
    }

    std::vector<failure_record> records;
    int count = 0;

    // Return most recent first
    for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit) {
        if (limit > 0 && count >= limit) break;
        records.push_back(*rit);
        count++;
    }

    return records;
}

circuit_breaker* failure_manager::get_circuit_breaker(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = pimpl->circuit_breakers.find(agent_id);
    if (it == pimpl->circuit_breakers.end()) {
        pimpl->circuit_breakers[agent_id] = std::make_unique<circuit_breaker>();
        it = pimpl->circuit_breakers.find(agent_id);
    }

    return it->second.get();
}

dead_letter_queue* failure_manager::get_dead_letter_queue() {
    return &pimpl->dlq;
}

void failure_manager::clear_history() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->history.clear();
}

failure_manager::stats failure_manager::get_stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    stats s;
    s.total_failures = 0;
    s.recovered_failures = 0;
    s.dead_letters = static_cast<int>(pimpl->dlq.size());

    for (const auto& [agent_id, history] : pimpl->history) {
        s.total_failures += history.size();
        s.failures_by_agent[agent_id] = history.size();

        for (const auto& record : history) {
            if (record.recovered) {
                s.recovered_failures++;
            }
            s.failures_by_type[record.error]++;
        }
    }

    return s;
}

} // namespace agent
