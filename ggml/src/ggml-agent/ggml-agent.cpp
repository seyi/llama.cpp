#include "ggml-agent.h"
#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

// ============================================================================
// Utility Functions
// ============================================================================

static uint64_t get_current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string ggml_agent_msg::generate_msg_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << dis(gen);
    return ss.str();
}

// ============================================================================
// Circuit Breaker Implementation
// ============================================================================

bool ggml_agent_circuit_breaker::allow_request() {
    ggml_agent_circuit_state current_state = state.load();

    if (current_state == GGML_AGENT_CIRCUIT_CLOSED) {
        return true;
    }

    if (current_state == GGML_AGENT_CIRCUIT_OPEN) {
        uint64_t now = get_current_time_ms();
        uint64_t last_fail = last_failure_time_ms.load();

        if (now - last_fail >= open_timeout_ms) {
            // Try transitioning to half-open
            ggml_agent_circuit_state expected = GGML_AGENT_CIRCUIT_OPEN;
            if (state.compare_exchange_strong(expected, GGML_AGENT_CIRCUIT_HALF_OPEN)) {
                success_count = 0;
                return true;
            }
        }
        return false;
    }

    // HALF_OPEN state - allow limited requests
    return true;
}

void ggml_agent_circuit_breaker::record_success() {
    ggml_agent_circuit_state current_state = state.load();

    if (current_state == GGML_AGENT_CIRCUIT_HALF_OPEN) {
        uint32_t successes = success_count.fetch_add(1) + 1;
        if (successes >= success_threshold) {
            reset();
        }
    } else if (current_state == GGML_AGENT_CIRCUIT_CLOSED) {
        // Reset failure count on success
        failure_count = 0;
    }
}

void ggml_agent_circuit_breaker::record_failure() {
    last_failure_time_ms = get_current_time_ms();
    ggml_agent_circuit_state current_state = state.load();

    if (current_state == GGML_AGENT_CIRCUIT_HALF_OPEN) {
        // Immediately go back to open
        state = GGML_AGENT_CIRCUIT_OPEN;
        failure_count = 0;
    } else if (current_state == GGML_AGENT_CIRCUIT_CLOSED) {
        uint32_t failures = failure_count.fetch_add(1) + 1;
        if (failures >= failure_threshold) {
            state = GGML_AGENT_CIRCUIT_OPEN;
        }
    }
}

void ggml_agent_circuit_breaker::reset() {
    state = GGML_AGENT_CIRCUIT_CLOSED;
    failure_count = 0;
    success_count = 0;
}

// ============================================================================
// Agent Base Implementation
// ============================================================================

ggml_agent::ggml_agent(const std::string& agent_id) : id(agent_id) {
    health.agent_id = agent_id;

    // Register default handlers
    register_handler(GGML_AGENT_MSG_HEARTBEAT,
        [this](const ggml_agent_msg& msg) { default_heartbeat_handler(msg); });
    register_handler(GGML_AGENT_MSG_SHUTDOWN,
        [this](const ggml_agent_msg& msg) { default_shutdown_handler(msg); });
}

ggml_agent::~ggml_agent() {
    if (state.load() == GGML_AGENT_STATE_RUNNING) {
        stop();
    }
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void ggml_agent::start() {
    ggml_agent_state expected = GGML_AGENT_STATE_CREATED;
    if (!state.compare_exchange_strong(expected, GGML_AGENT_STATE_STARTING)) {
        return; // Already started or starting
    }

    should_stop = false;
    worker_thread = std::thread(&ggml_agent::run, this);

    // Wait for running state
    while (state.load() != GGML_AGENT_STATE_RUNNING &&
           state.load() != GGML_AGENT_STATE_FAILED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Register with global registry
    auto self = ggml_agent_registry::instance().get_agent(id);
    if (!self) {
        // This is a bit tricky - we need a shared_ptr but we're in the object itself
        // For now, registry will create weak references where needed
    }
}

void ggml_agent::stop() {
    ggml_agent_state current = state.load();
    if (current == GGML_AGENT_STATE_STOPPED || current == GGML_AGENT_STATE_STOPPING) {
        return;
    }

    state = GGML_AGENT_STATE_STOPPING;
    should_stop = true;
    queue_cv.notify_all();
}

void ggml_agent::join() {
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void ggml_agent::send(const ggml_agent_msg& msg) {
    if (state.load() != GGML_AGENT_STATE_RUNNING) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        msg_queue.push(msg);
    }
    queue_cv.notify_one();
}

void ggml_agent::send_to(const std::string& to_id, ggml_agent_msg_type type,
                         const std::vector<uint8_t>& payload) {
    ggml_agent_msg msg(id, to_id, type, payload);
    ggml_agent_registry::instance().route_message(msg);
}

void ggml_agent::register_handler(ggml_agent_msg_type type, ggml_agent_msg_handler handler) {
    handlers[type] = handler;
}

void ggml_agent::send_heartbeat(const std::string& to_id) {
    send_to(to_id, GGML_AGENT_MSG_HEARTBEAT);
}

void ggml_agent::run() {
    state = GGML_AGENT_STATE_RUNNING;
    on_start();

    while (!should_stop.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait for messages or shutdown
        queue_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return !msg_queue.empty() || should_stop.load();
        });

        if (should_stop.load() && msg_queue.empty()) {
            break;
        }

        // Process all pending messages
        while (!msg_queue.empty()) {
            ggml_agent_msg msg = msg_queue.front();
            msg_queue.pop();
            lock.unlock();

            try {
                process_message(msg);
                health.update_heartbeat();
                circuit_breaker.record_success();
            } catch (const std::exception& e) {
                circuit_breaker.record_failure();
                if (supervisor) {
                    // Notify supervisor of failure
                    ggml_agent_msg error_msg(id, supervisor->id, GGML_AGENT_MSG_ERROR);
                    supervisor->send(error_msg);
                }
            }

            lock.lock();
        }
    }

    on_stop();
    state = GGML_AGENT_STATE_STOPPED;
}

void ggml_agent::process_message(const ggml_agent_msg& msg) {
    auto it = handlers.find(msg.type);
    if (it != handlers.end()) {
        it->second(msg);
    }
    on_message(msg);
}

void ggml_agent::default_heartbeat_handler(const ggml_agent_msg& msg) {
    // Send heartbeat acknowledgment
    send_to(msg.from_id, GGML_AGENT_MSG_HEARTBEAT_ACK);
}

void ggml_agent::default_shutdown_handler(const ggml_agent_msg& /* msg */) {
    should_stop = true;
}

// ============================================================================
// Agent Registry Implementation
// ============================================================================

void ggml_agent_registry::register_agent(std::shared_ptr<ggml_agent> agent) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    agents[agent->id] = agent;
}

void ggml_agent_registry::unregister_agent(const std::string& id) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    agents.erase(id);
}

std::shared_ptr<ggml_agent> ggml_agent_registry::get_agent(const std::string& id) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = agents.find(id);
    return (it != agents.end()) ? it->second : nullptr;
}

std::vector<std::string> ggml_agent_registry::list_agents() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    std::vector<std::string> result;
    result.reserve(agents.size());
    for (const auto& pair : agents) {
        result.push_back(pair.first);
    }
    return result;
}

bool ggml_agent_registry::route_message(const ggml_agent_msg& msg) {
    auto target = get_agent(msg.to_id);
    if (target) {
        target->send(msg);
        return true;
    }
    return false;
}

void ggml_agent_registry::broadcast(const ggml_agent_msg& msg, const std::string& except_id) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (const auto& pair : agents) {
        if (pair.first != except_id && pair.first != msg.from_id) {
            ggml_agent_msg broadcast_msg = msg;
            broadcast_msg.to_id = pair.first;
            pair.second->send(broadcast_msg);
        }
    }
}

// ============================================================================
// Supervisor Implementation
// ============================================================================

ggml_agent_supervisor::ggml_agent_supervisor(const std::string& id)
    : ggml_agent(id) {
}

ggml_agent_supervisor::~ggml_agent_supervisor() {
    should_stop = true;
    if (health_monitor_thread.joinable()) {
        health_monitor_thread.join();
    }
}

void ggml_agent_supervisor::start() {
    ggml_agent::start();

    // Start health monitoring thread
    health_monitor_thread = std::thread(&ggml_agent_supervisor::monitor_health, this);

    // Start all children
    std::lock_guard<std::mutex> lock(children_mutex);
    for (auto& child : children) {
        child->supervisor = this;
        child->start();
    }
}

void ggml_agent_supervisor::stop() {
    // Stop all children first
    {
        std::lock_guard<std::mutex> lock(children_mutex);
        for (auto& child : children) {
            child->stop();
        }
    }

    ggml_agent::stop();

    if (health_monitor_thread.joinable()) {
        health_monitor_thread.join();
    }
}

void ggml_agent_supervisor::add_child(std::shared_ptr<ggml_agent> child) {
    std::lock_guard<std::mutex> lock(children_mutex);
    child->supervisor = this;
    children.push_back(child);

    if (state.load() == GGML_AGENT_STATE_RUNNING) {
        child->start();
    }
}

void ggml_agent_supervisor::remove_child(const std::string& child_id) {
    std::lock_guard<std::mutex> lock(children_mutex);
    children.erase(
        std::remove_if(children.begin(), children.end(),
            [&child_id](const std::shared_ptr<ggml_agent>& child) {
                return child->id == child_id;
            }),
        children.end());
}

void ggml_agent_supervisor::handle_child_failure(const std::string& child_id) {
    if (should_restart(child_id)) {
        switch (strategy) {
        case GGML_AGENT_RESTART_ONE_FOR_ONE:
            restart_child(child_id);
            break;
        case GGML_AGENT_RESTART_ONE_FOR_ALL:
            restart_all_children();
            break;
        case GGML_AGENT_RESTART_REST_FOR_ONE:
            // Restart failed child and all children started after it
            {
                std::lock_guard<std::mutex> lock(children_mutex);
                bool found = false;
                for (auto& child : children) {
                    if (found || child->id == child_id) {
                        found = true;
                        child->stop();
                        child->join();
                        child->start();
                    }
                }
            }
            break;
        }
    }
}

bool ggml_agent_supervisor::should_restart(const std::string& child_id) {
    uint64_t now = get_current_time_ms();
    auto& history = restart_history[child_id];

    // Remove old restart timestamps outside the window
    history.erase(
        std::remove_if(history.begin(), history.end(),
            [now, this](uint64_t timestamp) {
                return now - timestamp > max_restart_window_ms;
            }),
        history.end());

    // Check if we've exceeded max restarts
    if (history.size() >= max_restarts) {
        return false;
    }

    history.push_back(now);
    return true;
}

void ggml_agent_supervisor::restart_child(const std::string& child_id) {
    std::lock_guard<std::mutex> lock(children_mutex);
    for (auto& child : children) {
        if (child->id == child_id) {
            child->stop();
            child->join();
            child->start();
            break;
        }
    }
}

void ggml_agent_supervisor::restart_all_children() {
    std::lock_guard<std::mutex> lock(children_mutex);
    for (auto& child : children) {
        child->stop();
    }
    for (auto& child : children) {
        child->join();
    }
    for (auto& child : children) {
        child->start();
    }
}

void ggml_agent_supervisor::run() {
    // Handle error messages from children
    register_handler(GGML_AGENT_MSG_ERROR,
        [this](const ggml_agent_msg& msg) {
            handle_child_failure(msg.from_id);
        });

    ggml_agent::run();
}

void ggml_agent_supervisor::monitor_health() {
    while (!should_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(health_check_interval_ms));

        std::lock_guard<std::mutex> lock(children_mutex);
        for (auto& child : children) {
            // Send heartbeat
            ggml_agent_msg heartbeat(id, child->id, GGML_AGENT_MSG_HEARTBEAT);
            child->send(heartbeat);

            // Check health
            if (!child->health.check_health()) {
                child->health.is_healthy = false;
                handle_child_failure(child->id);
            }
        }
    }
}

// ============================================================================
// Coordinator Implementation
// ============================================================================

ggml_agent_coordinator::ggml_agent_coordinator(const std::string& id, size_t num_sections)
    : ggml_agent(id) {

    // Initialize sections
    sections.resize(num_sections);
    size_t section_size = 1000; // Default section size
    for (size_t i = 0; i < num_sections; ++i) {
        sections[i].start_pos = i * section_size;
        sections[i].end_pos = (i + 1) * section_size;
    }

    // Initialize empty document
    document.resize(num_sections * section_size, 0);
}

void ggml_agent_coordinator::start() {
    ggml_agent::start();
}

void ggml_agent_coordinator::on_start() {
    // Register message handlers
    register_handler(GGML_AGENT_MSG_LOCK_REQUEST,
        [this](const ggml_agent_msg& msg) { handle_lock_request(msg); });
    register_handler(GGML_AGENT_MSG_LOCK_RELEASE,
        [this](const ggml_agent_msg& msg) { handle_lock_release(msg); });
    register_handler(GGML_AGENT_MSG_DOC_EDIT,
        [this](const ggml_agent_msg& msg) { handle_doc_edit(msg); });
}

bool ggml_agent_coordinator::try_lock_section(const std::string& agent_id, size_t section_idx) {
    std::lock_guard<std::mutex> lock(doc_mutex);

    if (section_idx >= sections.size()) {
        return false;
    }

    if (!sections[section_idx].is_locked()) {
        sections[section_idx].locked_by = agent_id;
        agent_locks[agent_id].push_back(section_idx);
        return true;
    }

    return false;
}

bool ggml_agent_coordinator::release_section(const std::string& agent_id, size_t section_idx) {
    std::lock_guard<std::mutex> lock(doc_mutex);

    if (section_idx >= sections.size()) {
        return false;
    }

    if (sections[section_idx].locked_by == agent_id) {
        sections[section_idx].locked_by.clear();

        auto& locks = agent_locks[agent_id];
        locks.erase(std::remove(locks.begin(), locks.end(), section_idx), locks.end());
        return true;
    }

    return false;
}

void ggml_agent_coordinator::apply_edit(const std::string& agent_id, size_t section_idx,
                                       const std::vector<uint8_t>& new_content) {
    std::lock_guard<std::mutex> lock(doc_mutex);

    if (section_idx >= sections.size()) {
        return;
    }

    // Verify agent has lock
    if (sections[section_idx].locked_by != agent_id) {
        return;
    }

    // Apply edit to document
    size_t start = sections[section_idx].start_pos;
    size_t end = sections[section_idx].end_pos;
    size_t copy_size = std::min(new_content.size(), end - start);

    std::copy_n(new_content.begin(), copy_size, document.begin() + start);

    // Broadcast update
    broadcast_update(section_idx);
}

void ggml_agent_coordinator::broadcast_update(size_t section_idx) {
    std::vector<uint8_t> payload;
    // Encode section_idx in payload
    payload.resize(sizeof(size_t));
    std::memcpy(payload.data(), &section_idx, sizeof(size_t));

    ggml_agent_msg update_msg(id, "", GGML_AGENT_MSG_DOC_UPDATE, payload);
    ggml_agent_registry::instance().broadcast(update_msg, id);
}

void ggml_agent_coordinator::handle_lock_request(const ggml_agent_msg& msg) {
    // Decode section index from payload
    if (msg.payload.size() < sizeof(size_t)) {
        return;
    }

    size_t section_idx;
    std::memcpy(&section_idx, msg.payload.data(), sizeof(size_t));

    bool acquired = try_lock_section(msg.from_id, section_idx);

    ggml_agent_msg_type response_type = acquired ?
        GGML_AGENT_MSG_LOCK_ACQUIRED : GGML_AGENT_MSG_LOCK_DENIED;

    send_to(msg.from_id, response_type, msg.payload);
}

void ggml_agent_coordinator::handle_lock_release(const ggml_agent_msg& msg) {
    if (msg.payload.size() < sizeof(size_t)) {
        return;
    }

    size_t section_idx;
    std::memcpy(&section_idx, msg.payload.data(), sizeof(size_t));

    release_section(msg.from_id, section_idx);
}

void ggml_agent_coordinator::handle_doc_edit(const ggml_agent_msg& msg) {
    if (msg.payload.size() < sizeof(size_t)) {
        return;
    }

    size_t section_idx;
    std::memcpy(&section_idx, msg.payload.data(), sizeof(size_t));

    // Extract content (rest of payload after section_idx)
    std::vector<uint8_t> content(
        msg.payload.begin() + sizeof(size_t),
        msg.payload.end());

    apply_edit(msg.from_id, section_idx, content);
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

struct ggml_agent* ggml_agent_create(const char* id) {
    return new ggml_agent(id);
}

void ggml_agent_free(struct ggml_agent* agent) {
    delete agent;
}

void ggml_agent_start(struct ggml_agent* agent) {
    agent->start();
}

void ggml_agent_stop(struct ggml_agent* agent) {
    agent->stop();
}

void ggml_agent_send_msg(struct ggml_agent* agent, const struct ggml_agent_msg* msg) {
    agent->send(*msg);
}

struct ggml_agent_supervisor* ggml_agent_supervisor_create(const char* id) {
    return new ggml_agent_supervisor(id);
}

void ggml_agent_supervisor_add_child(struct ggml_agent_supervisor* supervisor,
                                     struct ggml_agent* child) {
    supervisor->add_child(std::shared_ptr<ggml_agent>(child));
}

struct ggml_agent_coordinator* ggml_agent_coordinator_create(const char* id, size_t num_sections) {
    return new ggml_agent_coordinator(id, num_sections);
}

} // extern "C"
