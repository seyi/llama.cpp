#include "agent.h"
#include "../../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <mutex>
#include <atomic>

using json = nlohmann::json;

namespace agent {

// Convert agent status to string
const char* agent_status_to_string(agent_status status) {
    switch (status) {
        case AGENT_STATUS_ACTIVE: return "active";
        case AGENT_STATUS_IDLE: return "idle";
        case AGENT_STATUS_BUSY: return "busy";
        case AGENT_STATUS_ERROR: return "error";
        case AGENT_STATUS_OFFLINE: return "offline";
        case AGENT_STATUS_UNKNOWN: return "unknown";
        default: return "unknown";
    }
}

// agent_info implementation
std::string agent_info::to_json() const {
    json j;
    j["id"] = id;
    j["name"] = name;
    j["description"] = description;
    j["capabilities"] = capabilities;
    j["endpoint"] = endpoint;
    j["status"] = agent_status_to_string(status);
    j["last_heartbeat"] = last_heartbeat;
    j["created_at"] = created_at;
    j["metadata"] = metadata;
    return j.dump();
}

agent_info agent_info::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    agent_info info;
    info.id = j.value("id", "");
    info.name = j.value("name", "");
    info.description = j.value("description", "");
    info.capabilities = j.value("capabilities", std::vector<std::string>());
    info.endpoint = j.value("endpoint", "");

    std::string status_str = j.value("status", "unknown");
    if (status_str == "active") info.status = AGENT_STATUS_ACTIVE;
    else if (status_str == "idle") info.status = AGENT_STATUS_IDLE;
    else if (status_str == "busy") info.status = AGENT_STATUS_BUSY;
    else if (status_str == "error") info.status = AGENT_STATUS_ERROR;
    else if (status_str == "offline") info.status = AGENT_STATUS_OFFLINE;
    else info.status = AGENT_STATUS_UNKNOWN;

    info.last_heartbeat = j.value("last_heartbeat", get_timestamp_ms());
    info.created_at = j.value("created_at", get_timestamp_ms());
    info.metadata = j.value("metadata", std::map<std::string, std::string>());
    return info;
}

bool agent_info::has_capability(const std::string& capability) const {
    return std::find(capabilities.begin(), capabilities.end(), capability) != capabilities.end();
}

bool agent_info::is_healthy(int64_t timeout_ms) const {
    if (status == AGENT_STATUS_OFFLINE || status == AGENT_STATUS_ERROR) {
        return false;
    }
    int64_t now = get_timestamp_ms();
    return (now - last_heartbeat) < timeout_ms;
}

// agent_stats implementation
std::string agent_stats::to_json() const {
    json j;
    j["agent_id"] = agent_id;
    j["total_requests"] = total_requests;
    j["successful_requests"] = successful_requests;
    j["failed_requests"] = failed_requests;
    j["total_tokens"] = total_tokens;
    j["avg_response_time_ms"] = avg_response_time_ms;
    j["last_request_time"] = last_request_time;
    j["active_threads"] = active_threads;
    return j.dump();
}

// local_agent implementation
struct local_agent::impl {
    agent_info info;
    conversation_memory* memory;
    void* model_ctx;
    inference_callback callback;
    std::mutex mutex;

    // Statistics
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> successful_requests{0};
    std::atomic<int64_t> failed_requests{0};
    std::atomic<int64_t> total_tokens{0};
    std::atomic<int64_t> total_response_time_ms{0};
    int64_t last_request_time{0};

    impl(const agent_info& info_, conversation_memory* mem)
        : info(info_), memory(mem), model_ctx(nullptr) {}
};

local_agent::local_agent(const agent_info& info, conversation_memory* memory)
    : pimpl(std::make_unique<impl>(info, memory)) {
    pimpl->info.created_at = get_timestamp_ms();
    pimpl->info.last_heartbeat = pimpl->info.created_at;
}

local_agent::~local_agent() = default;

agent_info local_agent::get_info() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->info;
}

agent_response local_agent::process_request(const agent_request& request) {
    int64_t start_time = get_timestamp_ms();

    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->total_requests++;
    pimpl->last_request_time = start_time;
    pimpl->info.status = AGENT_STATUS_BUSY;

    agent_response response;
    response.status = RESPONSE_STATUS_SUCCESS;
    response.tokens_used = 0;

    try {
        // Reconstruct context if continuation
        agent_request full_request = request;
        if (!request.thread_id.empty() && pimpl->memory) {
            full_request = pimpl->memory->reconstruct_request(request);
        }

        // Use inference callback if set
        if (pimpl->callback) {
            std::map<std::string, std::string> params = full_request.params;
            params["max_tokens"] = std::to_string(full_request.max_tokens);
            params["temperature"] = std::to_string(full_request.temperature);

            response.content = pimpl->callback(full_request.prompt, params);
            response.tokens_used = token_estimator::estimate_tokens(response.content);
        } else {
            // No inference callback set
            response.status = RESPONSE_STATUS_ERROR;
            response.error_type = "no_inference_callback";
            response.error_message = "No inference callback set for local agent";
            pimpl->failed_requests++;
            pimpl->info.status = AGENT_STATUS_IDLE;
            return response;
        }

        // Create or update thread
        if (request.thread_id.empty() && pimpl->memory) {
            // New conversation
            response.thread_id = pimpl->memory->create_thread(pimpl->info.id, request);
            pimpl->memory->add_turn(response.thread_id, "user", request.prompt,
                                   request.files, request.images, pimpl->info.id);
            pimpl->memory->add_turn(response.thread_id, "assistant", response.content,
                                   {}, {}, pimpl->info.id);
        } else if (!request.thread_id.empty() && pimpl->memory) {
            // Continue conversation
            response.thread_id = request.thread_id;
            pimpl->memory->add_turn(response.thread_id, "user", request.prompt,
                                   request.files, request.images, pimpl->info.id);
            pimpl->memory->add_turn(response.thread_id, "assistant", response.content,
                                   {}, {}, pimpl->info.id);
        }

        pimpl->successful_requests++;
        pimpl->total_tokens += response.tokens_used;

    } catch (const std::exception& e) {
        response.status = RESPONSE_STATUS_ERROR;
        response.error_type = "inference_error";
        response.error_message = e.what();
        pimpl->failed_requests++;
    }

    int64_t end_time = get_timestamp_ms();
    pimpl->total_response_time_ms += (end_time - start_time);
    pimpl->info.status = AGENT_STATUS_IDLE;

    return response;
}

agent_response local_agent::handle_message(const agent_message& message) {
    // Parse message payload as request
    try {
        agent_request request = agent_request::from_json(message.payload);
        request.thread_id = message.thread_id;
        return process_request(request);
    } catch (const std::exception& e) {
        agent_response response;
        response.status = RESPONSE_STATUS_ERROR;
        response.error_type = "invalid_message";
        response.error_message = std::string("Failed to parse message: ") + e.what();
        return response;
    }
}

void local_agent::set_status(agent_status status) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->info.status = status;
}

void local_agent::heartbeat() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->info.last_heartbeat = get_timestamp_ms();
}

agent_stats local_agent::get_stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    agent_stats stats;
    stats.agent_id = pimpl->info.id;
    stats.total_requests = pimpl->total_requests;
    stats.successful_requests = pimpl->successful_requests;
    stats.failed_requests = pimpl->failed_requests;
    stats.total_tokens = pimpl->total_tokens;
    stats.last_request_time = pimpl->last_request_time;
    stats.active_threads = 0;

    if (pimpl->total_requests > 0) {
        stats.avg_response_time_ms = static_cast<double>(pimpl->total_response_time_ms) /
                                     static_cast<double>(pimpl->total_requests);
    } else {
        stats.avg_response_time_ms = 0.0;
    }

    if (pimpl->memory) {
        auto threads = pimpl->memory->get_agent_threads(pimpl->info.id);
        stats.active_threads = static_cast<int>(threads.size());
    }

    return stats;
}

void local_agent::shutdown() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->info.status = AGENT_STATUS_OFFLINE;
}

void local_agent::set_model_context(void* ctx) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->model_ctx = ctx;
}

void local_agent::set_inference_callback(inference_callback callback) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->callback = callback;
}

// remote_agent implementation (simplified stub)
struct remote_agent::impl {
    agent_info info;
    int64_t timeout_ms;
    int max_retries;
    int64_t retry_delay_ms;
    std::mutex mutex;

    impl(const agent_info& info_) : info(info_), timeout_ms(30000), max_retries(3), retry_delay_ms(1000) {}
};

remote_agent::remote_agent(const agent_info& info)
    : pimpl(std::make_unique<impl>(info)) {}

remote_agent::~remote_agent() = default;

agent_info remote_agent::get_info() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->info;
}

agent_response remote_agent::process_request(const agent_request& request) {
    // This is a stub - would need HTTP client implementation
    agent_response response;
    response.status = RESPONSE_STATUS_ERROR;
    response.error_type = "not_implemented";
    response.error_message = "Remote agent communication not yet implemented";
    return response;
}

agent_response remote_agent::handle_message(const agent_message& message) {
    // Stub implementation
    agent_response response;
    response.status = RESPONSE_STATUS_ERROR;
    response.error_type = "not_implemented";
    response.error_message = "Remote agent communication not yet implemented";
    return response;
}

void remote_agent::set_status(agent_status status) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->info.status = status;
}

void remote_agent::heartbeat() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->info.last_heartbeat = get_timestamp_ms();
}

agent_stats remote_agent::get_stats() const {
    agent_stats stats;
    stats.agent_id = pimpl->info.id;
    return stats;
}

void remote_agent::shutdown() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->info.status = AGENT_STATUS_OFFLINE;
}

void remote_agent::set_timeout(int64_t timeout_ms) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->timeout_ms = timeout_ms;
}

void remote_agent::set_retry_policy(int max_retries, int64_t retry_delay_ms) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->max_retries = max_retries;
    pimpl->retry_delay_ms = retry_delay_ms;
}

// agent_factory implementation
std::unique_ptr<agent_interface> agent_factory::create_local_agent(
    const std::string& name,
    const std::string& description,
    const std::vector<std::string>& capabilities,
    conversation_memory* memory) {

    agent_info info;
    info.id = generate_uuid();
    info.name = name;
    info.description = description;
    info.capabilities = capabilities;
    info.endpoint = "local";
    info.status = AGENT_STATUS_IDLE;
    info.last_heartbeat = get_timestamp_ms();
    info.created_at = info.last_heartbeat;

    return std::make_unique<local_agent>(info, memory);
}

std::unique_ptr<agent_interface> agent_factory::create_remote_agent(
    const std::string& endpoint,
    const std::string& name,
    const std::string& description,
    const std::vector<std::string>& capabilities) {

    agent_info info;
    info.id = generate_uuid();
    info.name = name.empty() ? "remote-agent" : name;
    info.description = description;
    info.capabilities = capabilities;
    info.endpoint = endpoint;
    info.status = AGENT_STATUS_UNKNOWN;
    info.last_heartbeat = get_timestamp_ms();
    info.created_at = info.last_heartbeat;

    return std::make_unique<remote_agent>(info);
}

} // namespace agent
