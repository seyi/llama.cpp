#include "registry.h"
#include "../../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>

using json = nlohmann::json;

namespace agent {

// registry_stats implementation
std::string agent_registry::registry_stats::to_json() const {
    json j;
    j["total_agents"] = total_agents;
    j["active_agents"] = active_agents;
    j["busy_agents"] = busy_agents;
    j["error_agents"] = error_agents;
    j["offline_agents"] = offline_agents;
    j["total_messages"] = total_messages;
    j["total_requests"] = total_requests;
    j["total_failures"] = total_failures;

    json agent_stats_json = json::object();
    for (const auto& [id, stats] : agent_stats_map) {
        agent_stats_json[id] = json::parse(stats.to_json());
    }
    j["agent_stats"] = agent_stats_json;

    return j.dump();
}

// agent_registry implementation
struct agent_registry::impl {
    std::map<std::string, std::unique_ptr<agent_interface>> agents;
    conversation_memory* memory;
    message_queue* msg_queue;
    failure_manager* failure_mgr;
    std::mutex mutex;
    std::atomic<bool> async_mode{false};
    std::atomic<bool> processor_running{false};
    std::thread processor_thread;
    message_handler msg_handler;

    // Statistics
    std::atomic<int64_t> total_messages{0};
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> total_failures{0};

    impl() : memory(nullptr), msg_queue(nullptr), failure_mgr(nullptr) {}

    ~impl() {
        processor_running = false;
        if (processor_thread.joinable()) {
            processor_thread.join();
        }
    }
};

agent_registry& agent_registry::instance() {
    static agent_registry registry;
    return registry;
}

agent_registry::agent_registry() : pimpl(std::make_unique<impl>()) {}

agent_registry::~agent_registry() = default;

bool agent_registry::register_agent(std::unique_ptr<agent_interface> agent) {
    if (!agent) return false;

    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto info = agent->get_info();
    pimpl->agents[info.id] = std::move(agent);

    return true;
}

bool agent_registry::unregister_agent(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto it = pimpl->agents.find(agent_id);
    if (it == pimpl->agents.end()) {
        return false;
    }

    it->second->shutdown();
    pimpl->agents.erase(it);
    return true;
}

agent_interface* agent_registry::get_agent(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto it = pimpl->agents.find(agent_id);
    if (it == pimpl->agents.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<agent_info> agent_registry::find_agents(const agent_query& query) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    std::vector<agent_info> results;

    for (const auto& [id, agent] : pimpl->agents) {
        auto info = agent->get_info();

        // Check status
        if (info.status < query.min_status) {
            continue;
        }

        // Check capabilities
        if (!query.capabilities.empty()) {
            bool matches = true;

            if (query.require_all_capabilities) {
                // AND: must have all capabilities
                for (const auto& cap : query.capabilities) {
                    if (!info.has_capability(cap)) {
                        matches = false;
                        break;
                    }
                }
            } else {
                // OR: must have at least one capability
                matches = false;
                for (const auto& cap : query.capabilities) {
                    if (info.has_capability(cap)) {
                        matches = true;
                        break;
                    }
                }
            }

            if (!matches) continue;
        }

        // Check metadata filters
        if (!query.metadata_filters.empty()) {
            bool matches = true;
            for (const auto& [key, value] : query.metadata_filters) {
                auto it = info.metadata.find(key);
                if (it == info.metadata.end() || it->second != value) {
                    matches = false;
                    break;
                }
            }
            if (!matches) continue;
        }

        results.push_back(info);
    }

    return results;
}

std::vector<agent_info> agent_registry::list_agents() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    std::vector<agent_info> results;
    for (const auto& [id, agent] : pimpl->agents) {
        results.push_back(agent->get_info());
    }
    return results;
}

agent_response agent_registry::send_message(const agent_message& message) {
    pimpl->total_messages++;

    agent_interface* target = get_agent(message.to_agent);
    if (!target) {
        agent_response response;
        response.status = RESPONSE_STATUS_NOT_FOUND;
        response.error_type = "agent_not_found";
        response.error_message = "Agent not found: " + message.to_agent;
        return response;
    }

    auto response = target->handle_message(message);

    // Call message handler if set
    if (pimpl->msg_handler) {
        pimpl->msg_handler(message, response);
    }

    return response;
}

agent_response agent_registry::send_request(
    const std::string& agent_id,
    const agent_request& request) {

    pimpl->total_requests++;

    agent_interface* agent = get_agent(agent_id);
    if (!agent) {
        pimpl->total_failures++;
        agent_response response;
        response.status = RESPONSE_STATUS_NOT_FOUND;
        response.error_type = "agent_not_found";
        response.error_message = "Agent not found: " + agent_id;
        return response;
    }

    auto response = agent->process_request(request);

    if (response.status != RESPONSE_STATUS_SUCCESS) {
        pimpl->total_failures++;
    }

    return response;
}

agent_response agent_registry::send_request_with_policy(
    const std::string& agent_id,
    const agent_request& request,
    const failure_policy& policy) {

    agent_response last_response;

    for (int attempt = 0; attempt <= policy.max_retries; attempt++) {
        last_response = send_request(agent_id, request);

        if (last_response.status == RESPONSE_STATUS_SUCCESS) {
            return last_response;
        }

        // Record failure
        if (pimpl->failure_mgr) {
            failure_record record;
            record.agent_id = agent_id;
            record.error_message = last_response.error_message;
            record.timestamp = get_timestamp_ms();
            record.retry_count = attempt;
            pimpl->failure_mgr->record_failure(record);
        }

        // Last attempt?
        if (attempt == policy.max_retries) {
            break;
        }

        // Retry delay with exponential backoff
        int64_t delay = policy.retry_delay_ms * static_cast<int64_t>(
            std::pow(policy.backoff_multiplier, attempt)
        );

        if (delay > policy.max_retry_delay_ms) {
            delay = policy.max_retry_delay_ms;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    // Try failover agents
    if (policy.enable_failover && !policy.fallback_agents.empty()) {
        for (const auto& fallback_id : policy.fallback_agents) {
            auto response = send_request(fallback_id, request);
            if (response.status == RESPONSE_STATUS_SUCCESS) {
                return response;
            }
        }
    }

    return last_response;
}

std::vector<agent_response> agent_registry::broadcast_message(const agent_message& message) {
    std::vector<agent_response> responses;

    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (const auto& [id, agent] : pimpl->agents) {
        auto response = agent->handle_message(message);
        responses.push_back(response);
    }

    return responses;
}

agent_registry::consensus_result agent_registry::consensus_request(
    const std::vector<std::string>& agent_ids,
    const agent_request& request,
    bool synthesize) {

    consensus_result result;

    // Send request to all agents
    for (const auto& agent_id : agent_ids) {
        auto response = send_request(agent_id, request);
        result.responses.push_back(response);
    }

    // Synthesize responses if requested
    if (synthesize && !result.responses.empty()) {
        std::stringstream synthesis;
        synthesis << "=== Multi-Agent Consensus ===\n\n";

        for (size_t i = 0; i < result.responses.size(); i++) {
            const auto& resp = result.responses[i];
            synthesis << "Agent " << (i + 1);
            if (i < agent_ids.size()) {
                synthesis << " (" << agent_ids[i] << ")";
            }
            synthesis << ":\n";
            synthesis << resp.content << "\n\n";
        }

        result.synthesized_response = synthesis.str();
    }

    return result;
}

std::optional<std::string> agent_registry::route_request(const agent_request& request) {
    // Simple routing: find first agent with matching capabilities from request params
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    // Check if specific capability requested
    auto cap_it = request.params.find("capability");
    if (cap_it != request.params.end()) {
        agent_query query;
        query.capabilities = {cap_it->second};
        query.min_status = AGENT_STATUS_IDLE;

        auto agents = find_agents(query);
        if (!agents.empty()) {
            return agents[0].id;
        }
    }

    // Default: return first idle agent
    for (const auto& [id, agent] : pimpl->agents) {
        auto info = agent->get_info();
        if (info.status == AGENT_STATUS_IDLE || info.status == AGENT_STATUS_ACTIVE) {
            return id;
        }
    }

    return std::nullopt;
}

void agent_registry::health_check() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    for (const auto& [id, agent] : pimpl->agents) {
        auto info = agent->get_info();

        // Check if agent is healthy
        if (!info.is_healthy()) {
            agent->set_status(AGENT_STATUS_OFFLINE);
        }

        // Send heartbeat
        agent->heartbeat();
    }
}

agent_stats agent_registry::get_agent_stats(const std::string& agent_id) {
    agent_interface* agent = get_agent(agent_id);
    if (!agent) {
        return agent_stats{};
    }
    return agent->get_stats();
}

agent_registry::registry_stats agent_registry::get_stats() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    registry_stats stats;
    stats.total_agents = static_cast<int>(pimpl->agents.size());
    stats.active_agents = 0;
    stats.busy_agents = 0;
    stats.error_agents = 0;
    stats.offline_agents = 0;
    stats.total_messages = pimpl->total_messages;
    stats.total_requests = pimpl->total_requests;
    stats.total_failures = pimpl->total_failures;

    for (const auto& [id, agent] : pimpl->agents) {
        auto info = agent->get_info();
        switch (info.status) {
            case AGENT_STATUS_ACTIVE:
            case AGENT_STATUS_IDLE:
                stats.active_agents++;
                break;
            case AGENT_STATUS_BUSY:
                stats.busy_agents++;
                break;
            case AGENT_STATUS_ERROR:
                stats.error_agents++;
                break;
            case AGENT_STATUS_OFFLINE:
                stats.offline_agents++;
                break;
            default:
                break;
        }

        stats.agent_stats_map[id] = agent->get_stats();
    }

    return stats;
}

void agent_registry::set_conversation_memory(conversation_memory* memory) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->memory = memory;
}

conversation_memory* agent_registry::get_conversation_memory() {
    return pimpl->memory;
}

void agent_registry::set_message_queue(message_queue* queue) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->msg_queue = queue;
}

message_queue* agent_registry::get_message_queue() {
    return pimpl->msg_queue;
}

void agent_registry::set_async_mode(bool enabled) {
    pimpl->async_mode = enabled;
}

void agent_registry::start_message_processor() {
    if (pimpl->processor_running) return;
    if (!pimpl->msg_queue) return;

    pimpl->processor_running = true;
    pimpl->processor_thread = std::thread([this]() {
        while (pimpl->processor_running) {
            agent_message msg;
            if (pimpl->msg_queue->pop(msg, 1000)) {  // 1 second timeout
                auto response = send_message(msg);

                // Call message handler if set
                if (pimpl->msg_handler) {
                    pimpl->msg_handler(msg, response);
                }
            }
        }
    });
}

void agent_registry::stop_message_processor() {
    pimpl->processor_running = false;
    if (pimpl->processor_thread.joinable()) {
        pimpl->processor_thread.join();
    }
}

void agent_registry::set_message_handler(message_handler handler) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->msg_handler = handler;
}

std::optional<failure_record> agent_registry::get_last_failure(const std::string& agent_id) {
    if (!pimpl->failure_mgr) {
        return std::nullopt;
    }

    auto history = pimpl->failure_mgr->get_history(agent_id, 1);
    if (history.empty()) {
        return std::nullopt;
    }

    return history[0];
}

void agent_registry::clear_failures() {
    if (pimpl->failure_mgr) {
        pimpl->failure_mgr->clear_history();
    }
}

std::string agent_registry::export_state() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    json j;
    json agents_json = json::array();

    for (const auto& [id, agent] : pimpl->agents) {
        auto info = agent->get_info();
        agents_json.push_back(json::parse(info.to_json()));
    }

    j["agents"] = agents_json;
    j["total_messages"] = pimpl->total_messages.load();
    j["total_requests"] = pimpl->total_requests.load();
    j["total_failures"] = pimpl->total_failures.load();

    return j.dump();
}

bool agent_registry::import_state(const std::string& json_str) {
    try {
        json j = json::parse(json_str);
        // State import would require agent recreation
        // This is a placeholder for future implementation
        return false;
    } catch (...) {
        return false;
    }
}

} // namespace agent
