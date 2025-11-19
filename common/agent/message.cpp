#include "message.h"
#include "../../vendor/nlohmann/json.hpp"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <condition_variable>
#include <deque>

using json = nlohmann::json;

namespace agent {

// Helper function to generate UUID v4
std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";
    for (int i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (int i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 12; i++) {
        ss << dis(gen);
    }
    return ss.str();
}

// Get current timestamp in milliseconds
int64_t get_timestamp_ms() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

// Convert message type to string
const char* message_type_to_string(message_type type) {
    switch (type) {
        case MESSAGE_TYPE_REQUEST: return "request";
        case MESSAGE_TYPE_RESPONSE: return "response";
        case MESSAGE_TYPE_NOTIFICATION: return "notification";
        case MESSAGE_TYPE_ERROR: return "error";
        case MESSAGE_TYPE_HEARTBEAT: return "heartbeat";
        case MESSAGE_TYPE_BROADCAST: return "broadcast";
        default: return "unknown";
    }
}

// Convert response status to string
const char* response_status_to_string(response_status status) {
    switch (status) {
        case RESPONSE_STATUS_SUCCESS: return "success";
        case RESPONSE_STATUS_ERROR: return "error";
        case RESPONSE_STATUS_CONTINUATION_REQUIRED: return "continuation_required";
        case RESPONSE_STATUS_TIMEOUT: return "timeout";
        case RESPONSE_STATUS_NOT_FOUND: return "not_found";
        case RESPONSE_STATUS_UNAVAILABLE: return "unavailable";
        default: return "unknown";
    }
}

// agent_message serialization
std::string agent_message::to_json() const {
    json j;
    j["message_id"] = message_id;
    j["from_agent"] = from_agent;
    j["to_agent"] = to_agent;
    j["type"] = message_type_to_string(type);
    j["payload"] = payload;
    j["thread_id"] = thread_id;
    j["timestamp"] = timestamp;
    j["priority"] = priority;
    j["metadata"] = metadata;
    return j.dump();
}

agent_message agent_message::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    agent_message msg;
    msg.message_id = j.value("message_id", "");
    msg.from_agent = j.value("from_agent", "");
    msg.to_agent = j.value("to_agent", "");

    std::string type_str = j.value("type", "request");
    if (type_str == "response") msg.type = MESSAGE_TYPE_RESPONSE;
    else if (type_str == "notification") msg.type = MESSAGE_TYPE_NOTIFICATION;
    else if (type_str == "error") msg.type = MESSAGE_TYPE_ERROR;
    else if (type_str == "heartbeat") msg.type = MESSAGE_TYPE_HEARTBEAT;
    else if (type_str == "broadcast") msg.type = MESSAGE_TYPE_BROADCAST;
    else msg.type = MESSAGE_TYPE_REQUEST;

    msg.payload = j.value("payload", "");
    msg.thread_id = j.value("thread_id", "");
    msg.timestamp = j.value("timestamp", get_timestamp_ms());
    msg.priority = j.value("priority", 5);
    msg.metadata = j.value("metadata", std::map<std::string, std::string>());
    return msg;
}

// agent_request serialization
std::string agent_request::to_json() const {
    json j;
    j["prompt"] = prompt;
    j["thread_id"] = thread_id;
    j["files"] = files;
    j["images"] = images;
    j["params"] = params;
    j["max_tokens"] = max_tokens;
    j["temperature"] = temperature;
    j["system_prompt"] = system_prompt;
    return j.dump();
}

agent_request agent_request::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    agent_request req;
    req.prompt = j.value("prompt", "");
    req.thread_id = j.value("thread_id", "");
    req.files = j.value("files", std::vector<std::string>());
    req.images = j.value("images", std::vector<std::string>());
    req.params = j.value("params", std::map<std::string, std::string>());
    req.max_tokens = j.value("max_tokens", 0);
    req.temperature = j.value("temperature", 0.7f);
    req.system_prompt = j.value("system_prompt", "");
    return req;
}

// agent_response serialization
std::string agent_response::to_json() const {
    json j;
    j["status"] = response_status_to_string(status);
    j["content"] = content;
    j["thread_id"] = thread_id;
    j["tokens_used"] = tokens_used;
    j["error_message"] = error_message;
    j["error_type"] = error_type;
    j["metadata"] = metadata;
    return j.dump();
}

agent_response agent_response::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    agent_response resp;

    std::string status_str = j.value("status", "success");
    if (status_str == "error") resp.status = RESPONSE_STATUS_ERROR;
    else if (status_str == "continuation_required") resp.status = RESPONSE_STATUS_CONTINUATION_REQUIRED;
    else if (status_str == "timeout") resp.status = RESPONSE_STATUS_TIMEOUT;
    else if (status_str == "not_found") resp.status = RESPONSE_STATUS_NOT_FOUND;
    else if (status_str == "unavailable") resp.status = RESPONSE_STATUS_UNAVAILABLE;
    else resp.status = RESPONSE_STATUS_SUCCESS;

    resp.content = j.value("content", "");
    resp.thread_id = j.value("thread_id", "");
    resp.tokens_used = j.value("tokens_used", 0);
    resp.error_message = j.value("error_message", "");
    resp.error_type = j.value("error_type", "");
    resp.metadata = j.value("metadata", std::map<std::string, std::string>());
    return resp;
}

// continuation_offer serialization
std::string continuation_offer::to_json() const {
    json j;
    j["continuation_id"] = continuation_id;
    j["note"] = note;
    j["remaining_turns"] = remaining_turns;
    j["expires_at"] = expires_at;
    return j.dump();
}

continuation_offer continuation_offer::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    continuation_offer offer;
    offer.continuation_id = j.value("continuation_id", "");
    offer.note = j.value("note", "");
    offer.remaining_turns = j.value("remaining_turns", 0);
    offer.expires_at = j.value("expires_at", 0LL);
    return offer;
}

// Message queue implementation
struct message_queue::impl {
    std::deque<agent_message> queue;
    size_t max_size;
    std::mutex mutex;
    std::condition_variable cv;
    bool shutdown = false;

    impl(size_t max) : max_size(max) {}
};

message_queue::message_queue(size_t max_size) : pimpl(std::make_unique<impl>(max_size)) {}

message_queue::~message_queue() {
    pimpl->shutdown = true;
    pimpl->cv.notify_all();
}

bool message_queue::push(const agent_message& msg) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->queue.size() >= pimpl->max_size) {
        return false;
    }
    pimpl->queue.push_back(msg);
    pimpl->cv.notify_one();
    return true;
}

bool message_queue::pop(agent_message& msg, int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(pimpl->mutex);

    if (timeout_ms <= 0) {
        // Non-blocking
        if (pimpl->queue.empty()) {
            return false;
        }
    } else {
        // Blocking with timeout
        auto timeout = std::chrono::milliseconds(timeout_ms);
        if (!pimpl->cv.wait_for(lock, timeout, [this]() {
            return !pimpl->queue.empty() || pimpl->shutdown;
        })) {
            return false;
        }
    }

    if (pimpl->shutdown || pimpl->queue.empty()) {
        return false;
    }

    msg = pimpl->queue.front();
    pimpl->queue.pop_front();
    return true;
}

size_t message_queue::size() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->queue.size();
}

bool message_queue::empty() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->queue.empty();
}

void message_queue::clear() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->queue.clear();
}

} // namespace agent
