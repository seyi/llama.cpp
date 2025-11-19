#include "conversation.h"
#include "../../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>

using json = nlohmann::json;

namespace agent {

// Token estimation (simple word-based approximation)
// ~4 characters per token on average for English text
int token_estimator::estimate_tokens(const std::string& text) {
    if (text.empty()) return 0;
    // Simple approximation: ~4 chars per token
    return static_cast<int>(text.length() / 4);
}

int token_estimator::estimate_file_tokens(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) return 0;

    std::stringstream buffer;
    buffer << file.rdbuf();
    return estimate_tokens(buffer.str());
}

int token_estimator::estimate_turn_tokens(const conversation_turn& turn) {
    int tokens = estimate_tokens(turn.content);
    tokens += estimate_tokens(turn.role) + 10;  // Role overhead
    return tokens;
}

// conversation_turn implementation
std::string conversation_turn::to_json() const {
    json j;
    j["role"] = role;
    j["content"] = content;
    j["timestamp"] = timestamp;
    j["files"] = files;
    j["images"] = images;
    j["agent_id"] = agent_id;
    j["model"] = model;
    j["metadata"] = metadata;
    return j.dump();
}

conversation_turn conversation_turn::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    conversation_turn turn;
    turn.role = j.value("role", "user");
    turn.content = j.value("content", "");
    turn.timestamp = j.value("timestamp", get_timestamp_ms());
    turn.files = j.value("files", std::vector<std::string>());
    turn.images = j.value("images", std::vector<std::string>());
    turn.agent_id = j.value("agent_id", "");
    turn.model = j.value("model", "");
    turn.metadata = j.value("metadata", std::map<std::string, std::string>());
    return turn;
}

int conversation_turn::estimate_tokens() const {
    return token_estimator::estimate_turn_tokens(*this);
}

// conversation_thread implementation
std::string conversation_thread::to_json() const {
    json j;
    j["thread_id"] = thread_id;
    j["parent_id"] = parent_id;
    j["created_at"] = created_at;
    j["updated_at"] = updated_at;
    j["initiating_agent"] = initiating_agent;
    j["context"] = context;
    j["expires_at"] = expires_at;

    json turns_json = json::array();
    for (const auto& turn : turns) {
        turns_json.push_back(json::parse(turn.to_json()));
    }
    j["turns"] = turns_json;

    return j.dump();
}

conversation_thread conversation_thread::from_json(const std::string& json_str) {
    json j = json::parse(json_str);
    conversation_thread thread;
    thread.thread_id = j.value("thread_id", "");
    thread.parent_id = j.value("parent_id", "");
    thread.created_at = j.value("created_at", get_timestamp_ms());
    thread.updated_at = j.value("updated_at", get_timestamp_ms());
    thread.initiating_agent = j.value("initiating_agent", "");
    thread.context = j.value("context", std::map<std::string, std::string>());
    thread.expires_at = j.value("expires_at", 0LL);

    if (j.contains("turns")) {
        for (const auto& turn_json : j["turns"]) {
            thread.turns.push_back(conversation_turn::from_json(turn_json.dump()));
        }
    }

    return thread;
}

int conversation_thread::estimate_total_tokens() const {
    int total = 0;
    for (const auto& turn : turns) {
        total += turn.estimate_tokens();
    }
    return total;
}

// conversation_memory implementation
struct conversation_memory::impl {
    std::map<std::string, conversation_thread> threads;
    std::mutex mutex;
    int64_t ttl_ms;
    size_t max_threads;

    impl(int64_t ttl_hours, size_t max)
        : ttl_ms(ttl_hours * 3600 * 1000), max_threads(max) {}

    bool is_expired(const conversation_thread& thread) const {
        return get_timestamp_ms() >= thread.expires_at;
    }
};

conversation_memory::conversation_memory(int64_t ttl_hours, size_t max_threads)
    : pimpl(std::make_unique<impl>(ttl_hours, max_threads)) {}

conversation_memory::~conversation_memory() = default;

std::string conversation_memory::create_thread(
    const std::string& agent_id,
    const agent_request& initial_request) {

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    // Clean up expired threads if at capacity
    if (pimpl->threads.size() >= pimpl->max_threads) {
        cleanup_expired();
    }

    conversation_thread thread;
    thread.thread_id = generate_uuid();
    thread.parent_id = initial_request.thread_id;  // For branching
    thread.created_at = get_timestamp_ms();
    thread.updated_at = thread.created_at;
    thread.initiating_agent = agent_id;
    thread.expires_at = thread.created_at + pimpl->ttl_ms;

    // Copy initial context from request params
    thread.context = initial_request.params;

    pimpl->threads[thread.thread_id] = thread;

    return thread.thread_id;
}

bool conversation_memory::add_turn(
    const std::string& thread_id,
    const std::string& role,
    const std::string& content,
    const std::vector<std::string>& files,
    const std::vector<std::string>& images,
    const std::string& agent_id,
    const std::string& model) {

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = pimpl->threads.find(thread_id);
    if (it == pimpl->threads.end()) {
        return false;
    }

    auto& thread = it->second;

    // Check if expired
    if (pimpl->is_expired(thread)) {
        pimpl->threads.erase(it);
        return false;
    }

    conversation_turn turn;
    turn.role = role;
    turn.content = content;
    turn.timestamp = get_timestamp_ms();
    turn.files = files;
    turn.images = images;
    turn.agent_id = agent_id;
    turn.model = model;

    thread.turns.push_back(turn);
    thread.updated_at = turn.timestamp;

    return true;
}

std::optional<conversation_thread> conversation_memory::get_thread(const std::string& thread_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = pimpl->threads.find(thread_id);
    if (it == pimpl->threads.end()) {
        return std::nullopt;
    }

    auto& thread = it->second;

    // Check if expired
    if (pimpl->is_expired(thread)) {
        pimpl->threads.erase(it);
        return std::nullopt;
    }

    return thread;
}

bool conversation_memory::touch_thread(const std::string& thread_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = pimpl->threads.find(thread_id);
    if (it == pimpl->threads.end()) {
        return false;
    }

    auto& thread = it->second;
    thread.updated_at = get_timestamp_ms();
    thread.expires_at = thread.updated_at + pimpl->ttl_ms;

    return true;
}

bool conversation_memory::delete_thread(const std::string& thread_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->threads.erase(thread_id) > 0;
}

bool conversation_memory::has_thread(const std::string& thread_id) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->threads.find(thread_id) != pimpl->threads.end();
}

reconstructed_context conversation_memory::build_conversation_history(
    const std::string& thread_id,
    int max_tokens,
    bool include_files) {

    auto thread_opt = get_thread(thread_id);
    if (!thread_opt) {
        return {
            .full_context = "",
            .tokens_used = 0,
            .turns_included = 0,
            .files_included = {},
            .truncated = false
        };
    }

    auto& thread = thread_opt.value();
    std::stringstream context_stream;
    int total_tokens = 0;
    int turns_included = 0;
    std::vector<std::string> files_included;
    bool truncated = false;

    // Build context header
    context_stream << "=== Conversation Thread: " << thread_id << " ===\n";
    context_stream << "Initiated by: " << thread.initiating_agent << "\n";
    context_stream << "Created: " << thread.created_at << "\n\n";

    // Add initial context
    if (!thread.context.empty()) {
        context_stream << "Initial Context:\n";
        for (const auto& [key, value] : thread.context) {
            context_stream << "  " << key << ": " << value << "\n";
        }
        context_stream << "\n";
    }

    // Collect files (newest first priority)
    std::vector<std::string> all_files;
    for (auto it = thread.turns.rbegin(); it != thread.turns.rend(); ++it) {
        for (const auto& file : it->files) {
            if (std::find(all_files.begin(), all_files.end(), file) == all_files.end()) {
                all_files.push_back(file);
            }
        }
    }

    // Add file contents if requested
    int file_tokens = 0;
    if (include_files && !all_files.empty()) {
        context_stream << "Referenced Files:\n";
        for (const auto& file : all_files) {
            int file_token_estimate = token_estimator::estimate_file_tokens(file);

            // Check if adding this file exceeds budget
            if (max_tokens > 0 && (total_tokens + file_token_estimate) > max_tokens / 2) {
                truncated = true;
                break;  // Reserve at least half the budget for conversation
            }

            context_stream << "\n--- File: " << file << " ---\n";
            std::ifstream file_stream(file);
            if (file_stream.is_open()) {
                context_stream << file_stream.rdbuf();
                file_stream.close();
                files_included.push_back(file);
                file_tokens += file_token_estimate;
            }
            context_stream << "\n--- End File ---\n";
        }
        context_stream << "\n";
    }

    total_tokens += file_tokens;

    // Add conversation turns (newest-first collection, chronological presentation)
    context_stream << "Conversation History:\n";

    std::vector<const conversation_turn*> included_turns;

    // Collect turns from newest to oldest
    for (auto it = thread.turns.rbegin(); it != thread.turns.rend(); ++it) {
        int turn_tokens = it->estimate_tokens();

        // Check budget
        if (max_tokens > 0 && (total_tokens + turn_tokens) > max_tokens) {
            truncated = true;
            break;
        }

        included_turns.push_back(&(*it));
        total_tokens += turn_tokens;
        turns_included++;
    }

    // Present turns in chronological order
    std::reverse(included_turns.begin(), included_turns.end());

    for (const auto* turn : included_turns) {
        context_stream << "\n[" << turn->role << "]";
        if (!turn->agent_id.empty()) {
            context_stream << " (agent: " << turn->agent_id << ")";
        }
        if (!turn->model.empty()) {
            context_stream << " (model: " << turn->model << ")";
        }
        context_stream << ":\n" << turn->content << "\n";

        if (!turn->files.empty()) {
            context_stream << "  Files: ";
            for (size_t i = 0; i < turn->files.size(); ++i) {
                if (i > 0) context_stream << ", ";
                context_stream << turn->files[i];
            }
            context_stream << "\n";
        }
    }

    if (truncated) {
        context_stream << "\n[Note: Context was truncated due to token budget]\n";
    }

    return {
        .full_context = context_stream.str(),
        .tokens_used = total_tokens,
        .turns_included = turns_included,
        .files_included = files_included,
        .truncated = truncated
    };
}

agent_request conversation_memory::reconstruct_request(const agent_request& continuation_request) {
    if (continuation_request.thread_id.empty()) {
        return continuation_request;
    }

    auto context = build_conversation_history(
        continuation_request.thread_id,
        continuation_request.max_tokens > 0 ? continuation_request.max_tokens / 2 : 0
    );

    agent_request reconstructed = continuation_request;

    // Prepend conversation history to prompt
    if (!context.full_context.empty()) {
        reconstructed.prompt = context.full_context + "\n\n[Current Request]:\n" + continuation_request.prompt;
    }

    // Merge files
    for (const auto& file : context.files_included) {
        if (std::find(reconstructed.files.begin(), reconstructed.files.end(), file) == reconstructed.files.end()) {
            reconstructed.files.push_back(file);
        }
    }

    return reconstructed;
}

size_t conversation_memory::cleanup_expired() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    size_t removed = 0;
    auto it = pimpl->threads.begin();
    while (it != pimpl->threads.end()) {
        if (pimpl->is_expired(it->second)) {
            it = pimpl->threads.erase(it);
            removed++;
        } else {
            ++it;
        }
    }

    return removed;
}

size_t conversation_memory::thread_count() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->threads.size();
}

std::vector<std::string> conversation_memory::get_agent_threads(const std::string& agent_id) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    std::vector<std::string> thread_ids;
    for (const auto& [id, thread] : pimpl->threads) {
        if (thread.initiating_agent == agent_id) {
            thread_ids.push_back(id);
        }
    }

    return thread_ids;
}

std::string conversation_memory::branch_thread(
    const std::string& parent_id,
    const std::string& agent_id) {

    auto parent_opt = get_thread(parent_id);
    if (!parent_opt) {
        return "";
    }

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    conversation_thread child;
    child.thread_id = generate_uuid();
    child.parent_id = parent_id;
    child.created_at = get_timestamp_ms();
    child.updated_at = child.created_at;
    child.initiating_agent = agent_id;
    child.expires_at = child.created_at + pimpl->ttl_ms;

    // Copy context and turns from parent
    child.context = parent_opt->context;
    child.turns = parent_opt->turns;

    pimpl->threads[child.thread_id] = child;

    return child.thread_id;
}

std::string conversation_memory::export_thread(const std::string& thread_id) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    auto it = pimpl->threads.find(thread_id);
    if (it == pimpl->threads.end()) {
        return "{}";
    }

    return it->second.to_json();
}

bool conversation_memory::import_thread(const std::string& json_str) {
    try {
        auto thread = conversation_thread::from_json(json_str);

        std::lock_guard<std::mutex> lock(pimpl->mutex);
        pimpl->threads[thread.thread_id] = thread;

        return true;
    } catch (...) {
        return false;
    }
}

} // namespace agent
