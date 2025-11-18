// agent_memory.h
#pragma once

#include "agent_types.h"
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <cstring>

// Abstract memory interface (similar to llama_memory_i)
class agent_memory_i {
public:
    virtual ~agent_memory_i() = default;

    // Store a message
    virtual void store(const message & msg) = 0;

    // Retrieve relevant context
    virtual std::vector<message> retrieve_all() const = 0;
    virtual std::vector<message> retrieve_recent(size_t n) const = 0;

    // Clear memory
    virtual void clear() = 0;

    // State persistence
    virtual size_t save_state(std::ostream & out) const = 0;
    virtual size_t load_state(std::istream & in) = 0;

    // Statistics
    virtual size_t size() const = 0;
    virtual bool is_full() const = 0;
};

// Simple buffer memory implementation
class buffer_memory : public agent_memory_i {
public:
    explicit buffer_memory(size_t max_size) : max_size(max_size) {}

    void store(const message & msg) override {
        messages.push_back(msg);
        if (messages.size() > max_size) {
            messages.pop_front();
        }
    }

    std::vector<message> retrieve_all() const override {
        return std::vector<message>(messages.begin(), messages.end());
    }

    std::vector<message> retrieve_recent(size_t n) const override {
        std::vector<message> result;
        size_t start = messages.size() > n ? messages.size() - n : 0;
        for (size_t i = start; i < messages.size(); ++i) {
            result.push_back(messages[i]);
        }
        return result;
    }

    void clear() override {
        messages.clear();
    }

    size_t save_state(std::ostream & out) const override {
        // Simple serialization
        size_t written = 0;
        size_t count = messages.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        written += sizeof(count);

        for (const auto & msg : messages) {
            // Serialize each message
            // (simplified - production code would use proper serialization)
            size_t role = static_cast<size_t>(msg.role);
            out.write(reinterpret_cast<const char*>(&role), sizeof(role));

            size_t content_size = msg.content.size();
            out.write(reinterpret_cast<const char*>(&content_size), sizeof(content_size));
            out.write(msg.content.data(), content_size);

            out.write(reinterpret_cast<const char*>(&msg.timestamp_us), sizeof(msg.timestamp_us));

            written += sizeof(role) + sizeof(content_size) + content_size + sizeof(msg.timestamp_us);
        }

        return written;
    }

    size_t load_state(std::istream & in) override {
        size_t read = 0;
        size_t count;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        read += sizeof(count);

        messages.clear();
        for (size_t i = 0; i < count; ++i) {
            message msg;

            size_t role;
            in.read(reinterpret_cast<char*>(&role), sizeof(role));
            msg.role = static_cast<message_role>(role);

            size_t content_size;
            in.read(reinterpret_cast<char*>(&content_size), sizeof(content_size));

            msg.content.resize(content_size);
            in.read(&msg.content[0], content_size);

            in.read(reinterpret_cast<char*>(&msg.timestamp_us), sizeof(msg.timestamp_us));

            messages.push_back(msg);
            read += sizeof(role) + sizeof(content_size) + content_size + sizeof(msg.timestamp_us);
        }

        return read;
    }

    size_t size() const override {
        return messages.size();
    }

    bool is_full() const override {
        return messages.size() >= max_size;
    }

private:
    std::deque<message> messages;
    size_t max_size;
};

using agent_memory_ptr = std::unique_ptr<agent_memory_i>;
