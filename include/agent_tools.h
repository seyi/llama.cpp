// agent_tools.h
#pragma once

#include "agent_types.h"
#include <functional>
#include <chrono>

struct tool_result {
    bool success;
    std::string output;
    std::string error;
    int64_t execution_time_us;
};

// Abstract tool executor
class tool_executor_i {
public:
    virtual ~tool_executor_i() = default;

    virtual tool_result execute(
        const std::string & tool_name,
        const std::string & arguments) = 0;

    virtual bool has_tool(const std::string & tool_name) const = 0;
    virtual std::vector<std::string> list_tools() const = 0;
};

// Simple function-based tool executor
class function_tool_executor : public tool_executor_i {
public:
    using tool_fn = std::function<tool_result(const std::string&)>;

    void register_tool(const std::string & name, tool_fn fn) {
        tools[name] = fn;
    }

    tool_result execute(const std::string & tool_name, const std::string & arguments) override {
        auto it = tools.find(tool_name);
        if (it == tools.end()) {
            return tool_result{false, "", "Tool not found: " + tool_name, 0};
        }

        auto start = std::chrono::steady_clock::now();
        tool_result result = it->second(arguments);
        auto end = std::chrono::steady_clock::now();

        result.execution_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();

        return result;
    }

    bool has_tool(const std::string & tool_name) const override {
        return tools.find(tool_name) != tools.end();
    }

    std::vector<std::string> list_tools() const override {
        std::vector<std::string> result;
        for (const auto & [name, _] : tools) {
            result.push_back(name);
        }
        return result;
    }

private:
    std::map<std::string, tool_fn> tools;
};

using tool_executor_ptr = std::unique_ptr<tool_executor_i>;
