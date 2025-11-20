#pragma once

#include "agent-collab.h"
#include "server-http.h"

#include <sstream>

using json = nlohmann::ordered_json;

namespace agent_collab {

// ============================================================================
// Agent Collaboration API Routes
// ============================================================================

class agent_routes {
private:
    agent_orchestrator & orchestrator;

    // Helper to create error response
    static json error_response(const std::string & message, const std::string & type = "server_error") {
        return json{
            {"error", {
                {"message", message},
                {"type", type}
            }}
        };
    }

    // Helper to generate task ID
    static std::string generate_task_id() {
        static std::atomic<int> counter{0};
        return "task-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
               "-" + std::to_string(counter++);
    }

public:
    agent_routes(agent_orchestrator & orch) : orchestrator(orch) {}

    // ========================================================================
    // Agent Management Routes
    // ========================================================================

    // POST /v1/agents/spawn - Create new agent
    void route_spawn_agent(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        // Validate required fields
        if (!body.contains("role") || !body.contains("slot_id")) {
            res.status = 400;
            res.set_content(error_response("Missing required fields: role, slot_id").dump(), "application/json");
            return;
        }

        std::string role = body["role"];
        int slot_id = body["slot_id"];
        std::vector<std::string> capabilities = body.value("capabilities", std::vector<std::string>{});
        json config = body.value("config", json::object());

        std::string agent_id = orchestrator.spawn_agent(role, capabilities, slot_id, config);

        if (agent_id.empty()) {
            res.status = 500;
            res.set_content(error_response("Failed to spawn agent").dump(), "application/json");
            return;
        }

        json response = {
            {"agent_id", agent_id},
            {"role", role},
            {"slot_id", slot_id},
            {"status", "spawned"}
        };

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/agents - List all agents
    void route_list_agents(const httplib::Request & req, httplib::Response & res) {
        auto agents = orchestrator.list_agents();

        json agents_json = json::array();
        for (const auto & agent : agents) {
            agents_json.push_back(agent.to_json());
        }

        json response = {
            {"agents", agents_json},
            {"count", agents.size()}
        };

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/agents/{agent_id} - Get agent info
    void route_get_agent(const httplib::Request & req, httplib::Response & res) {
        std::string agent_id = req.path_params.at("agent_id");

        agent_info agent;
        if (!orchestrator.get_agent_info(agent_id, agent)) {
            res.status = 404;
            res.set_content(error_response("Agent not found").dump(), "application/json");
            return;
        }

        res.set_content(agent.to_json().dump(), "application/json");
    }

    // DELETE /v1/agents/{agent_id} - Terminate agent
    void route_terminate_agent(const httplib::Request & req, httplib::Response & res) {
        std::string agent_id = req.path_params.at("agent_id");

        if (!orchestrator.terminate_agent(agent_id)) {
            res.status = 404;
            res.set_content(error_response("Agent not found").dump(), "application/json");
            return;
        }

        json response = {
            {"success", true},
            {"agent_id", agent_id},
            {"status", "terminated"}
        };

        res.set_content(response.dump(), "application/json");
    }

    // ========================================================================
    // Task Management Routes
    // ========================================================================

    // POST /v1/tasks/submit - Submit new task
    void route_submit_task(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        agent_task task;
        task.task_id = generate_task_id();
        task.type = str_to_agent_task_type(body.value("type", "custom"));
        task.description = body.value("description", "");
        task.parameters = body.value("parameters", json::object());
        task.dependencies = body.value("dependencies", std::vector<std::string>{});
        task.required_roles = body.value("required_roles", std::vector<std::string>{});
        task.priority = body.value("priority", 5);
        task.parent_task_id = body.value("parent_task_id", "");
        task.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        task.deadline = body.value("deadline", 0);
        task.status = TASK_STATUS_PENDING;

        std::string task_id = orchestrator.submit_task(task);

        json response = {
            {"task_id", task_id},
            {"status", "submitted"}
        };

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/tasks/{task_id} - Get task status
    void route_get_task(const httplib::Request & req, httplib::Response & res) {
        std::string task_id = req.path_params.at("task_id");

        agent_task task;
        if (!orchestrator.get_task_status(task_id, task)) {
            res.status = 404;
            res.set_content(error_response("Task not found").dump(), "application/json");
            return;
        }

        json response = task.to_json();

        // Check if result is available
        task_result result;
        if (orchestrator.get_task_result(task_id, result)) {
            response["result"] = result.to_json();
        }

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/tasks - List all tasks
    void route_list_tasks(const httplib::Request & req, httplib::Response & res) {
        auto tasks = orchestrator.list_tasks();

        json tasks_json = json::array();
        for (const auto & task : tasks) {
            tasks_json.push_back(task.to_json());
        }

        json response = {
            {"tasks", tasks_json},
            {"count", tasks.size()}
        };

        res.set_content(response.dump(), "application/json");
    }

    // DELETE /v1/tasks/{task_id} - Cancel task
    void route_cancel_task(const httplib::Request & req, httplib::Response & res) {
        std::string task_id = req.path_params.at("task_id");

        if (!orchestrator.cancel_task(task_id)) {
            res.status = 404;
            res.set_content(error_response("Task not found").dump(), "application/json");
            return;
        }

        json response = {
            {"success", true},
            {"task_id", task_id},
            {"status", "cancelled"}
        };

        res.set_content(response.dump(), "application/json");
    }

    // POST /v1/tasks/workflow - Submit task workflow
    void route_submit_workflow(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        if (!body.contains("tasks") || !body["tasks"].is_array()) {
            res.status = 400;
            res.set_content(error_response("Missing or invalid 'tasks' array").dump(), "application/json");
            return;
        }

        std::string workflow_id = "workflow-" + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count());

        json task_ids = json::array();

        for (const auto & task_def : body["tasks"]) {
            agent_task task;
            task.task_id = task_def.contains("id") ? task_def["id"].get<std::string>() : generate_task_id();
            task.type = str_to_agent_task_type(task_def.value("type", "custom"));
            task.description = task_def.value("description", "");
            task.parameters = task_def.value("parameters", json::object());
            task.dependencies = task_def.value("dependencies", std::vector<std::string>{});
            task.required_roles = task_def.value("required_roles", std::vector<std::string>{});
            task.priority = task_def.value("priority", 5);
            task.parent_task_id = workflow_id;
            task.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            task.deadline = task_def.value("deadline", 0);
            task.status = TASK_STATUS_PENDING;

            orchestrator.submit_task(task);
            task_ids.push_back(task.task_id);
        }

        json response = {
            {"workflow_id", workflow_id},
            {"task_ids", task_ids},
            {"status", "scheduled"}
        };

        res.set_content(response.dump(), "application/json");
    }

    // ========================================================================
    // Knowledge Base Routes
    // ========================================================================

    // POST /v1/knowledge - Store knowledge
    void route_store_knowledge(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        if (!body.contains("key") || !body.contains("value")) {
            res.status = 400;
            res.set_content(error_response("Missing required fields: key, value").dump(), "application/json");
            return;
        }

        std::string key = body["key"];
        std::string value = body["value"].dump();
        std::string agent_id = body.value("agent_id", "system");
        std::vector<std::string> tags = body.value("tags", std::vector<std::string>{});

        orchestrator.store_knowledge(key, value, agent_id, tags);

        json response = {
            {"success", true},
            {"key", key}
        };

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/knowledge/{key} - Retrieve knowledge
    void route_get_knowledge(const httplib::Request & req, httplib::Response & res) {
        std::string key = req.path_params.at("key");

        knowledge_entry entry;
        if (!orchestrator.retrieve_knowledge(key, entry)) {
            res.status = 404;
            res.set_content(error_response("Knowledge entry not found").dump(), "application/json");
            return;
        }

        res.set_content(entry.to_json().dump(), "application/json");
    }

    // GET /v1/knowledge/query - Query knowledge by tags
    void route_query_knowledge(const httplib::Request & req, httplib::Response & res) {
        // Parse tags from query params
        std::vector<std::string> tags;
        if (req.has_param("tags")) {
            std::string tags_param = req.get_param_value("tags");
            std::istringstream ss(tags_param);
            std::string tag;
            while (std::getline(ss, tag, ',')) {
                tags.push_back(tag);
            }
        }

        auto entries = orchestrator.query_knowledge(tags);

        json entries_json = json::array();
        for (const auto & entry : entries) {
            entries_json.push_back(entry.to_json());
        }

        json response = {
            {"entries", entries_json},
            {"count", entries.size()}
        };

        res.set_content(response.dump(), "application/json");
    }

    // ========================================================================
    // Messaging Routes
    // ========================================================================

    // POST /v1/messages/send - Send message
    void route_send_message(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        agent_message msg;
        msg.message_id = "msg-" + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count());
        msg.from_agent_id = body.value("from_agent_id", "system");
        msg.to_agent_id = body.value("to_agent_id", "");
        msg.type = str_to_message_type(body.value("type", "direct"));
        msg.subject = body.value("subject", "");
        msg.payload = body.value("payload", json::object());
        msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        msg.conversation_id = body.value("conversation_id", "");

        orchestrator.send_message(msg);

        json response = {
            {"success", true},
            {"message_id", msg.message_id}
        };

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/messages/{agent_id} - Receive messages
    void route_receive_messages(const httplib::Request & req, httplib::Response & res) {
        std::string agent_id = req.path_params.at("agent_id");

        size_t max_count = 100;
        if (req.has_param("max_count")) {
            max_count = std::stoull(req.get_param_value("max_count"));
        }

        auto messages = orchestrator.receive_messages(agent_id, max_count);

        json messages_json = json::array();
        for (const auto & msg : messages) {
            messages_json.push_back(msg.to_json());
        }

        json response = {
            {"messages", messages_json},
            {"count", messages.size()}
        };

        res.set_content(response.dump(), "application/json");
    }

    // POST /v1/messages/broadcast - Broadcast message
    void route_broadcast_message(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        agent_message msg;
        msg.message_id = "msg-" + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count());
        msg.from_agent_id = body.value("from_agent_id", "system");
        msg.to_agent_id = "";
        msg.type = MSG_TYPE_BROADCAST;
        msg.subject = body.value("subject", "");
        msg.payload = body.value("payload", json::object());
        msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        msg.conversation_id = body.value("conversation_id", "");

        orchestrator.broadcast_message(msg);

        json response = {
            {"success", true},
            {"message_id", msg.message_id}
        };

        res.set_content(response.dump(), "application/json");
    }

    // ========================================================================
    // Consensus Routes
    // ========================================================================

    // POST /v1/consensus/vote/create - Create vote
    void route_create_vote(const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        if (!body.contains("question") || !body.contains("options")) {
            res.status = 400;
            res.set_content(error_response("Missing required fields: question, options").dump(), "application/json");
            return;
        }

        std::string question = body["question"];
        std::vector<std::string> options = body["options"].get<std::vector<std::string>>();
        consensus_type type = str_to_consensus_type(body.value("type", "simple_majority"));
        int64_t deadline = body.value("deadline", 0);

        std::string vote_id = orchestrator.create_vote(question, options, type, deadline);

        json response = {
            {"vote_id", vote_id},
            {"status", "created"}
        };

        res.set_content(response.dump(), "application/json");
    }

    // POST /v1/consensus/vote/{vote_id}/cast - Cast vote
    void route_cast_vote(const httplib::Request & req, httplib::Response & res) {
        std::string vote_id = req.path_params.at("vote_id");

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_response("Invalid JSON in request body").dump(), "application/json");
            return;
        }

        if (!body.contains("agent_id") || !body.contains("option")) {
            res.status = 400;
            res.set_content(error_response("Missing required fields: agent_id, option").dump(), "application/json");
            return;
        }

        std::string agent_id = body["agent_id"];
        std::string option = body["option"];
        float weight = body.value("weight", 1.0f);

        if (!orchestrator.cast_vote(vote_id, agent_id, option, weight)) {
            res.status = 400;
            res.set_content(error_response("Failed to cast vote").dump(), "application/json");
            return;
        }

        json response = {
            {"success", true},
            {"vote_id", vote_id},
            {"agent_id", agent_id}
        };

        res.set_content(response.dump(), "application/json");
    }

    // GET /v1/consensus/vote/{vote_id} - Get vote result
    void route_get_vote(const httplib::Request & req, httplib::Response & res) {
        std::string vote_id = req.path_params.at("vote_id");

        consensus_vote vote;
        if (!orchestrator.get_vote_result(vote_id, vote)) {
            res.status = 404;
            res.set_content(error_response("Vote not found").dump(), "application/json");
            return;
        }

        res.set_content(vote.to_json().dump(), "application/json");
    }

    // ========================================================================
    // Stats Routes
    // ========================================================================

    // GET /v1/agents/stats - Get system stats
    void route_get_stats(const httplib::Request & req, httplib::Response & res) {
        json stats = orchestrator.get_stats();
        res.set_content(stats.dump(), "application/json");
    }

    // ========================================================================
    // Route Registration
    // ========================================================================

    template<typename T>
    void register_routes(T & server) {
        // Agent management
        server.Post("/v1/agents/spawn",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_spawn_agent(req, res);
                   });

        server.Get("/v1/agents",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_list_agents(req, res);
                  });

        server.Get(R"(/v1/agents/:agent_id)",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_get_agent(req, res);
                  });

        server.Delete(R"(/v1/agents/:agent_id)",
                     [this](const httplib::Request & req, httplib::Response & res) {
                         route_terminate_agent(req, res);
                     });

        // Task management
        server.Post("/v1/tasks/submit",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_submit_task(req, res);
                   });

        server.Post("/v1/tasks/workflow",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_submit_workflow(req, res);
                   });

        server.Get(R"(/v1/tasks/:task_id)",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_get_task(req, res);
                  });

        server.Get("/v1/tasks",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_list_tasks(req, res);
                  });

        server.Delete(R"(/v1/tasks/:task_id)",
                     [this](const httplib::Request & req, httplib::Response & res) {
                         route_cancel_task(req, res);
                     });

        // Knowledge base
        server.Post("/v1/knowledge",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_store_knowledge(req, res);
                   });

        server.Get(R"(/v1/knowledge/:key)",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_get_knowledge(req, res);
                  });

        server.Get("/v1/knowledge/query",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_query_knowledge(req, res);
                  });

        // Messaging
        server.Post("/v1/messages/send",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_send_message(req, res);
                   });

        server.Post("/v1/messages/broadcast",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_broadcast_message(req, res);
                   });

        server.Get(R"(/v1/messages/:agent_id)",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_receive_messages(req, res);
                  });

        // Consensus
        server.Post("/v1/consensus/vote/create",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_create_vote(req, res);
                   });

        server.Post(R"(/v1/consensus/vote/:vote_id/cast)",
                   [this](const httplib::Request & req, httplib::Response & res) {
                       route_cast_vote(req, res);
                   });

        server.Get(R"(/v1/consensus/vote/:vote_id)",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_get_vote(req, res);
                  });

        // Stats
        server.Get("/v1/agents/stats",
                  [this](const httplib::Request & req, httplib::Response & res) {
                      route_get_stats(req, res);
                  });
    }
};

}  // namespace agent_collab
