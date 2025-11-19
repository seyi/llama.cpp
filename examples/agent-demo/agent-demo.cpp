#include "ggml-agent.h"
#include <iostream>
#include <thread>
#include <chrono>

// ============================================================================
// Example Worker Agent
// ============================================================================

class WorkerAgent : public ggml_agent {
public:
    WorkerAgent(const std::string& id) : ggml_agent(id) {}

protected:
    void on_start() override {
        std::cout << "[" << id << "] Worker started" << std::endl;

        // Register handler for task messages
        register_handler(GGML_AGENT_MSG_TASK,
            [this](const ggml_agent_msg& msg) {
                std::cout << "[" << id << "] Received task from " << msg.from_id << std::endl;
                process_task(msg);
            });

        register_handler(GGML_AGENT_MSG_DOC_UPDATE,
            [this](const ggml_agent_msg& msg) {
                std::cout << "[" << id << "] Document updated" << std::endl;
            });
    }

    void on_stop() override {
        std::cout << "[" << id << "] Worker stopped" << std::endl;
    }

private:
    void process_task(const ggml_agent_msg& msg) {
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Send result back
        std::vector<uint8_t> result_data = {'D', 'O', 'N', 'E'};
        send_to(msg.from_id, GGML_AGENT_MSG_TASK_RESULT, result_data);
        std::cout << "[" << id << "] Task completed" << std::endl;
    }
};

// ============================================================================
// Example Document Editor Agent
// ============================================================================

class EditorAgent : public ggml_agent {
private:
    std::string coordinator_id;
    size_t section_to_edit;
    bool lock_acquired = false;

public:
    EditorAgent(const std::string& id, const std::string& coord_id, size_t section)
        : ggml_agent(id), coordinator_id(coord_id), section_to_edit(section) {}

protected:
    void on_start() override {
        std::cout << "[" << id << "] Editor started, requesting lock on section "
                  << section_to_edit << std::endl;

        // Register handlers
        register_handler(GGML_AGENT_MSG_LOCK_ACQUIRED,
            [this](const ggml_agent_msg& msg) {
                lock_acquired = true;
                std::cout << "[" << id << "] Lock acquired, editing..." << std::endl;
                perform_edit();
            });

        register_handler(GGML_AGENT_MSG_LOCK_DENIED,
            [this](const ggml_agent_msg& msg) {
                std::cout << "[" << id << "] Lock denied, retrying..." << std::endl;
                // Retry after a delay
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                request_lock();
            });

        register_handler(GGML_AGENT_MSG_DOC_UPDATE,
            [this](const ggml_agent_msg& msg) {
                std::cout << "[" << id << "] Received document update" << std::endl;
            });

        // Request lock on section
        request_lock();
    }

    void on_stop() override {
        if (lock_acquired) {
            release_lock();
        }
        std::cout << "[" << id << "] Editor stopped" << std::endl;
    }

private:
    void request_lock() {
        std::vector<uint8_t> payload(sizeof(size_t));
        std::memcpy(payload.data(), &section_to_edit, sizeof(size_t));
        send_to(coordinator_id, GGML_AGENT_MSG_LOCK_REQUEST, payload);
    }

    void release_lock() {
        std::vector<uint8_t> payload(sizeof(size_t));
        std::memcpy(payload.data(), &section_to_edit, sizeof(size_t));
        send_to(coordinator_id, GGML_AGENT_MSG_LOCK_RELEASE, payload);
        lock_acquired = false;
    }

    void perform_edit() {
        // Simulate editing
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Send edit to coordinator
        std::vector<uint8_t> payload(sizeof(size_t));
        std::memcpy(payload.data(), &section_to_edit, sizeof(size_t));

        std::string edit_content = "Edited by " + id;
        payload.insert(payload.end(), edit_content.begin(), edit_content.end());

        send_to(coordinator_id, GGML_AGENT_MSG_DOC_EDIT, payload);
        std::cout << "[" << id << "] Edit applied" << std::endl;

        // Release lock after edit
        release_lock();
    }
};

// ============================================================================
// Demo Functions
// ============================================================================

void demo_supervisor_recovery() {
    std::cout << "\n=== Demo 1: Supervisor with Failure Recovery ===" << std::endl;

    // Create supervisor
    auto supervisor = std::make_shared<ggml_agent_supervisor>("supervisor");
    supervisor->strategy = GGML_AGENT_RESTART_ONE_FOR_ONE;
    supervisor->max_restarts = 3;

    // Create worker agents
    auto worker1 = std::make_shared<WorkerAgent>("worker1");
    auto worker2 = std::make_shared<WorkerAgent>("worker2");

    // Register agents
    ggml_agent_registry::instance().register_agent(supervisor);
    ggml_agent_registry::instance().register_agent(worker1);
    ggml_agent_registry::instance().register_agent(worker2);

    // Add workers to supervisor
    supervisor->add_child(worker1);
    supervisor->add_child(worker2);

    // Start supervisor (will start children)
    supervisor->start();

    // Send tasks to workers
    std::cout << "\nSending tasks to workers..." << std::endl;
    std::vector<uint8_t> task_data = {'T', 'A', 'S', 'K'};

    ggml_agent_msg task1("main", "worker1", GGML_AGENT_MSG_TASK, task_data);
    ggml_agent_msg task2("main", "worker2", GGML_AGENT_MSG_TASK, task_data);

    worker1->send(task1);
    worker2->send(task2);

    // Let workers process
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Simulate worker failure
    std::cout << "\nSimulating worker1 failure..." << std::endl;
    worker1->circuit_breaker.record_failure();
    worker1->circuit_breaker.record_failure();
    worker1->circuit_breaker.record_failure();
    worker1->circuit_breaker.record_failure();
    worker1->circuit_breaker.record_failure();

    ggml_agent_msg error_msg("worker1", "supervisor", GGML_AGENT_MSG_ERROR);
    supervisor->send(error_msg);

    // Wait for recovery
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Cleanup
    supervisor->stop();
    supervisor->join();

    ggml_agent_registry::instance().unregister_agent("supervisor");
    ggml_agent_registry::instance().unregister_agent("worker1");
    ggml_agent_registry::instance().unregister_agent("worker2");

    std::cout << "Demo 1 completed\n" << std::endl;
}

void demo_document_coordination() {
    std::cout << "\n=== Demo 2: Document Coordination with Concurrent Editing ===" << std::endl;

    // Create coordinator
    auto coordinator = std::make_shared<ggml_agent_coordinator>("coordinator", 5);

    // Create editor agents
    auto editor1 = std::make_shared<EditorAgent>("editor1", "coordinator", 0);
    auto editor2 = std::make_shared<EditorAgent>("editor2", "coordinator", 1);
    auto editor3 = std::make_shared<EditorAgent>("editor3", "coordinator", 0); // Conflict!

    // Register agents
    ggml_agent_registry::instance().register_agent(coordinator);
    ggml_agent_registry::instance().register_agent(editor1);
    ggml_agent_registry::instance().register_agent(editor2);
    ggml_agent_registry::instance().register_agent(editor3);

    // Start all agents
    coordinator->start();
    editor1->start();
    editor2->start();
    editor3->start();

    std::cout << "\nEditors working on document..." << std::endl;
    std::cout << "Note: editor1 and editor3 both want section 0 (conflict)" << std::endl;

    // Let editors work
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Stop all
    editor1->stop();
    editor2->stop();
    editor3->stop();
    coordinator->stop();

    editor1->join();
    editor2->join();
    editor3->join();
    coordinator->join();

    ggml_agent_registry::instance().unregister_agent("coordinator");
    ggml_agent_registry::instance().unregister_agent("editor1");
    ggml_agent_registry::instance().unregister_agent("editor2");
    ggml_agent_registry::instance().unregister_agent("editor3");

    std::cout << "Demo 2 completed\n" << std::endl;
}

void demo_circuit_breaker() {
    std::cout << "\n=== Demo 3: Circuit Breaker Pattern ===" << std::endl;

    ggml_agent_circuit_breaker breaker;
    breaker.failure_threshold = 3;
    breaker.open_timeout_ms = 2000;

    std::cout << "Circuit state: CLOSED (normal operation)" << std::endl;

    // Simulate failures
    for (int i = 0; i < 5; i++) {
        bool allowed = breaker.allow_request();
        std::cout << "Request " << (i+1) << ": " << (allowed ? "ALLOWED" : "DENIED") << std::endl;

        if (allowed) {
            breaker.record_failure();
            std::cout << "  -> Failed (count: " << breaker.failure_count << ")" << std::endl;

            if (breaker.state == GGML_AGENT_CIRCUIT_OPEN) {
                std::cout << "  -> Circuit OPENED (fast-fail mode)" << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nWaiting for circuit to transition to HALF_OPEN..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));

    std::cout << "Attempting request after timeout..." << std::endl;
    bool allowed = breaker.allow_request();
    std::cout << "Request: " << (allowed ? "ALLOWED" : "DENIED") << std::endl;
    if (allowed) {
        std::cout << "Circuit state: HALF_OPEN (testing recovery)" << std::endl;

        // Simulate success
        breaker.record_success();
        breaker.record_success();
        std::cout << "Recording successes... Circuit CLOSED (recovered)" << std::endl;
    }

    std::cout << "Demo 3 completed\n" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  GGML Agent-to-Agent Protocol Demo" << std::endl;
    std::cout << "==================================================" << std::endl;

    try {
        // Run demos
        demo_supervisor_recovery();
        demo_document_coordination();
        demo_circuit_breaker();

        std::cout << "\n==================================================" << std::endl;
        std::cout << "  All demos completed successfully!" << std::endl;
        std::cout << "==================================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
