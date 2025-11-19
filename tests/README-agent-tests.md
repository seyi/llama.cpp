# Agent Collaboration Framework Tests

Comprehensive test suite for the agent collaboration framework.

## Test Coverage

The test suite includes 25 tests covering all major components:

### Core Functionality (5 tests)
1. **UUID Generation** - Validates UUID format and uniqueness
2. **Timestamp Generation** - Tests timestamp creation and ordering
3. **Message Serialization** - Tests agent_request JSON serialization
4. **Response Serialization** - Tests agent_response JSON serialization
5. **Message Queue** - Tests thread-safe message queue operations

### Conversation Memory (8 tests)
6. **Conversation Memory Creation** - Tests thread creation and tracking
7. **Conversation Turns** - Tests adding user/assistant turns
8. **Conversation History Building** - Tests history reconstruction
9. **Context Reconstruction** - Tests conversation continuation
10. **Thread Expiration** - Tests TTL and thread keep-alive
11. **Thread Branching** - Tests creating child threads
12. **Thread Cleanup** - Tests thread deletion
13. **Token Estimation** - Tests token counting utilities

### Agent Management (8 tests)
14. **Agent Creation** - Tests local agent factory
15. **Agent Registration** - Tests agent registration with registry
16. **Agent Discovery** - Tests capability-based agent discovery
17. **Agent Request Processing** - Tests request/response cycle
18. **Multi-turn Conversation** - Tests conversation continuity
19. **Agent Statistics** - Tests per-agent stats tracking
20. **Agent Status** - Tests agent status management
21. **Registry Statistics** - Tests registry-wide statistics

### Failure Handling (3 tests)
22. **Failure Policy** - Tests default/aggressive/conservative policies
23. **Circuit Breaker** - Tests circuit breaker state transitions
24. **Error Type Conversion** - Tests error type string conversion

### Concurrency (1 test)
25. **Concurrent Message Queue** - Tests thread-safe queue with producers/consumers

## Running Tests

### Build Tests

```bash
cd build
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
make test-agent-collaboration
```

### Run All Tests

```bash
./bin/test-agent-collaboration
```

### Run with CTest

```bash
ctest -R test-agent-collaboration -V
```

## Test Output

Successful run output:
```
=== Agent Collaboration Framework Tests ===

Running test_uuid_generation...
  PASS
Running test_timestamp_generation...
  PASS
...
(all 25 tests)
...

=== Test Results ===
Total:  25
Passed: 25
Failed: 0

All tests passed! ✓
```

## Test Structure

Each test follows this pattern:

```cpp
static bool test_name() {
    // Setup
    conversation_memory memory(1);

    // Test logic
    agent_request req;
    req.prompt = "Test";

    // Assertions
    TEST_ASSERT(condition, "Error message");

    // Cleanup (if needed)
    return true;
}
```

## Adding New Tests

1. Add test function following the pattern above
2. Add forward declaration after the mock_inference function
3. Add `RUN_TEST(test_name)` in main()
4. Rebuild and run

Example:
```cpp
// Forward declaration
static bool test_my_feature();

// Implementation
static bool test_my_feature() {
    // Your test code
    TEST_ASSERT(condition, "Message");
    return true;
}

// In main()
RUN_TEST(test_my_feature);
```

## CI/CD Integration

The tests are automatically run as part of the CMake test suite:

```bash
cmake --build build --target test
# or
ctest --test-dir build
```

## Test Categories

Tests are labeled with "agent" category:
```bash
ctest -L agent  # Run only agent tests
```

## Debugging Failed Tests

When a test fails, the output shows:
- Test name
- Exact line where assertion failed
- Error message describing what failed

Example failure output:
```
Running test_message_queue...
FAIL: Queue should not be empty at test-agent-collaboration.cpp:155
  FAIL
```

## Performance

All 25 tests complete in less than 1 second on typical hardware.

## Coverage Summary

- ✅ Message Protocol (serialization, queuing)
- ✅ Conversation Memory (creation, turns, history, branching)
- ✅ Agent Lifecycle (creation, registration, discovery)
- ✅ Request Processing (single and multi-turn)
- ✅ Failure Handling (policies, circuit breaker, retries)
- ✅ Statistics & Monitoring
- ✅ Concurrency & Thread Safety

## Known Limitations

- Tests use mock inference (not real llama.cpp inference)
- File-based context reconstruction not tested (requires actual files)
- Remote agents not tested (only local agents)
- Network failure scenarios not covered
- Performance/stress testing not included

## Future Test Additions

- Integration tests with actual llama.cpp inference
- Performance/benchmarking tests
- Stress tests (many agents, many threads)
- Network failure simulation
- MCP protocol compatibility tests
- End-to-end workflow tests

## See Also

- [Agent Collaboration Documentation](../docs/agent-collaboration.md)
- [Example Usage](../examples/agent-collaboration/README.md)
- [Main README](../README.md)
