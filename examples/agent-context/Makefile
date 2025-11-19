# Agent Context Management System Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I../../include
LDFLAGS =

# Targets
EXAMPLE = agent-example
TEST = test-agent-context

.PHONY: all clean test example

all: example test

example: $(EXAMPLE)

test: $(TEST)
	@echo "Running tests..."
	@./$(TEST)

$(EXAMPLE): agent-example.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(TEST): test-agent-context.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(EXAMPLE) $(TEST) *.state *.bin

.DEFAULT_GOAL := all
