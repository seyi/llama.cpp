#!/usr/bin/env python3
"""
Agent function calling example using llama_cpp Python bindings.

This example demonstrates how to use the llama_cpp bindings for agent-like
functionality with function calling / tool use. It shows how to integrate
with the llama-server's OpenAI-compatible API.

The example follows the pattern from:
- tools/server/tests/unit/test_tool_call.py
- docs/function-calling.md

NOTE: This example requires llama-server to be running with --jinja flag.
Start the server with:
    llama-server --jinja -m model.gguf --port 8080
"""

import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import LlamaModel


class LlamaAgent:
    """
    Agent class that uses llama-server for function calling.

    This demonstrates the agent pattern described in the llama.cpp repository,
    using the OpenAI-compatible function calling API provided by llama-server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model_name: str = "gpt-3.5-turbo",
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.conversation_history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
        })

    def define_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Define a tool/function following OpenAI format.

        Example from test_tool_call.py:
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a chat message and optionally provide tools for the agent to use.

        This follows the pattern from test_tool_call.py and function-calling.md.

        Args:
            message: User message to send
            tools: List of tool definitions
            tool_choice: "auto", "required", or "none"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Response from the model including any tool calls
        """
        if message:
            self.add_message("user", message)

        # Prepare request following the test_tool_call.py pattern
        request_data = {
            "model": self.model_name,
            "messages": self.conversation_history,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = tool_choice

        # Send request to llama-server
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Server returned {response.status_code}: {response.text}"
            )

        result = response.json()
        assistant_message = result["choices"][0]["message"]

        # Add assistant response to history
        self.conversation_history.append(assistant_message)

        return result

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """Add tool execution result to conversation"""
        self.conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })


def example_weather_agent():
    """
    Example agent that can answer weather queries.

    This follows the weather example from test_tool_call.py:do_test_weather()
    """
    print("=" * 60)
    print("Weather Agent Example")
    print("=" * 60)

    # Create agent
    agent = LlamaAgent()

    # Add system message
    agent.add_message(
        "system",
        "You are a chatbot that uses tools/functions. Don't overthink things."
    )

    # Define weather tool (from test_tool_call.py:WEATHER_TOOL)
    weather_tool = agent.define_tool(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country/state, e.g. 'San Francisco, CA', or 'Paris, France'"
                }
            },
            "required": ["location"]
        }
    )

    # Query the agent
    print("\nUser: What is the weather in Istanbul?")
    response = agent.chat(
        message="What is the weather in Istanbul?",
        tools=[weather_tool],
    )

    # Check for tool calls
    choice = response["choices"][0]
    tool_calls = choice["message"].get("tool_calls")

    if tool_calls:
        print(f"\nAgent wants to call tools: {len(tool_calls)}")
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            print(f"  - {func_name}({json.dumps(arguments)})")

            # Simulate tool execution
            if func_name == "get_current_weather":
                location = arguments.get("location", "")
                # Mock weather result
                weather_result = json.dumps({
                    "location": location,
                    "temperature": "22°C",
                    "condition": "Partly cloudy"
                })

                # Add tool result to conversation
                agent.add_tool_result(
                    tool_call_id=tool_call.get("id", "call_0"),
                    tool_name=func_name,
                    result=weather_result,
                )

        # Get final response with tool results
        print("\nGetting final response with tool results...")
        final_response = agent.chat()
        final_content = final_response["choices"][0]["message"]["content"]
        print(f"\nAgent: {final_content}")
    else:
        content = choice["message"].get("content")
        print(f"\nAgent: {content}")


def example_python_code_agent():
    """
    Example agent that can execute Python code.

    This follows the python tool example from test_tool_call.py:PYTHON_TOOL
    """
    print("\n" + "=" * 60)
    print("Python Code Agent Example")
    print("=" * 60)

    agent = LlamaAgent()

    # Add system message
    agent.add_message("system", "You are a coding assistant.")

    # Define python tool (from test_tool_call.py:PYTHON_TOOL)
    python_tool = agent.define_tool(
        name="python",
        description="Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to run in the ipython interpreter."
                }
            },
            "required": ["code"]
        }
    )

    # Query the agent
    print("\nUser: Print a hello world message with python.")
    response = agent.chat(
        message="Print a hello world message with python.",
        tools=[python_tool],
    )

    # Check for tool calls
    choice = response["choices"][0]
    tool_calls = choice["message"].get("tool_calls")

    if tool_calls:
        print(f"\nAgent wants to execute code:")
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            code = arguments.get("code", "")
            print(f"\n```python\n{code}\n```")
    else:
        content = choice["message"].get("content")
        print(f"\nAgent: {content}")


def main():
    """Main function to run agent examples"""
    print("\nLlama.cpp Agent Function Calling Examples")
    print("==========================================")
    print("\nThis demonstrates the 'agent' functionality in llama.cpp")
    print("through the OpenAI-compatible function calling API.\n")
    print("Prerequisites:")
    print("  1. Build llama.cpp with BUILD_SHARED_LIBS=ON")
    print("  2. Start llama-server:")
    print("     llama-server --jinja -m model.gguf --port 8080")
    print("     (Use a model with tool calling support, e.g., Llama 3.1+)\n")

    try:
        # Test server connection
        response = requests.get("http://localhost:8080/health")
        if response.status_code != 200:
            raise ConnectionError("Server not responding")
    except Exception as e:
        print(f"Error: Cannot connect to llama-server at http://localhost:8080")
        print(f"Please start the server with: llama-server --jinja -m model.gguf --port 8080")
        return 1

    try:
        # Run examples
        example_weather_agent()
        example_python_code_agent()

        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nError running examples: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
