#!/usr/bin/env python3
"""
Multi-provider agent examples using llama_cpp Python bindings.

This example demonstrates how to use the same agent interface with different
AI providers:
- OpenAI (GPT-4, GPT-4.5, GPT-5)
- Anthropic (Claude Opus, Sonnet, Haiku)
- Google (Gemini 1.5 Pro, Flash, 2.0)
- Moonshot (Kimi models)
- Llama (local llama-server)

All providers support function calling using the same unified interface.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp.agents import (
    BaseAgent,
    Tool,
    LlamaAgent,
    OpenAIAgent,
    AnthropicAgent,
    GoogleAgent,
    MoonshotAgent,
)


def create_weather_tool() -> Tool:
    """
    Create a weather tool following the pattern from test_tool_call.py.

    This tool is compatible with all providers.
    """
    return Tool(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country/state, e.g. 'San Francisco, CA' or 'Paris, France'"
                }
            },
            "required": ["location"]
        }
    )


def create_calculator_tool() -> Tool:
    """Create a calculator tool for mathematical operations."""
    return Tool(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    )


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Execute a tool and return the result.

    In a real application, this would actually call external APIs or
    perform computations. Here we return mock results.
    """
    if tool_name == "get_current_weather":
        location = arguments.get("location", "Unknown")
        # Mock weather data
        return json.dumps({
            "location": location,
            "temperature": "22°C",
            "condition": "Partly cloudy",
            "humidity": "65%"
        })
    elif tool_name == "calculate":
        expression = arguments.get("expression", "")
        try:
            # Safe evaluation (in production, use a proper math parser)
            result = eval(expression, {"__builtins__": {}}, {})
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})
    else:
        return json.dumps({"error": "Unknown tool"})


def run_agent_conversation(agent: BaseAgent, tools: list[Tool]):
    """
    Run a conversation with function calling using any agent provider.

    This demonstrates the unified interface across all providers.
    """
    print(f"\n{'=' * 70}")
    print(f"Testing: {agent.__class__.__name__} with model '{agent.model}'")
    print(f"{'=' * 70}\n")

    # Add system message
    agent.add_message("system", "You are a helpful assistant with access to tools.")

    # Test 1: Weather query
    print("User: What's the weather like in Tokyo?")
    response = agent.chat(
        message="What's the weather like in Tokyo?",
        tools=tools,
    )

    # Check for tool calls
    choice = response["choices"][0]
    message = choice["message"]
    tool_calls = message.get("tool_calls")

    if tool_calls:
        print(f"Agent: [Calling {len(tool_calls)} tool(s)]")
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            print(f"  → {func_name}({json.dumps(arguments)})")

            # Execute the tool
            result = execute_tool(func_name, arguments)

            # Add tool result to conversation
            agent.add_tool_result(
                tool_call_id=tool_call.get("id", "call_0"),
                tool_name=func_name,
                result=result,
            )

        # Get final response with tool results
        final_response = agent.chat()
        final_content = final_response["choices"][0]["message"]["content"]
        print(f"Agent: {final_content}")
    else:
        content = message.get("content")
        print(f"Agent: {content}")

    print()


def example_openai():
    """Example using OpenAI GPT models."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Skipping OpenAI: OPENAI_API_KEY not set")
        return

    try:
        agent = OpenAIAgent(
            model="gpt-4-turbo",  # or "gpt-4", "gpt-4.5-preview", "gpt-5"
            api_key=api_key,
        )

        tools = [create_weather_tool(), create_calculator_tool()]
        run_agent_conversation(agent, tools)

    except Exception as e:
        print(f"❌ OpenAI error: {e}\n")


def example_anthropic():
    """Example using Anthropic Claude models."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  Skipping Anthropic: ANTHROPIC_API_KEY not set")
        return

    try:
        agent = AnthropicAgent(
            model="claude-3-5-sonnet-20241022",  # or "claude-3-opus-20240229", etc.
            api_key=api_key,
        )

        tools = [create_weather_tool(), create_calculator_tool()]
        run_agent_conversation(agent, tools)

    except Exception as e:
        print(f"❌ Anthropic error: {e}\n")


def example_google():
    """Example using Google Gemini models."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Skipping Google: GOOGLE_API_KEY not set")
        return

    try:
        agent = GoogleAgent(
            model="gemini-1.5-pro",  # or "gemini-1.5-flash", "gemini-2.0-flash-exp"
            api_key=api_key,
        )

        tools = [create_weather_tool(), create_calculator_tool()]
        run_agent_conversation(agent, tools)

    except Exception as e:
        print(f"❌ Google error: {e}\n")


def example_moonshot():
    """Example using Moonshot Kimi models."""
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("⚠️  Skipping Moonshot: MOONSHOT_API_KEY not set")
        return

    try:
        agent = MoonshotAgent(
            model="moonshot-v1-32k",  # or "moonshot-v1-8k", "moonshot-v1-128k"
            api_key=api_key,
        )

        tools = [create_weather_tool(), create_calculator_tool()]
        run_agent_conversation(agent, tools)

    except Exception as e:
        print(f"❌ Moonshot error: {e}\n")


def example_llama():
    """Example using local llama-server."""
    try:
        agent = LlamaAgent(
            model="llama-3.1-8b",
            base_url="http://localhost:8080",
        )

        # Check if server is running
        if not agent.health_check():
            print("⚠️  Skipping Llama: llama-server not running at http://localhost:8080")
            print("   Start with: llama-server --jinja -m model.gguf --port 8080\n")
            return

        tools = [create_weather_tool(), create_calculator_tool()]
        run_agent_conversation(agent, tools)

    except Exception as e:
        print(f"❌ Llama error: {e}\n")


def main():
    """Main function to run all agent examples."""
    print("\n" + "="*70)
    print("Multi-Provider Agent Function Calling Examples")
    print("="*70)
    print("\nThis demonstrates using the same agent interface with different")
    print("AI providers: OpenAI, Anthropic, Google, Moonshot, and local Llama.\n")
    print("Set API keys via environment variables:")
    print("  - OPENAI_API_KEY      (for GPT-4, GPT-4.5, GPT-5)")
    print("  - ANTHROPIC_API_KEY   (for Claude Opus, Sonnet, Haiku)")
    print("  - GOOGLE_API_KEY      (for Gemini 1.5 Pro, Flash, 2.0)")
    print("  - MOONSHOT_API_KEY    (for Kimi models)")
    print("  - (llama-server at http://localhost:8080 for local Llama)")
    print()

    # Run examples for each provider
    example_openai()
    example_anthropic()
    example_google()
    example_moonshot()
    example_llama()

    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
