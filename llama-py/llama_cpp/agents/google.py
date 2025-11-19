"""
Google agent provider for Gemini models.

Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0, and future models.
"""

import os
import requests
from typing import List, Dict, Any, Optional

from .base import BaseAgent, Tool, Message


class GoogleAgent(BaseAgent):
    """
    Agent that uses Google Gemini API for function calling.

    Supports:
    - gemini-1.5-pro
    - gemini-1.5-flash
    - gemini-2.0-flash-exp
    - And future Gemini models

    Example:
        >>> import os
        >>> agent = GoogleAgent(
        ...     model="gemini-1.5-pro",
        ...     api_key=os.getenv("GOOGLE_API_KEY")
        ... )
        >>> response = agent.chat("Hello!", tools=[calculator_tool])
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        **kwargs
    ):
        """
        Initialize Google Gemini agent.

        Args:
            model: Model name (e.g., "gemini-1.5-pro", "gemini-2.0-flash-exp")
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            base_url: API base URL
            **kwargs: Additional parameters
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Google API key must be provided via api_key parameter "
                    "or GOOGLE_API_KEY environment variable"
                )

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def chat(
        self,
        message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Google Gemini API.

        Args:
            message: User message to send
            tools: List of Tool objects
            tool_choice: "auto" (Gemini doesn't support "required")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            system: System instruction (optional)
            **kwargs: Additional Gemini parameters

        Returns:
            Response from the model (normalized to OpenAI-like format)
        """
        if message:
            self.add_message("user", message)

        # Convert messages to Gemini format
        contents = []
        system_instruction = system

        for msg in self.conversation_history:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})

                # Add function calls
                if msg.tool_calls:
                    import json
                    for tool_call in msg.tool_calls:
                        parts.append({
                            "functionCall": {
                                "name": tool_call["function"]["name"],
                                "args": json.loads(tool_call["function"]["arguments"]),
                            }
                        })

                contents.append({
                    "role": "model",
                    "parts": parts
                })
            elif msg.role == "tool":
                # Function response
                import json
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.name,
                            "response": {"result": msg.content}
                        }
                    }]
                })

        # Prepare request
        request_data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            request_data["generationConfig"]["maxOutputTokens"] = max_tokens

        if system_instruction:
            request_data["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Add any additional kwargs to generationConfig
        request_data["generationConfig"].update(kwargs)

        # Add tools if provided
        if tools:
            request_data["tools"] = [{
                "functionDeclarations": self._convert_tools_to_provider_format(tools)
            }]

        # Build URL with API key
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        # Send request to Google
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            error_detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"Google API returned {response.status_code}: {error_detail}"
            )

        result = response.json()

        # Convert Gemini response to OpenAI-like format for consistency
        normalized_result = self._normalize_response(result)

        # Add assistant response to history
        assistant_message = normalized_result["choices"][0]["message"]
        msg = Message(
            role="assistant",
            content=assistant_message.get("content"),
            tool_calls=assistant_message.get("tool_calls"),
        )
        self.conversation_history.append(msg)

        return normalized_result

    def _convert_tools_to_provider_format(
        self,
        tools: List[Tool]
    ) -> List[Dict[str, Any]]:
        """
        Convert Tool objects to Gemini function declaration format.
        """
        gemini_tools = []
        for tool in tools:
            gemini_tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return gemini_tools

    def _normalize_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Gemini response format to OpenAI-like format for consistency.
        """
        import json

        content_text = ""
        tool_calls = []

        # Parse candidate parts
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])

            for part in parts:
                if "text" in part:
                    content_text += part["text"]
                elif "functionCall" in part:
                    # Convert to OpenAI tool call format
                    fc = part["functionCall"]
                    tool_calls.append({
                        "id": f"call_{fc['name']}_{len(tool_calls)}",  # Generate ID
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc.get("args", {})),
                        }
                    })

        # Build OpenAI-like response
        normalized = {
            "id": "gemini-" + str(hash(str(result)))[:16],
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_text if content_text else None,
                },
                "finish_reason": candidates[0].get("finishReason", "stop") if candidates else "stop",
            }],
            "usage": {
                "prompt_tokens": result.get("usageMetadata", {}).get("promptTokenCount", 0),
                "completion_tokens": result.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                "total_tokens": result.get("usageMetadata", {}).get("totalTokenCount", 0),
            }
        }

        if tool_calls:
            normalized["choices"][0]["message"]["tool_calls"] = tool_calls

        return normalized
