"""Async wrapper around the Gemini Interactions API."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from google import genai


@dataclass
class Usage:
    """Token usage stats for a single interaction."""
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StreamChunk:
    """A piece of streamed interaction output."""
    text: str = ""
    thought_summary: str = ""
    is_complete: bool = False
    usage: Usage | None = None
    interaction_id: str = ""


@dataclass
class SessionUsage:
    """Cumulative usage across a session."""
    total_input: int = 0
    total_output: int = 0
    total_thought: int = 0
    total_tokens: int = 0
    interaction_count: int = 0

    def add(self, usage: Usage) -> None:
        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens
        self.total_thought += usage.thought_tokens
        self.total_tokens += usage.total_tokens
        self.interaction_count += 1


class ChittiClient:
    """Async client for Gemini Interactions API with streaming and multi-turn."""

    def __init__(self, api_key: str, config: dict, soul: str) -> None:
        self._genai = genai.Client(api_key=api_key)
        self._model: str = config["model"]["name"]
        self._system_instruction: str = soul
        self._generation_config: dict = {
            "temperature": config["generation"]["temperature"],
            "max_output_tokens": config["generation"]["max_output_tokens"],
            "thinking_level": config["generation"]["thinking_level"],
        }
        self._previous_id: str | None = None
        self._last_usage: Usage | None = None
        self.session_usage = SessionUsage()

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_usage(self) -> Usage | None:
        return self._last_usage

    @property
    def previous_interaction_id(self) -> str | None:
        return self._previous_id

    def new_conversation(self) -> None:
        """Reset multi-turn state for a fresh conversation."""
        self._previous_id = None

    async def send(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """Send a message and yield streamed response chunks."""
        stream = self._genai.aio.interactions.create(
            model=self._model,
            input=user_input,
            system_instruction=self._system_instruction,
            generation_config=self._generation_config,
            previous_interaction_id=self._previous_id,
            stream=True,
        )

        async for chunk in await stream:
            match chunk.event_type:
                case "content.delta":
                    delta_type = chunk.delta.type if hasattr(chunk, "delta") and chunk.delta else None
                    if delta_type == "text":
                        text = chunk.delta.text if hasattr(chunk.delta, "text") else ""
                        yield StreamChunk(text=text)
                    elif delta_type == "thought_summary":
                        summary = ""
                        if hasattr(chunk.delta, "content") and chunk.delta.content:
                            summary = chunk.delta.content.text if hasattr(chunk.delta.content, "text") else ""
                        yield StreamChunk(thought_summary=summary)

                case "interaction.complete":
                    interaction = chunk.interaction
                    self._previous_id = interaction.id

                    usage = Usage()
                    if interaction.usage:
                        u = interaction.usage
                        usage = Usage(
                            input_tokens=getattr(u, "total_input_tokens", 0) or 0,
                            output_tokens=getattr(u, "total_output_tokens", 0) or 0,
                            thought_tokens=getattr(u, "total_thought_tokens", 0) or 0,
                            total_tokens=getattr(u, "total_tokens", 0) or 0,
                        )
                    self._last_usage = usage
                    self.session_usage.add(usage)

                    yield StreamChunk(
                        is_complete=True,
                        usage=usage,
                        interaction_id=interaction.id,
                    )

    async def send_sync(self, user_input: str) -> str:
        """Send a message and return the full response text (non-streaming)."""
        interaction = await self._genai.aio.interactions.create(
            model=self._model,
            input=user_input,
            system_instruction=self._system_instruction,
            generation_config=self._generation_config,
            previous_interaction_id=self._previous_id,
        )

        self._previous_id = interaction.id

        if interaction.usage:
            u = interaction.usage
            usage = Usage(
                input_tokens=getattr(u, "total_input_tokens", 0) or 0,
                output_tokens=getattr(u, "total_output_tokens", 0) or 0,
                thought_tokens=getattr(u, "total_thought_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )
            self._last_usage = usage
            self.session_usage.add(usage)

        if interaction.outputs:
            return interaction.outputs[-1].text or ""
        return ""
