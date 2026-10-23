"""LLM provider interface (SPEC 8/9).

Both LLM stages (repair, format) only need one ability: send a system prompt
plus a user prompt and get plain text back. More exotic features would leak
provider details into the pipeline, so the interface stays this small.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LlmProvider(ABC):
    """Provider of chat completions for the repair and formatting stages."""

    #: short machine-readable name, used in cache metadata
    name: str = "llm"

    @property
    def model(self) -> str:
        """Model identifier, for cache metadata and logs."""
        return ""

    @abstractmethod
    async def chat(
        self, *, system: str, user: str, temperature: float = 0.0
    ) -> str:
        """Return the model's text answer (may raise)."""
