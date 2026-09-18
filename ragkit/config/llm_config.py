"""
Purpose
-------
Configuration for language models.

Responsibilities
----------------
- Store LLM configuration.

Does NOT
--------
- Generate responses.
"""

from __future__ import annotations

from dataclasses import dataclass

# => This is just like DTO in java

@dataclass(frozen=True, slots=True)
class LLMConfig:
    """
    Configuration for an LLM.
    """

    model: str = "qwen3:8b"
    temperature: float = 0.0

    #
    # Whether to enable "thinking" mode for reasoning models such as
    # qwen3 / deepseek-r1 (see OllamaLLM). Ignored by models that
    # don't support it.
    #
    think: bool = True
