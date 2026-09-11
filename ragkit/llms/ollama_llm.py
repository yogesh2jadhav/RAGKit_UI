"""
Purpose
-------
Generates text using an Ollama Large Language Model.

Responsibilities
----------------
- Connect to an Ollama server.
- Send prompts.
- Return generated responses.

Does NOT
--------
- Retrieve documents.
- Build prompts.
- Generate embeddings.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import ollama

from ragkit.config.llm_config import LLMConfig
from ragkit.exceptions import LLMError
from ragkit.llms.llm import LLM
from ragkit.logger import logger
from ragkit.models.llm_response import LLMResponse

'''
=> OllamaLLM is class which is implement LLM interface. with generate method.
'''
class OllamaLLM(LLM):
    """
    Ollama implementation of the LLM interface.
    => Following is the constructor for this class which take model name and host url. as input and
    connect to an Ollama server.
    """
    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        *,
        config: LLMConfig | None = None,
        think: bool | None = False,
    ) -> None:
        """
        Initialize the Ollama LLM.

        Parameters
        ----------
        model Name of the language model.
        host Ollama server URL.
        config  Optional LLM configuration.
        think
            Whether to enable "thinking" mode for reasoning models such
            as qwen3 / deepseek-r1. Thinking mode produces a long
            chain-of-thought before the final answer, which is usually
            not shown to the user but still has to be generated -
            this can take minutes on CPU-only machines. Defaults to
            ``False`` (disabled) for faster, more predictable latency.
            Pass ``None`` to use the model's own default, or ``True``
            to force it on. Ignored by models that don't support it.
        """

        #
        # Configuration overrides explicit parameters.
        #
        if config is not None:
            model = config.model

        self._model_name = model
        self._think = think

        self._client = ollama.Client(  # => We are createing ollam client here.
            host=host,
        )

    def generate(
        self,
        prompt: str,
        options: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Generate a response using Ollama.
        """

        started_at = time.monotonic()

        try:
            # => where we send Prompt (all search output and question to Ollama)
            # and get response
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                think=self._think,
                options=dict(options) if options else None,
            )

            elapsed = time.monotonic() - started_at

            logger.info(
                "Ollama generate: model=%s, think=%s, took=%.2fs",
                self._model_name,
                self._think,
                elapsed,
            )

            return LLMResponse(
                content=response["response"],
            )

        except Exception as ex:
            elapsed = time.monotonic() - started_at

            logger.error(
                "Ollama generate failed: model=%s, took=%.2fs, error=%s",
                self._model_name,
                elapsed,
                ex,
            )

            raise LLMError(
                f"Failed to generate response using Ollama: {ex}"
            ) from ex
