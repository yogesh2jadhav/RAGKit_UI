from unittest.mock import MagicMock, patch

from pytest import raises

from ragkit.config.llm_config import LLMConfig
from ragkit.exceptions import LLMError
from ragkit.llms.ollama_llm import OllamaLLM


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate(mock_client):
    """
    Verify OllamaLLM generates a response.
    """

    client = MagicMock()

    client.generate.return_value = {
        "response": "Hello World",
    }

    mock_client.return_value = client

    llm = OllamaLLM(
        model="llama3.2",
    )

    response = llm.generate(
        prompt="Say hello",
    )

    assert response.content == "Hello World"

    client.generate.assert_called_once_with(
        model="llama3.2",
        prompt="Say hello",
        think=False,
        options=None,
    )


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate_with_options(mock_client):
    """
    Verify generation options are forwarded.
    """

    client = MagicMock()

    client.generate.return_value = {
        "response": "Hello",
    }

    mock_client.return_value = client

    llm = OllamaLLM(
        model="llama3.2",
    )

    llm.generate(
        prompt="Hello",
        options={
            "temperature": 0,
        },
    )

    client.generate.assert_called_once_with(
        model="llama3.2",
        prompt="Hello",
        think=False,
        options={
            "temperature": 0,
        },
    )


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate_failure(mock_client):
    """
    Verify provider exceptions are wrapped.
    """

    client = MagicMock()

    client.generate.side_effect = RuntimeError(
        "Connection failed",
    )

    mock_client.return_value = client

    llm = OllamaLLM(
        model="llama3.2",
    )

    with raises(LLMError) as ex:
        llm.generate(
            prompt="Hello",
        )

    assert str(ex.value) == (
        "Failed to generate response using Ollama: Connection failed"
    )


def test_create_llm_with_default_model():
    """
    Verify the default model is used.
    """

    llm = OllamaLLM()

    assert llm._model_name == "qwen3:8b"


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate_think_override(mock_client):
    """
    Verify the `think` flag can be overridden.
    """

    client = MagicMock()

    client.generate.return_value = {
        "response": "Hello",
    }

    mock_client.return_value = client

    llm = OllamaLLM(
        model="qwen3:8b",
        think=True,
    )

    llm.generate(
        prompt="Hello",
    )

    client.generate.assert_called_once_with(
        model="qwen3:8b",
        prompt="Hello",
        think=True,
        options=None,
    )


def test_create_llm_with_config():
    """
    Verify LLMConfig overrides the default model.
    """

    config = LLMConfig(
        model="llama3.1:8b",
    )

    llm = OllamaLLM(
        config=config,
    )

    assert llm._model_name == "llama3.1:8b"