from ragkit.config.llm_config import LLMConfig


def test_create_llm_config():
    config = LLMConfig(
        model="qwen3:8b",
        temperature=0.2,
    )

    assert config.model == "qwen3:8b"
    assert config.temperature == 0.2


def test_llm_config_think_defaults_true():
    config = LLMConfig()

    assert config.think is True


def test_llm_config_think_override():
    config = LLMConfig(
        think=False,
    )

    assert config.think is False