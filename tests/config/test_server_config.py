from pathlib import Path

from ragkit.config.server_config import default_server_config


def test_default_server_config():
    """
    Verify the default server configuration.
    """

    config = default_server_config()

    assert isinstance(
        config.vector_db_path,
        Path,
    )

    assert config.vector_db_path.name == "vector_db_rrf"
    assert config.collection_name == "ragkit_rrf"