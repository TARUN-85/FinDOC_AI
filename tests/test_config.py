import pytest

from config import PROJECT_ROOT, load_settings
from schemas import ConfigError


def load(env=None, yaml_text=None, tmp_path=None):
    f = None
    if yaml_text is not None:
        f = tmp_path / "settings.yaml"
        f.write_text(yaml_text)
    return load_settings(env=env or {}, settings_file=f or (tmp_path / "none.yaml" if tmp_path else None), use_dotenv=False)


def test_defaults(tmp_path):
    s = load(tmp_path=tmp_path)
    assert s.llm_provider == "gemini" and s.chunk_size == 1000 and s.chunk_overlap == 150 and s.top_k == 5
    assert s.llm_temperature is None


def test_env_overrides_yaml(tmp_path):
    s = load({"TOP_K": "8"}, "top_k: 3\nchunk_size: 500\n", tmp_path)
    assert s.top_k == 8 and s.chunk_size == 500


def test_blank_env_falls_through(tmp_path):
    s = load({"EMBEDDING_MODEL": "  "}, "embedding_model: custom-model\n", tmp_path)
    assert s.embedding_model == "custom-model"
    assert load({"EMBEDDING_MODEL": ""}, tmp_path=tmp_path).embedding_model == "gemini-embedding-001"


def test_provider_names_lowercased(tmp_path):
    assert load({"LLM_PROVIDER": "Gemini"}, tmp_path=tmp_path).llm_provider == "gemini"


def test_paths_resolve_against_project_root(tmp_path):
    s = load(tmp_path=tmp_path)
    assert s.input_dir == PROJECT_ROOT / "InputDoc"
    assert load({"LOG_DIR": "custom_logs"}, tmp_path=tmp_path).log_dir == (PROJECT_ROOT / "custom_logs").resolve()


@pytest.mark.parametrize("env", [
    {"CHUNK_SIZE": "abc"}, {"CHUNK_OVERLAP": "1000"}, {"CHUNK_SIZE": "50"},
    {"TOP_K": "0"}, {"MIN_RELEVANCE": "2"}, {"LOG_LEVEL": "LOUD"}, {"EMBEDDING_DIMENSIONS": "10"},
])
def test_invalid_values_raise(env, tmp_path):
    with pytest.raises(ConfigError):
        load(env, tmp_path=tmp_path)


def test_secret_in_yaml_rejected(tmp_path):
    with pytest.raises(ConfigError, match="secrets"):
        load(yaml_text="gemini_api_key: abc\n", tmp_path=tmp_path)


def test_api_key_from_env_and_hidden_in_repr(tmp_path):
    s = load({"GEMINI_API_KEY": "super-secret"}, tmp_path=tmp_path)
    assert s.gemini_api_key == "super-secret"
    assert "super-secret" not in repr(s)


def test_unsupported_provider_message(tmp_path):
    from embeddings import create_embedding_provider
    from llm_provider import create_llm_provider

    s = load({"LLM_PROVIDER": "other", "EMBEDDING_PROVIDER": "other"}, tmp_path=tmp_path)
    with pytest.raises(ConfigError, match="Supported: gemini"):
        create_llm_provider(s)
    with pytest.raises(ConfigError, match="Supported: gemini"):
        create_embedding_provider(s)


def test_missing_key_raises_clear_error(tmp_path):
    from llm_provider import create_llm_provider
    from schemas import MissingCredentialsError

    with pytest.raises(MissingCredentialsError, match="GEMINI_API_KEY"):
        create_llm_provider(load(tmp_path=tmp_path))
