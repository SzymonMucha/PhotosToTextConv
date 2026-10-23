"""Config tests: .env loading, defaults, validation of missing keys."""

from __future__ import annotations

import pytest

from book2md.config import Settings, load_settings
from book2md.ocr import build_ocr_provider
from book2md.llm import build_llm_provider


def test_defaults_match_spec_env(monkeypatch):
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.ocr_provider == "ocrspace"
    assert settings.book_language == "pl"
    assert settings.ocr_language == "pol"
    assert settings.ocr_concurrency == 4
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_concurrency == 2
    assert settings.chunk_words == 2000
    assert settings.ocr_engine == "2"  # newer OCR.space engine by default


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OCR_CONCURRENCY", "9")
    monkeypatch.setenv("CHUNK_WORDS", "1200")
    monkeypatch.setenv("LLM_MODEL", "test-model-from-env")
    settings = load_settings()
    assert settings.ocr_concurrency == 9
    assert settings.chunk_words == 1200
    assert settings.llm_model == "test-model-from-env"


def test_missing_llm_key_gives_guidance():
    settings = Settings(_env_file=None, llm_api_key="", llm_model="m")
    with pytest.raises(Exception, match="LLM_API_KEY"):
        build_llm_provider(settings)


def test_missing_llm_model_gives_guidance():
    settings = Settings(_env_file=None, llm_api_key="k", llm_model="")
    with pytest.raises(Exception, match="LLM_MODEL"):
        build_llm_provider(settings)


def test_unknown_ocr_provider_rejected():
    settings = Settings(_env_file=None, ocr_api_key="k", ocr_provider="doesnotexist")
    with pytest.raises(Exception, match="Unknown OCR_PROVIDER"):
        build_ocr_provider("doesnotexist", settings)


def test_ocr_provider_missing_key_rejected():
    settings = Settings(_env_file=None, ocr_api_key="")
    with pytest.raises(Exception, match="OCR_API_KEY"):
        build_ocr_provider("ocrspace", settings)
