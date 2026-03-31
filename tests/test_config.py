"""Tests for configuration loading."""

import os
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

# We test the loading functions directly
BASE_DIR = Path(__file__).parent.parent


def test_chitti_toml_exists():
    assert (BASE_DIR / "chitti.toml").exists()


def test_chitti_toml_parses():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        config = tomllib.load(f)
    assert "model" in config
    assert "generation" in config
    assert "assistant" in config
    assert "display" in config


def test_chitti_toml_model_section():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        config = tomllib.load(f)
    assert config["model"]["name"] == "gemini-3.1-flash-lite-preview"


def test_chitti_toml_generation_section():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        config = tomllib.load(f)
    gen = config["generation"]
    assert isinstance(gen["temperature"], (int, float))
    assert isinstance(gen["max_output_tokens"], int)
    assert gen["thinking_level"] in ("minimal", "low", "medium", "high")


def test_soul_file_exists():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        config = tomllib.load(f)
    soul_path = BASE_DIR / config["assistant"]["soul_file"]
    assert soul_path.exists()
    content = soul_path.read_text()
    assert len(content) > 50, "SOUL.md should have meaningful content"
    assert "Chitti" in content


def test_display_section():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        config = tomllib.load(f)
    assert isinstance(config["display"]["show_usage"], bool)
    assert isinstance(config["display"]["show_thinking"], bool)
