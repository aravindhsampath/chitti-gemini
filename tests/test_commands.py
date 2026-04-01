"""Tests for REPL command handling."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from chitti.client import ChittiClient, SessionUsage
from chitti.__main__ import handle_command


@pytest.fixture
def mock_client():
    client = MagicMock(spec=ChittiClient)
    client.model = "gemini-3.1-flash-lite-preview"
    client.session_usage = SessionUsage()
    return client


@pytest.fixture
def config():
    return {
        "model": {"name": "gemini-3.1-flash-lite-preview"},
        "generation": {"temperature": 1.0, "max_output_tokens": 8192, "thinking_level": "low"},
        "assistant": {"soul_file": "SOUL.md"},
        "pricing": {"input_per_million": 0.25, "output_per_million": 1.50},
        "display": {"show_thinking": False},
    }


def test_new_command(mock_client, config):
    result = handle_command("/new", mock_client, config)
    assert result is True
    mock_client.new_conversation.assert_called_once()


def test_model_command(mock_client, config):
    result = handle_command("/model", mock_client, config)
    assert result is True


def test_usage_command(mock_client, config):
    result = handle_command("/usage", mock_client, config)
    assert result is True


def test_help_command(mock_client, config):
    result = handle_command("/help", mock_client, config)
    assert result is True


def test_unknown_command(mock_client, config):
    result = handle_command("/foobar", mock_client, config)
    assert result is True  # still handled (prints error)
