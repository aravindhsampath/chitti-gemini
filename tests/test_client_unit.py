"""Unit tests for ChittiClient — mocked, no API calls."""

import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from client import ChittiClient, SessionUsage, Usage


BASE_DIR = Path(__file__).parent.parent


@pytest.fixture
def config():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture
def soul():
    return "You are a test assistant."


@pytest.fixture
def client(config, soul):
    with patch("client.genai.Client"):
        return ChittiClient(api_key="fake-key", config=config, soul=soul)


def test_client_init(client, config):
    assert client.model == config["model"]["name"]
    assert client.previous_interaction_id is None
    assert client.last_usage is None
    assert client.session_usage.interaction_count == 0


def test_new_conversation(client):
    client._previous_id = "some-id"
    client.new_conversation()
    assert client.previous_interaction_id is None


def test_session_usage_add():
    su = SessionUsage()
    su.add(Usage(input_tokens=10, output_tokens=5, thought_tokens=2, total_tokens=17))
    assert su.total_input == 10
    assert su.total_output == 5
    assert su.total_thought == 2
    assert su.total_tokens == 17
    assert su.interaction_count == 1

    su.add(Usage(input_tokens=20, output_tokens=10, thought_tokens=0, total_tokens=30))
    assert su.total_input == 30
    assert su.total_output == 15
    assert su.total_tokens == 47
    assert su.interaction_count == 2


def test_usage_defaults():
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.thought_tokens == 0
    assert u.total_tokens == 0
