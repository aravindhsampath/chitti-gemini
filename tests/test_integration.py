"""Integration test — requires GEMINI_API_KEY in .env. Skipped if not set."""

import asyncio
import os
import tomllib
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from chitti.client import ChittiClient


pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)

BASE_DIR = Path(__file__).parent.parent


@pytest.fixture
def config():
    with open(BASE_DIR / "chitti.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture
def soul():
    return (BASE_DIR / "SOUL.md").read_text()


@pytest.fixture
def client(config, soul):
    return ChittiClient(
        api_key=os.environ["GEMINI_API_KEY"],
        config=config,
        soul=soul,
    )


@pytest.mark.asyncio
async def test_streaming_roundtrip(client):
    """One streaming round-trip: send a prompt, get text back."""
    text_parts = []
    got_complete = False

    async for chunk in client.send("Reply with exactly: TEST_PASS"):
        if chunk.text:
            text_parts.append(chunk.text)
        if chunk.is_complete:
            got_complete = True

    response = "".join(text_parts)
    assert "TEST_PASS" in response
    assert got_complete
    assert client.previous_interaction_id is not None
    assert client.session_usage.interaction_count == 1
    assert client.session_usage.total_tokens > 0
