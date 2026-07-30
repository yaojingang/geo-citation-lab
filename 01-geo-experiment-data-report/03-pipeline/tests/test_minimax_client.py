"""Unit tests for the MiniMax provider client and categorization dispatcher.

These tests run without network access or the optional OpenAI SDK: they stub
the chat-completions surface to verify the dispatcher routes to MiniMax and
degrades gracefully when no client is configured.
"""

import importlib.util
import os
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

_spec = importlib.util.spec_from_file_location(
    "minimax_client", PIPELINE_DIR / "minimax_client.py"
)
minimax_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(minimax_client)


def test_resolve_base_url_defaults_and_regions():
    assert minimax_client.resolve_base_url(None).endswith("/v1")
    assert minimax_client.resolve_base_url("global_en") == "https://api.minimax.io/v1"
    assert minimax_client.resolve_base_url("cn_zh") == "https://api.minimaxi.com/v1"
    # Unknown region falls back to the global endpoint.
    assert minimax_client.resolve_base_url("mars") == "https://api.minimax.io/v1"


def test_resolve_model_selectable():
    assert minimax_client.resolve_model(None) == "MiniMax-M3"
    assert minimax_client.resolve_model("MiniMax-M2.7") == "MiniMax-M2.7"
    assert minimax_client.resolve_model("MiniMax-M3") == "MiniMax-M3"
    # Unknown model falls back to the default.
    assert minimax_client.resolve_model("bogus") == "MiniMax-M3"


def test_build_client_without_key_returns_none(monkeypatch=None):
    os.environ.pop("MINIMAX_API_KEY", None)
    assert minimax_client.build_minimax_client() is None


def _fake_openai_client(content="新闻"):
    """Build an object that mimics openai.OpenAI for chat.completions.create."""

    class Message:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, content):
            self.message = Message(content)

    class Response:
        def __init__(self, content):
            self.choices = [Choice(content)]

    class Completions:
        def create(self, **kwargs):
            assert "messages" in kwargs
            return Response(content)

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    return Client()


def test_categorize_website_routes_to_minimax():
    client = _fake_openai_client(content="新闻")
    result = minimax_client.categorize_website("example.com", (client, "MiniMax-M3"))
    assert result == "新闻"


def test_categorize_website_without_client_returns_null():
    assert minimax_client.categorize_website("example.com", None) == "null"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
