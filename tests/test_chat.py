import io
import json
import urllib.error

import pandas as pd
import pytest

from tram_mlops import chat
from tram_mlops.config import load_config

NOW = pd.Timestamp("2026-11-11T10:00:00Z")


def settings(**overrides):
    values = {"base_url": "https://llm.invalid/v1", "model": "big", "fallback_model": "small", "api_key": "k"}
    values.update(overrides)
    return chat.Settings(**values)


class StubData:
    """Stands in for chat_tools.TramData: the tools just echo what they were asked."""
    tz = "Europe/Helsinki"

    def clock(self):
        return NOW

    def next_trams(self, stop, line=None, limit=8):
        return {"stop": stop, "arrivals": [{"line": "4", "in_min": 3}]}


def tool_call(name, arguments, id_="c1"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptedClient:
    """Returns the scripted messages (or raises the scripted errors) in turn, recording each request."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def complete(self, model, messages, tools, tool_choice="auto"):
        self.requests.append({"model": model, "messages": json.loads(json.dumps(messages)),
                              "tool_choice": tool_choice})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def test_assistant_calls_tools_and_answers_from_their_results():
    client = ScriptedClient(
        {"role": "assistant", "content": None, "reasoning": "thinking...",
         "tool_calls": [tool_call("next_trams", '{"stop": "Lasipalatsi"}')]},
        {"role": "assistant", "content": "Line 4 arrives in 3 min."},
    )
    reply = chat.Assistant(client, StubData(), settings()).ask("Next tram at Lasipalatsi?")
    assert reply.text == "Line 4 arrives in 3 min." and reply.model == "big"
    assert [(u.name, u.arguments) for u in reply.tools] == [("next_trams", {"stop": "Lasipalatsi"})]

    system, user = client.requests[0]["messages"]
    assert system["role"] == "system" and "Wednesday 11 November 2026, 12:00" in system["content"]
    assert user == {"role": "user", "content": "Next tram at Lasipalatsi?"}
    # The tool's result goes back to the model; the model's own extra fields (reasoning) do not.
    _, _, assistant, result = client.requests[1]["messages"]
    assert set(assistant) == {"role", "content", "tool_calls"}
    assert result["role"] == "tool" and result["tool_call_id"] == "c1"
    assert json.loads(result["content"])["arrivals"] == [{"line": "4", "in_min": 3}]


def test_model_must_answer_after_the_last_round_of_lookups():
    looking = {"role": "assistant", "content": "", "tool_calls": [tool_call("next_trams", '{"stop": "x"}')]}
    client = ScriptedClient(looking, looking, {"role": "assistant", "content": "Here is what I found."})
    reply = chat.Assistant(client, StubData(), settings(max_tool_rounds=2)).ask("?")
    assert [r["tool_choice"] for r in client.requests] == ["auto", "auto", "none"]
    assert reply.text == "Here is what I found." and len(reply.tools) == 2


def test_bad_tool_arguments_and_unknown_tools_are_reported_to_the_model():
    client = ScriptedClient(
        {"role": "assistant", "content": "", "tool_calls": [tool_call("next_trams", "{not json", "a"),
                                                            tool_call("rm_rf", "{}", "b")]},
        {"role": "assistant", "content": "Sorry."},
    )
    reply = chat.Assistant(client, StubData(), settings()).ask("?")
    assert [u.result["error"] for u in reply.tools] == ["Arguments were not a JSON object.", "Unknown tool 'rm_rf'."]


def test_rate_limited_model_falls_back_to_the_second():
    client = ScriptedClient(chat.RateLimited("429"), {"role": "assistant", "content": "Hi."})
    reply = chat.Assistant(client, StubData(), settings()).ask("?")
    assert [r["model"] for r in client.requests] == ["big", "small"] and reply.model == "small"

    both_limited = ScriptedClient(chat.RateLimited("429"), chat.RateLimited("429"))
    with pytest.raises(chat.RateLimited):
        chat.Assistant(both_limited, StubData(), settings()).ask("?")


def test_malformed_tool_call_is_retried_once_on_the_same_model():
    client = ScriptedClient(chat.LLMError('HTTP 400: {"code": "tool_use_failed"}'),
                            {"role": "assistant", "content": "Done."})
    assert chat.Assistant(client, StubData(), settings()).ask("?").text == "Done."
    assert [r["model"] for r in client.requests] == ["big", "big"]


def test_history_is_trimmed_and_questions_are_capped():
    client = ScriptedClient({"role": "assistant", "content": "ok"})
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    history.append({"role": "assistant", "content": "", "tools": []})  # nothing to send
    chat.Assistant(client, StubData(), settings(history_messages=4, max_question_chars=5)).ask("abcdefgh", history)
    messages = client.requests[0]["messages"]
    assert [m["content"] for m in messages[1:]] == ["m6", "m7", "m8", "m9", "abcde"]


def test_quota_limits_questions_per_minute_and_per_day():
    now = [0.0]
    quota = chat.Quota(per_minute=2, per_day=3, clock=lambda: now[0])
    assert quota.take() is None and quota.take() is None
    assert "minute" in quota.take()
    now[0] = 61
    assert quota.take() is None
    now[0] = 200
    assert "tomorrow" in quota.take()
    now[0] = 86_401
    assert quota.take() is None


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_posts_an_openai_style_request(monkeypatch):
    sent = {}

    def urlopen(request, timeout):
        sent["url"], sent["auth"] = request.full_url, request.get_header("Authorization")
        sent["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}).encode())
    monkeypatch.setattr(chat.urllib.request, "urlopen", urlopen)
    client = chat.OpenAICompatibleClient(settings(reasoning_effort="low"))
    assert client.complete("big", [{"role": "user", "content": "?"}], [])["content"] == "hi"
    assert sent["url"] == "https://llm.invalid/v1/chat/completions" and sent["auth"] == "Bearer k"
    assert sent["body"]["model"] == "big" and sent["body"]["reasoning_effort"] == "low"


def test_client_turns_429_into_rate_limited(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"retry-after": "7"},
                                     io.BytesIO(b'{"error": "slow down"}'))
    monkeypatch.setattr(chat.urllib.request, "urlopen", urlopen)
    with pytest.raises(chat.RateLimited) as err:
        chat.OpenAICompatibleClient(settings()).complete("big", [], [])
    assert err.value.retry_after_s == 7


def test_settings_read_the_key_from_the_environment(monkeypatch):
    load_config.cache_clear()
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert not chat.Settings.from_config().configured
    monkeypatch.setenv("GROQ_API_KEY", " gsk_test ")
    assert chat.Settings.from_config().api_key == "gsk_test"
    load_config.cache_clear()
