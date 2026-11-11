"""A chat assistant that answers tram questions by calling ``chat_tools`` (roadmap E18).

The LLM runs at any OpenAI-compatible chat completions API; the default is Groq's free
tier (``[chat]`` in configs/config.toml, key in the ``GROQ_API_KEY`` environment variable).
The model never sees the data directly: it picks tools and arguments, ``chat_tools``
runs them, and the model writes the answer from their results.

``python -m tram_mlops.chat "When is the next tram at Lasipalatsi?"`` asks one question
from the command line and prints the answer and the tools used.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from tram_mlops import chat_tools
from tram_mlops.config import load_config

USER_AGENT = "tram-mlops-chat (+https://github.com/girimohan/helsinki-tram-mlops)"

SYSTEM_PROMPT = """\
You answer questions about Helsinki's trams on the "Helsinki Tram Delays" dashboard, \
using live and historical data this project collects from HSL.
It is now {now:%A %d %B %Y, %H:%M} in Helsinki.

- Get every fact from the tools: arrival times, delays, alerts, punctuality, forecasts. \
Never guess a time or a number. If a tool returns an error or nothing, say so plainly.
- Stop names are Finnish, as HSL writes them. Turn English or Swedish place names into the \
Finnish stop name before calling a tool (Railway Station = Rautatieasema, Glass Palace = \
Lasipalatsi, Kamppi = Kampintori). If a stop isn't found, try the similar names the tool suggests.
- Times are Helsinki time on the 24-hour clock. For upcoming trams give minutes away and the \
clock time, the line and where it is heading.
- Tools give delays in minutes, positive = late, negative = early. Write them in words: \
"2 min late", "1 min early", "on time" (between 1 min early and 3 min late, HSL's definition); \
never as a signed number.
- Arrival times are HSL's own predictions. The 5-minute delay forecasts come from this project's model.
- Quote alerts faithfully: their own text and the "until" time the tool gives.
- Be brief: a sentence or a short list.
- Reply in the language the user wrote in. An English question gets an English answer, even though \
stop names and tool results are in Finnish. Keep stop names as they are.
- Only discuss Helsinki public transport and this project's data; politely decline anything else."""


class LLMError(Exception):
    """The chat API failed or returned something unusable."""


class RateLimited(LLMError):
    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ChatClient(Protocol):
    def complete(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 tool_choice: str = "auto") -> dict[str, Any]: ...


@dataclass
class Settings:
    base_url: str
    model: str
    fallback_model: str
    api_key: str
    reasoning_effort: str = ""
    temperature: float = 0.2
    max_tool_rounds: int = 4
    max_tool_result_chars: int = 6000
    history_messages: int = 6
    max_question_chars: int = 500
    questions_per_session: int = 30
    questions_per_minute: int = 6
    questions_per_day: int = 300
    timeout_s: float = 30

    @classmethod
    def from_config(cls) -> Settings:
        cfg = dict(load_config()["chat"])
        cfg["api_key"] = os.environ.get(cfg.pop("api_key_env"), "").strip()
        return cls(**cfg)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class OpenAICompatibleClient:
    """POST /chat/completions with the standard library; no SDK needed."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 tool_choice: str = "auto") -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                                "temperature": self.settings.temperature, "max_completion_tokens": 1024}
        if self.settings.reasoning_effort:
            body["reasoning_effort"] = self.settings.reasoning_effort
        request = urllib.request.Request(
            self.settings.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json",
                     "User-Agent": USER_AGENT}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            if err.code == 429:
                retry = err.headers.get("retry-after")
                raise RateLimited(f"rate limited: {detail}", float(retry) if retry else None) from err
            raise LLMError(f"HTTP {err.code}: {detail}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            raise LLMError(f"could not reach the chat API: {err}") from err
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError(f"unexpected response: {str(data)[:300]}") from err


class Quota:
    """Questions allowed across all visitors per minute and per day (the free API tier is shared)."""

    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.time):
        self.per_minute, self.per_day, self.clock = per_minute, per_day, clock
        self._asked: deque[float] = deque()
        self._lock = threading.Lock()

    def take(self) -> str | None:
        """Count one question, or say why it can't be asked now."""
        with self._lock:
            now = self.clock()
            while self._asked and self._asked[0] <= now - 86400:
                self._asked.popleft()
            if len(self._asked) >= self.per_day:
                return "The assistant has answered its daily quota of questions. Please try again tomorrow."
            if sum(1 for t in self._asked if t > now - 60) >= self.per_minute:
                return "Lots of questions right now. Please try again in a minute."
            self._asked.append(now)
            return None


@dataclass
class ToolUse:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Reply:
    text: str
    tools: list[ToolUse] = field(default_factory=list)
    model: str | None = None


class Assistant:
    def __init__(self, client: ChatClient, data: chat_tools.TramData, settings: Settings):
        self.client, self.data, self.settings = client, data, settings

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(now=self.data.clock().tz_convert(self.data.tz))

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> Reply:
        """Answer ``question``; ``history`` is the earlier ``{"role", "content"}`` turns."""
        question = question.strip()[: self.settings.max_question_chars]
        earlier = [{"role": m["role"], "content": m["content"]} for m in (history or [])
                   if m.get("role") in ("user", "assistant") and m.get("content")]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            *earlier[-self.settings.history_messages:],
            {"role": "user", "content": question},
        ]
        used: list[ToolUse] = []
        model = None
        for round_ in range(self.settings.max_tool_rounds + 1):
            # After the last allowed round of lookups, the model must answer with what it has.
            last = round_ == self.settings.max_tool_rounds
            message, model = self._complete(messages, "none" if last else "auto")
            calls = message.get("tool_calls") or []
            if not calls or last:
                text = (message.get("content") or "").strip()
                return Reply(text or "Sorry, I couldn't put an answer together. Please try rephrasing.", used, model)
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
                for c in calls]})
            for call in calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"].get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError
                except ValueError:
                    arguments, result = {}, {"error": "Arguments were not a JSON object."}
                else:
                    result = chat_tools.call_tool(self.data, name, arguments)
                used.append(ToolUse(name, arguments, result))
                content = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
                if len(content) > self.settings.max_tool_result_chars:
                    content = content[: self.settings.max_tool_result_chars] + " ...(truncated)"
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        raise AssertionError("unreachable")  # the last round always returns

    def _complete(self, messages: list[dict[str, Any]], tool_choice: str) -> tuple[dict[str, Any], str]:
        """One completion, falling back to the second model if the first is rate limited or fails."""
        models = [m for m in (self.settings.model, self.settings.fallback_model) if m]
        error: LLMError | None = None
        for model in models:
            for attempt in range(2):
                try:
                    return self.client.complete(model, messages, chat_tools.TOOLS, tool_choice), model
                except RateLimited as exc:
                    error = exc
                    break  # each model has its own limits: try the next one
                except LLMError as exc:
                    error = exc
                    # A malformed tool call (Groq: "tool_use_failed") often succeeds on a second try.
                    if attempt == 0 and "tool_use_failed" in str(exc):
                        continue
                    break
        assert error is not None
        raise error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    args = parser.parse_args()
    settings = Settings.from_config()
    if not settings.configured:
        raise SystemExit("Set the API key first, e.g. GROQ_API_KEY=... (see [chat] in configs/config.toml).")
    assistant = Assistant(OpenAICompatibleClient(settings), chat_tools.TramData.from_config(), settings)
    started = time.monotonic()
    reply = assistant.ask(args.question)
    for use in reply.tools:
        print(f"[{use.name}({json.dumps(use.arguments, ensure_ascii=False)})]")
    print(reply.text)
    print(f"({reply.model}, {time.monotonic() - started:.1f} s, at {pd.Timestamp.now(tz='UTC'):%H:%M:%S} UTC)")


if __name__ == "__main__":
    main()
