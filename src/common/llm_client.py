"""
Provider-agnostic LLM client.  Only FREE, publicly accessible modes are wired in:

  ollama  local open-weight models (llama3.1, qwen2.5, mistral …); no account,
          no key, no quota.  https://ollama.com   (default when reachable)
  gemini  Google AI Studio free tier (GEMINI_API_KEY from a free account)
  groq    Groq free tier, hosted open-weight Llama models (GROQ_API_KEY)
  mock    deterministic offline stand-in used by the self-test; never use it
          to produce results

A model is addressed as "provider:model", e.g. "ollama:llama3.1:8b",
"gemini:gemini-2.5-flash", "groq:llama-3.3-70b-versatile".

Only the standard library is used for Ollama and Groq (no extra dependency).
"""

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import vocab as V


class LLMError(RuntimeError):
    """Generic provider failure."""


class RateLimited(LLMError):
    """Temporary 429: retry after `wait` seconds."""
    def __init__(self, msg: str, wait: float = 60.0):
        super().__init__(msg)
        self.wait = wait


class QuotaExhausted(LLMError):
    """Daily quota is gone: stop and resume another day."""


DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "mock":   "demo",
}
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def tag(self) -> str:
        """Filesystem-safe identifier used to name annotation folders."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{self.provider}-{self.model}")

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_spec(spec: "str | ModelSpec") -> ModelSpec:
    if isinstance(spec, ModelSpec):
        return spec
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider {provider!r}. Use one of {sorted(DEFAULT_MODELS)} "
            f"as 'provider:model' (e.g. ollama:llama3.1:8b)."
        )
    return ModelSpec(provider, model.strip() or DEFAULT_MODELS[provider])


def _load_env() -> None:
    """
    Load KEY=VALUE pairs from .env into os.environ (existing variables win).
    Pure standard library on purpose: it must work with ANY interpreter, also
    one where python-dotenv is not installed.  Looks in the project root
    (three levels above this file) and in the current working directory.
    """
    root = Path(__file__).resolve().parents[2]
    for path in (root / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.removeprefix("export ").strip()
            value = value.strip().strip("\"'")
            if key and value and not os.environ.get(key):
                os.environ[key] = value


def ollama_models() -> list[str]:
    """Names of locally installed Ollama models ([] if Ollama is not running)."""
    try:
        with urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=1.5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def default_spec() -> ModelSpec:
    """
    Pick a free model automatically.  Order: $LLM_MODEL, local Ollama,
    Gemini free tier, Groq free tier.
    """
    _load_env()
    if os.getenv("LLM_MODEL"):
        return parse_spec(os.environ["LLM_MODEL"])
    installed = ollama_models()
    if installed:
        return ModelSpec("ollama", installed[0])
    if os.getenv("GEMINI_API_KEY"):
        return parse_spec("gemini")
    if os.getenv("GROQ_API_KEY"):
        return parse_spec("groq")
    raise LLMError(
        "No free LLM available. Either install Ollama (https://ollama.com) and run "
        "`ollama pull llama3.1:8b`, or create a free key and put GEMINI_API_KEY "
        "(Google AI Studio) or GROQ_API_KEY (console.groq.com) in .env."
    )


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "stories-we-tell/2.0", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        _raise_for_text(f"HTTP {e.code}: {body}", e.code)
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach {url}: {e.reason}") from e


def _retry_delay(msg: str, fallback: float) -> float:
    m = (re.search(r"retry(?:Delay)?['\"\s:]+(\d+(?:\.\d+)?)s", msg, re.I)
         or re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.I)
         or re.search(r"try again in (\d+(?:\.\d+)?)s", msg, re.I))
    return float(m.group(1)) + 5.0 if m else fallback


def _raise_for_text(msg: str, code: int | None = None):
    low = msg.lower()
    compact = re.sub(r"[^a-z]", "", low)        # "PerDayPerProject", "per_day", "per day" -> perday
    if code == 429 or "429" in low or "resource_exhausted" in low or "rate limit" in low:
        if "perday" in compact or "daily" in low:
            raise QuotaExhausted(msg)
        raise RateLimited(msg, _retry_delay(msg, 60.0))
    raise LLMError(msg)


# ── providers ─────────────────────────────────────────────────────────────────

def _ollama(prompt, model, temperature, seed, json_mode):
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"
    return _post_json(f"{_OLLAMA_URL}/api/generate", payload)["response"]


def _groq(prompt, model, temperature, seed, json_mode):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set (free key: console.groq.com)")
    payload = {
        "model": model, "temperature": temperature, "seed": seed,
        "messages": [{"role": "user", "content": prompt}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    out = _post_json("https://api.groq.com/openai/v1/chat/completions", payload,
                     {"Authorization": f"Bearer {key}"})
    return out["choices"][0]["message"]["content"]


_gemini_client = None


def _gemini(prompt, model, temperature, seed, json_mode):
    global _gemini_client
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set (free key: aistudio.google.com)")
    from google import genai
    from google.genai import types
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        temperature=temperature, seed=seed,
        response_mime_type="application/json" if json_mode else None,
    )
    try:
        resp = _gemini_client.models.generate_content(model=model, contents=prompt, config=cfg)
    except Exception as exc:                        # google.genai.errors.APIError & co.
        _raise_for_text(str(exc), getattr(exc, "code", None))
    return resp.text or ""


def _mock(prompt, model, temperature, seed, json_mode):
    """Deterministic offline stand-in: plausible JSON / prose, no intelligence."""
    h = int(hashlib.md5(f"{prompt}|{seed}|{temperature}".encode()).hexdigest(), 16)
    rng = random.Random(h)
    if "### TASK: RETELL" in prompt:
        body = prompt.split("### STORY", 1)[-1].strip()
        target = re.search(r"TRADITION: (.+)", prompt)
        return (f"[{target.group(1) if target else 'tradition'}] " + body[:1200])
    # ### TASK: ANNOTATE  — keep a story-specific core so retellings stay related
    core = random.Random(int(hashlib.md5(
        prompt.split("### STORY", 1)[-1][:400].encode()).hexdigest(), 16))
    funcs = [core.choice(V.FUNCTIONS) if rng.random() > 0.25 else rng.choice(V.FUNCTIONS)
             for _ in range(V.N_EVENTS)]
    names = ["Hero", "Rival", "Guide", "Elder"]
    events = []
    for i, f in enumerate(funcs, start=1):
        events.append({
            "ID": i, "EVENT": f"event {i} about {names[i % 4]}", "FUNCTION": f,
            "CHARACTERS": ["Hero", names[1 + i % 3]], "LOCATION": f"Place {1 + i % 2}",
            "CAUSE": "Story opening" if i == 1 else f"consequence of event {i - 1}",
            "EFFECT": f"state change {i}", "CAUSED_BY": [] if i == 1 else [i - 1],
        })
    return json.dumps({
        "NARRATIVE_ARC": core.choice(V.ARCS),
        "SETTING_DOMAIN": core.choice(V.SETTING_DOMAINS),
        "CHARACTERS": [
            {"CHARACTER": "Hero", "ROLE": "PROTAGONIST", "TRANSFORMATION": "grows wiser",
             "TRANSFORMATION_TYPE": "GROWTH", "CHANGE_EVENT": 5},
            {"CHARACTER": "Rival", "ROLE": "ANTAGONIST", "TRANSFORMATION": "",
             "TRANSFORMATION_TYPE": "FALL", "CHANGE_EVENT": 4},
            {"CHARACTER": "Guide", "ROLE": "HELPER", "TRANSFORMATION": "",
             "TRANSFORMATION_TYPE": "NONE", "CHANGE_EVENT": None},
        ],
        "LOCATIONS": ["Place 1", "Place 2"],
        "EVENTS": events,
        "RELATIONS": [{"SOURCE": "Rival", "TARGET": "Hero", "TYPE": "OPPOSES"},
                      {"SOURCE": "Guide", "TARGET": "Hero", "TYPE": "HELPS"}],
    })


_PROVIDERS = {"ollama": _ollama, "groq": _groq, "gemini": _gemini, "mock": _mock}


# ── public API ────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    spec: "str | ModelSpec",
    *,
    temperature: float = 0.0,
    seed: int = 0,
    json_mode: bool = False,
    max_retries: int = 5,
) -> str:
    """
    One completion.  Retries 429s with the provider-suggested delay (or an
    exponential back-off); raises QuotaExhausted when the daily quota is gone.
    """
    _load_env()
    spec = parse_spec(spec)
    fn = _PROVIDERS[spec.provider]
    for attempt in range(1, max_retries + 1):
        try:
            return fn(prompt, spec.model, temperature, seed, json_mode).strip()
        except QuotaExhausted:
            raise
        except RateLimited as exc:
            if attempt == max_retries:           # still limited after all retries: stop the run
                raise QuotaExhausted(str(exc)) from exc
            wait = exc.wait * attempt         # provider hint, growing with retries
            print(f"  [RATE LIMIT] attempt {attempt}/{max_retries}, waiting {wait:.0f}s …")
            time.sleep(wait)
        except LLMError:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise LLMError("unreachable")


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates fences / prose)."""
    clean = re.sub(r"```(?:json)?", "", text).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(clean[start:end + 1])
