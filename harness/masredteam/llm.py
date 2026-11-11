"""LLM backends for the red-team harness.

Three interchangeable modes behind one interface, selected by MAS_LLM_MODE:

  mock    - deterministic, no network, no keys. Runs end to end today and is
            what the unit tests and the shipped demo results use. The mock is
            *adversarially honest*: it models an agent that follows injected
            instructions unless a defense has neutralised them, so attack
            success and defense success are both real functions of the
            pipeline, not hard-coded outcomes.
  openai  - real OpenAI chat completions (needs OPENAI_API_KEY).
  ollama  - a local model via Ollama (no key, needs `ollama serve`).

Switch with:  export MAS_LLM_MODE=openai   (or ollama, or mock)
"""
from __future__ import annotations
import os, re, json, hashlib, urllib.request


class LLMResult:
    def __init__(self, text: str, raw: dict | None = None):
        self.text = text
        self.raw = raw or {}


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
def get_backend():
    mode = os.environ.get("MAS_LLM_MODE", "mock").lower()
    if mode == "openai":
        return OpenAIBackend()
    if mode == "ollama":
        return OllamaBackend()
    return MockBackend()


# --------------------------------------------------------------------------
# Real backends
# --------------------------------------------------------------------------
class OpenAIBackend:
    """Minimal OpenAI chat wrapper. Kept dependency-light on purpose."""
    def __init__(self):
        self.model = os.environ.get("MAS_OPENAI_MODEL", "gpt-4o-mini")
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("MAS_LLM_MODE=openai but OPENAI_API_KEY is unset.")

    def chat(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return LLMResult(data["choices"][0]["message"]["content"], data)


class OllamaBackend:
    """Local model through Ollama's /api/chat. No API key needed."""
    def __init__(self):
        self.model = os.environ.get("MAS_OLLAMA_MODEL", "llama3.1:8b")
        self.host = os.environ.get("MAS_OLLAMA_HOST", "http://localhost:11434")

    def chat(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return LLMResult(data["message"]["content"], data)


# --------------------------------------------------------------------------
# Mock backend: deterministic agent that actually obeys / resists injections
# --------------------------------------------------------------------------
# The mock reads structured control markers that our agents and attacks place
# in the text. This lets the SAME attack code drive a real model later; only
# the backend changes. The mock's job is to behave like a plausibly-compliant
# agent so that defenses have something real to stop.

# Markers the attacks embed (see attacks/*). The mock treats an ACTIVE marker
# as an instruction the agent will follow. A defense can DEFUSE a marker,
# after which the mock ignores it -- exactly how a real defense would work.
MARK = re.compile(r"\[\[(?P<kind>[A-Z_]+):(?P<payload>.*?)\]\]", re.S)


class MockBackend:
    def chat(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        # Deterministic "reasoning": echo the task, then act on any active
        # control markers found in the user content.
        active = [m.groupdict() for m in MARK.finditer(user)
                  if "DEFUSED" not in m.group(0)]
        role = _role_of(system)
        out = {"role": role, "followed": [], "reasoning": "", "emit": []}

        # Base behaviour: summarise / pass work along.
        out["reasoning"] = f"{role}: handling task."

        for a in active:
            kind, payload = a["kind"], a["payload"]
            if kind == "INJECT":
                # follow a direct instruction
                out["followed"].append("INJECT")
                out["emit"].append(("inject", payload))
            elif kind == "INFECT":
                # follow, and re-emit the infection to the next agent
                out["followed"].append("INFECT")
                out["emit"].append(("infect", payload))
            elif kind == "FAKE_ERROR":
                # a trusted-looking error: the agent relays it as a status report
                out["followed"].append("FAKE_ERROR")
                out["emit"].append(("relay_error", payload))
            elif kind == "MEM_WRITE":
                out["followed"].append("MEM_WRITE")
                out["emit"].append(("mem_write", payload))
            elif kind == "MEM_USE":
                out["followed"].append("MEM_USE")
                out["emit"].append(("mem_use", payload))
            elif kind == "BYZANTINE":
                out["followed"].append("BYZANTINE")
                out["emit"].append(("byzantine", payload))
            elif kind == "TOOL":
                out["followed"].append("TOOL")
                out["emit"].append(("tool", payload))

        # Deterministic jitter so repeated trials are not identical but are
        # reproducible given a seed embedded in the user text.
        seed = _seed_of(user)
        out["nonce"] = seed
        return LLMResult(json.dumps(out), {"mock": True})


def _role_of(system: str) -> str:
    m = re.search(r"ROLE=(\w+)", system or "")
    return m.group(1) if m else "agent"


def _seed_of(text: str) -> int:
    m = re.search(r"SEED=(\d+)", text or "")
    if m:
        return int(m.group(1))
    return int(hashlib.sha256((text or "").encode()).hexdigest(), 16) % 10**6
