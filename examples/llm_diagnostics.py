#!/usr/bin/env python
"""
llm_diagnostics_fixed.py – capability-by-capability test runner
==============================================================
Exercises **text → stream → tools → stream_tools → vision** for every provider
registered in chuk-llm.  Prints a live ✅/❌ ticker and then timing/error/summary
tables (Rich if available, plain text otherwise).

Key features
------------
* **No heuristics** – simply tries each capability and records success (✅),
  failure (❌) or “—” (capability skipped / not supported).
* **Per-capability model overrides** via `--model`, e.g.

      --model "
          openai:text=gpt-4o-mini,
          openai:vision=gpt-4o-vision,
          ollama:vision=llama3.2-vision
      "

  Omitting `:capability` means the model is used for *all* tests on that
  provider.
* **Skip flags**: `--skip-streaming`, `--skip-tools`, `--skip-image`.

Run
---
    uv run examples/llm_diagnostics_fixed.py
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

# ───────────────────── optional Rich UI ──────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console: Console | None = Console()
    _RICH = True
except ModuleNotFoundError:  # pragma: no cover
    console = None
    _RICH = False

# ─────────────── chuk-llm imports (local package) ──────────────
from chuk_llm.llm_client import get_llm_client
from chuk_llm.provider_config import DEFAULTS, ProviderConfig

# ─────────────────────── prompts & assets ──────────────────────
TEXT_PROMPT     = "Why is testing LLM providers important? (3–4 sentences)"
FUNCTION_PROMPT = "What is the weather in London? Use get_weather."
VISION_PROMPT   = "Describe what you see."

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

# 1 × 1 px white PNG (tiny & always valid)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
)

_CAPS: list[tuple[str, str]] = [            # (CLI-name , dataclass attribute)
    ("text",         "text_completion"),
    ("stream",       "streaming_text"),
    ("tools",        "function_call"),
    ("stream_tools", "streaming_function_call"),
    ("vision",       "vision"),
]

# ───────────────────── result container ───────────────────────
@dataclass
class ProviderResult:
    provider: str
    models: Dict[str, str]       # capability → model

    text_completion:           Optional[bool] = None
    streaming_text:            Optional[bool] = None
    function_call:             Optional[bool] = None
    streaming_function_call:   Optional[bool] = None
    vision:                    Optional[bool] = None

    errors:   Dict[str, str]  = field(default_factory=dict)
    timings:  Dict[str, float] = field(default_factory=dict)

    def record(self, attr: str, value: Optional[bool]) -> None:
        setattr(self, attr, value)

    @property
    def feature_set(self) -> Set[str]:
        feats: Set[str] = set()
        if self.text_completion:           feats.add("text")
        if self.streaming_text:            feats.add("streaming")
        if self.function_call:             feats.add("tools")
        if self.streaming_function_call:   feats.add("streaming_tools")
        if self.vision:                    feats.add("vision")
        return feats

# ─────────────── model-override CLI parsing ────────────────
_OVERRIDE_RE = re.compile(
    r"^(?P<prov>[\w-]+)(:(?P<cap>text|vision|tools))?=(?P<model>.+)$", re.I
)

def parse_overrides(arg: str) -> Dict[str, Dict[str, str]]:
    """
    \"openai:text=gpt-4o,openai:vision=gpt-4o-vision\" → {\"openai\": {\"text\": \"gpt-4o\", ...}}
    """
    mapping: Dict[str, Dict[str, str]] = {}
    for frag in arg.split(","):
        frag = frag.strip()
        if not frag:
            continue
        m = _OVERRIDE_RE.match(frag)
        if not m:
            raise ValueError(f"Invalid --model fragment: '{frag}'")
        prov = m.group("prov").lower()
        cap  = m.group("cap") or "*"
        mapping.setdefault(prov, {})[cap] = m.group("model")
    return mapping

# ───────────────────────── async helpers ─────────────────────────
async def _timed(res: ProviderResult, key: str, awaitable):
    start = time.perf_counter()
    try:
        return await awaitable
    finally:
        res.timings[key] = time.perf_counter() - start

async def _get_client(cache: Dict[Tuple[str, str], Any], provider: str, model: str):
    key = (provider, model)
    if key not in cache:
        cache[key] = get_llm_client(provider=provider, model=model)
    return cache[key]

def _yn(val: Optional[bool]) -> str:
    return {True: "✅", False: "❌", None: "—"}[val]

# ───────────── capability check coroutines ──────────────
async def _check_text(res, cache, tick):
    try:
        cl  = await _get_client(cache, res.provider, res.models["text"])
        out = await _timed(res, "text",
                           cl.create_completion([{"role": "user", "content": TEXT_PROMPT}]))
        ok = bool(out.get("response"))
        res.record("text_completion", ok); tick("text", ok)
    except Exception as exc:
        res.record("text_completion", False)
        res.errors["text"] = str(exc); tick("text", False)

async def _check_stream(res, cache, tick):
    try:
        cl = await _get_client(cache, res.provider, res.models["text"])
        stream = await _timed(res, "stream",
                              cl.create_completion([{"role": "user", "content": TEXT_PROMPT}],
                                                   stream=True))
        if not hasattr(stream, "__aiter__"):
            raise TypeError("stream returned non-async iterator")
        async for ch in stream:
            if isinstance(ch, dict) and ch.get("response"):
                res.record("streaming_text", True); break
        else:
            res.record("streaming_text", False)
        tick("stream", res.streaming_text)
    except Exception as exc:
        res.record("streaming_text", False)
        res.errors["stream"] = str(exc); tick("stream", False)

async def _check_tools(res, cache, tick):
    try:
        cl = await _get_client(cache, res.provider, res.models["tools"])
        out = await _timed(res, "tools",
                           cl.create_completion([{"role": "user", "content": FUNCTION_PROMPT}],
                                                tools=[_WEATHER_TOOL]))
        ok = bool(out.get("tool_calls"))
        res.record("function_call", ok); tick("tools", ok)
    except Exception as exc:
        if "does not support tools" in str(exc).lower():
            res.record("function_call", None); tick("tools", None)
        else:
            res.record("function_call", False)
            res.errors["tools"] = str(exc); tick("tools", False)

async def _check_stream_tools(res, cache, tick):
    try:
        cl = await _get_client(cache, res.provider, res.models["tools"])
        stream = await _timed(res, "stream_tools",
                              cl.create_completion([{"role": "user", "content": FUNCTION_PROMPT}],
                                                   tools=[_WEATHER_TOOL], stream=True))
        if not hasattr(stream, "__aiter__"):
            raise TypeError("stream_tools non-async iterator")
        async for ch in stream:
            if isinstance(ch, dict) and ch.get("tool_calls"):
                res.record("streaming_function_call", True); break
        else:
            res.record("streaming_function_call", False)
        tick("stream_tools", res.streaming_function_call)
    except Exception as exc:
        if "does not support tools" in str(exc).lower():
            res.record("streaming_function_call", None); tick("stream_tools", None)
        else:
            res.record("streaming_function_call", False)
            res.errors["stream_tools"] = str(exc); tick("stream_tools", False)

async def _check_vision(res, cache, tick):
    try:
        cl   = await _get_client(cache, res.provider, res.models["vision"])
        prov = res.provider.lower()
        img  = _TINY_PNG_B64

        # provider-specific multimodal message structures
        if prov == "ollama":
            msg = {"role": "user", "content": VISION_PROMPT,
                   "images": [base64.b64decode(img)]}
        elif prov in {"anthropic", "claude"}:
            msg = {"role": "user", "content": [
                {"type": "text",   "text": VISION_PROMPT},
                {"type": "image",  "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": img}},
            ]}
        else:  # OpenAI, Gemini, Groq etc.
            msg = {"role": "user", "content": [
                {"type": "text",      "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}" }},
            ]}

        out = await _timed(res, "vision", cl.create_completion([msg]))
        ok  = bool(out.get("response"))
        res.record("vision", ok); tick("vision", ok)
    except Exception as exc:
        res.record("vision", False)
        res.errors["vision"] = str(exc); tick("vision", False)

# ─────────────── per-provider orchestrator ────────────────
async def run_provider(provider: str, models: Dict[str, str], args) -> ProviderResult:
    res   = ProviderResult(provider, models)
    cache: Dict[Tuple[str, str], Any] = {}

    prefix = f"[{provider}]"
    tick   = lambda stage, ok: print(f"{prefix} {stage:<12} {_yn(ok)}")  # noqa: E731

    await _check_text(res, cache, tick)

    if not args.skip_streaming:
        await _check_stream(res, cache, tick)
    else:
        res.record("streaming_text", None)

    if not args.skip_tools:
        await _check_tools(res, cache, tick)
        if not args.skip_streaming:
            await _check_stream_tools(res, cache, tick)
        else:
            res.record("streaming_function_call", None)
    else:
        res.record("function_call", None)
        res.record("streaming_function_call", None)

    if not args.skip_image:
        await _check_vision(res, cache, tick)
    else:
        res.record("vision", None)

    return res

# ─────────────────────── table rendering ───────────────────────
def _render(results: List[ProviderResult]):
    if console is None:            # simple fallback
        for r in results:
            feats = ", ".join(sorted(r.feature_set)) or "—"
            print(f"{r.provider:<10} → {feats}")
        return

    # timing table
    t = Table(title="Execution Time (s)")
    t.add_column("Provider")
    for col in ("Text", "Stream", "Tools", "Str.Tools", "Vision"):
        t.add_column(col)
    for r in results:
        t.add_row(
            r.provider,
            f"{r.timings.get('text', 0):.2f}"         if 'text'         in r.timings else "—",
            f"{r.timings.get('stream', 0):.2f}"       if 'stream'       in r.timings else "—",
            f"{r.timings.get('tools', 0):.2f}"        if 'tools'        in r.timings else "—",
            f"{r.timings.get('stream_tools', 0):.2f}" if 'stream_tools' in r.timings else "—",
            f"{r.timings.get('vision', 0):.2f}"       if 'vision'       in r.timings else "—",
        )
    console.print(t)

    # error table
    errs = [(r.provider, stage, msg.splitlines()[0][:120])
            for r in results for stage, msg in r.errors.items()]
    if errs:
        et = Table(title="Errors")
        et.add_column("Provider"); et.add_column("Stage"); et.add_column("Message")
        for p, s, m in errs:
            et.add_row(p, s, m)
        console.print(et)

    # summary table
    s = Table(title="LLM Provider Diagnostics – Final")
    s.add_column("Provider", style="cyan")
    s.add_column("Model(s)", style="blue")
    for col in ("Text", "Stream", "Tools", "Str.Tools", "Vision"):
        s.add_column(col)
    s.add_column("Features", style="magenta")

    for r in results:
        model_summary = ", ".join(f"{cap}:{mdl}" for cap, mdl in r.models.items())
        s.add_row(
            r.provider,
            model_summary,
            _yn(r.text_completion),
            _yn(r.streaming_text),
            _yn(r.function_call),
            _yn(r.streaming_function_call),
            _yn(r.vision),
            ", ".join(sorted(r.feature_set)) or "—",
        )
    console.print(s)

# ─────────────────────────── main() ────────────────────────────
def build_models_for_provider(
    prov: str,
    overrides: Dict[str, Dict[str, str]],
    cfg: ProviderConfig,
) -> Dict[str, str]:
    """
    Decide which model each capability uses for *prov*, respecting overrides.
    """
    prov_lc   = prov.lower()
    prov_ov   = overrides.get(prov_lc, {})
    default_m = cfg.get_default_model(prov)

    models: Dict[str, str] = {}
    for cap, _ in _CAPS:
        models[cap] = prov_ov.get(cap) or prov_ov.get("*") or default_m
    return models

def parse_args():
    p = argparse.ArgumentParser("LLM diagnostics")
    p.add_argument("--providers", nargs="*", help="Only test these providers")
    p.add_argument("--model", help="override model(s) e.g. 'openai:text=g4,ollama=llama3'")
    p.add_argument("--skip-streaming", action="store_true", help="skip all streaming tests")
    p.add_argument("--skip-tools",    action="store_true", help="skip function-calling tests")
    p.add_argument("--skip-image",    action="store_true", help="skip vision tests")
    return p.parse_args()

async def main():
    args = parse_args()
    cfg  = ProviderConfig()

    providers = [p for p in DEFAULTS if p != "__global__"]
    if args.providers:
        targets = {p.lower() for p in args.providers}
        providers = [p for p in providers if p.lower() in targets]
        if not providers:
            sys.exit("No matching providers found.")

    overrides = parse_overrides(args.model) if args.model else {}

    results: List[ProviderResult] = []

    if _RICH:
        with Progress(SpinnerColumn(), TextColumn("{task.description}")) as prog:
            task = prog.add_task("Diagnostics", total=len(providers))
            for prov in providers:
                models = build_models_for_provider(prov, overrides, cfg)
                prog.update(task, description=f"Testing {prov}")
                res = await run_provider(prov, models, args)
                results.append(res); prog.advance(task)
    else:
        for prov in providers:
            models = build_models_for_provider(prov, overrides, cfg)
            print(f"== {prov} ==")
            res = await run_provider(prov, models, args)
            results.append(res)

    _render(results)

if __name__ == "__main__":   # pragma: no cover
    asyncio.run(main())
