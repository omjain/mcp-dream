"""The dream loop.

A long-running loop that gives the LLM a goal and a toolset, then lets it
explore until a budget cap fires. The loop is intentionally simple — we
trust the model to drive, we trust the budget to stop it.

Two providers are supported:
  - Anthropic (paid, full tool-use)
  - Ollama  (local/free, simplified — no real tool-use, just iterative prompting)

The Anthropic provider is the "real" one. Ollama is a polite fallback so
the project can be tried without a credit card.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .budget import BudgetTracker, StopReason
from .config import Config
from .report import build_synthesis_prompt, render_pre_report, save_report
from .tools import build_toolset, dispatch


SYSTEM_PROMPT = """\
You are dreaming on behalf of a human who is asleep. They gave you a goal and
went to bed. Your job: explore that goal thoughtfully and leave them a useful
morning report.

How to dream well:
- Don't just answer. Explore. Follow threads. Be curious.
- Use the `note` tool generously to record findings as you go. The report
  is built from your notes, so if you don't write it down, it didn't happen.
- Try multiple angles. If your first search is unhelpful, rephrase and retry.
- It's okay to be wrong or to strike out — say so honestly in your notes.
- Stop when you've explored enough, not when you've found "the answer." The
  user knows real exploration doesn't always end with a bow on top.
- When you feel done, just stop calling tools and write a brief wrap-up.

You will not see the user again until morning. Make their wake-up worth it.
"""


@dataclass
class DreamRun:
    """Captures everything about a single dream run."""

    dream_id: str
    goal: str
    started_at: datetime
    dream_dir: Path
    scratch_notes: list[str] = field(default_factory=list)
    stop_reason: StopReason | None = None
    report_path: Path | None = None
    # Status snapshot for `dream_status` queries.
    status: str = "starting"  # "starting" | "dreaming" | "synthesizing" | "done" | "error"
    error: str | None = None
    tracker: BudgetTracker | None = None


# ── Anthropic provider ───────────────────────────────────────────────────────


def _run_anthropic(cfg: Config, run: DreamRun) -> None:
    """Run a dream using the Anthropic API with full tool-use."""
    from anthropic import Anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either export it or set "
            "provider.name = 'ollama' in config.toml to use a local model."
        )

    client = Anthropic()
    schemas, impls = build_toolset(cfg.tools, run.scratch_notes)
    tracker = BudgetTracker(config=cfg.budget, model=cfg.provider.anthropic_model)
    run.tracker = tracker

    messages: list[dict[str, Any]] = [{"role": "user", "content": run.goal}]
    run.status = "dreaming"

    while True:
        stop = tracker.check()
        if stop is not None:
            run.stop_reason = stop
            break

        try:
            response = client.messages.create(
                model=cfg.provider.anthropic_model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=schemas,
                messages=messages,
            )
        except Exception as e:  # noqa: BLE001
            run.error = f"Anthropic API error: {e}"
            run.stop_reason = "model_stopped"
            break

        # Record usage.
        usage = getattr(response, "usage", None)
        if usage is not None:
            tracker.record_turn(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        else:
            tracker.record_turn()

        # Append assistant message into history (full content blocks).
        messages.append({"role": "assistant", "content": response.content})

        # Did the model stop without tool use?
        if response.stop_reason != "tool_use":
            run.stop_reason = "model_stopped"
            break

        # Run each tool call and gather results.
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                result = dispatch(block.name, dict(block.input), impls)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if not tool_results:
            # Defensive: stop_reason said tool_use but no tool_use blocks present.
            run.stop_reason = "model_stopped"
            break

        messages.append({"role": "user", "content": tool_results})


# ── Ollama provider (no tool-use, simplified loop) ───────────────────────────


def _run_ollama(cfg: Config, run: DreamRun) -> None:
    """A degraded but functional dream loop using a local Ollama model.

    Ollama models in general don't support reliable tool-use, so we run a
    much simpler "iterative reflection" loop: ask the model to think about
    the goal, save notes inline, and we extract them by convention.
    """
    try:
        import ollama  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The ollama provider is selected but the `ollama` package isn't installed. "
            "Install it with: pip install mcp-dream[ollama]"
        ) from e

    tracker = BudgetTracker(config=cfg.budget, model=cfg.provider.ollama_model)
    run.tracker = tracker
    run.status = "dreaming"

    convo = [
        {"role": "system", "content": SYSTEM_PROMPT + (
            "\n\nIMPORTANT: You can't call tools in this mode. Instead, when you "
            "want to save a note, write a line starting with NOTE: on its own. "
            "Lines starting with NOTE: will be extracted into the morning report."
        )},
        {"role": "user", "content": run.goal},
    ]

    client = ollama.Client(host=cfg.provider.ollama_host)
    while True:
        stop = tracker.check()
        if stop is not None:
            run.stop_reason = stop
            break

        try:
            resp = client.chat(model=cfg.provider.ollama_model, messages=convo)
        except Exception as e:  # noqa: BLE001
            run.error = f"Ollama error: {e}"
            run.stop_reason = "model_stopped"
            break

        msg_content = resp["message"]["content"]
        convo.append({"role": "assistant", "content": msg_content})
        tracker.record_turn()  # no token counts from Ollama; just counts iters

        # Extract NOTE: lines.
        for line in msg_content.splitlines():
            line = line.strip()
            if line.lower().startswith("note:"):
                run.scratch_notes.append(line[5:].strip())

        # Nudge the loop forward with a "what next" prompt — Ollama models
        # tend to want to stop early without one.
        convo.append(
            {
                "role": "user",
                "content": (
                    "Keep going. Pull on another thread. Remember to write NOTE: "
                    "lines for anything worth saving. Stop only when you've truly "
                    "explored the goal from multiple angles."
                ),
            }
        )


# ── Synthesis & save ─────────────────────────────────────────────────────────


def _synthesize_report(cfg: Config, run: DreamRun) -> str:
    """Generate the final report using the same provider that drove the dream."""
    stats = run.tracker.summary() if run.tracker else {}
    stop_reason = run.stop_reason or "unknown"
    prompt = build_synthesis_prompt(run.goal, run.scratch_notes, stop_reason, stats)

    if cfg.provider.name == "anthropic":
        try:
            from anthropic import Anthropic

            client = Anthropic()
            resp = client.messages.create(
                model=cfg.provider.anthropic_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text blocks
            chunks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return "\n".join(chunks).strip() or render_pre_report(
                run.goal, run.started_at, stop_reason, stats, run.scratch_notes
            )
        except Exception:
            return render_pre_report(
                run.goal, run.started_at, stop_reason, stats, run.scratch_notes
            )

    # Ollama synthesis
    try:
        import ollama  # type: ignore

        client = ollama.Client(host=cfg.provider.ollama_host)
        resp = client.chat(
            model=cfg.provider.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"].strip() or render_pre_report(
            run.goal, run.started_at, stop_reason, stats, run.scratch_notes
        )
    except Exception:
        return render_pre_report(
            run.goal, run.started_at, stop_reason, stats, run.scratch_notes
        )


# ── Public entry point ───────────────────────────────────────────────────────


def dream(cfg: Config, goal: str, dream_id: str | None = None) -> DreamRun:
    """Run one dream synchronously. Returns the completed DreamRun.

    Blocks until the dream finishes (cap hit, model stopped, or error).
    """
    cfg.ensure_dirs()
    started_at = datetime.now()
    dream_id = dream_id or started_at.strftime("%Y%m%d-%H%M%S")
    dream_dir = cfg.dreams_dir / dream_id

    run = DreamRun(
        dream_id=dream_id,
        goal=goal,
        started_at=started_at,
        dream_dir=dream_dir,
    )

    try:
        if cfg.provider.name == "anthropic":
            _run_anthropic(cfg, run)
        elif cfg.provider.name == "ollama":
            _run_ollama(cfg, run)
        else:
            raise RuntimeError(f"Unknown provider: {cfg.provider.name}")
    except Exception as e:  # noqa: BLE001
        run.error = str(e)
        if run.stop_reason is None:
            run.stop_reason = "model_stopped"

    run.status = "synthesizing"
    report_text = _synthesize_report(cfg, run)

    stats = run.tracker.summary() if run.tracker else {}
    report_path = save_report(
        report_text=report_text,
        dream_dir=dream_dir,
        goal=goal,
        started_at=started_at,
        stop_reason=run.stop_reason or "unknown",
        stats=stats,
        notes=run.scratch_notes,
    )
    run.report_path = report_path
    run.status = "done"
    return run
