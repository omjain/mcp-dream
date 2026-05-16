"""MCP server: exposes dream tools to any MCP-compatible client.

Tools exposed:
  - dream_start(goal, duration_hours?, max_dollars?)  → kicks off a dream
  - dream_status(dream_id?)                           → check on a running dream
  - dream_list()                                      → list past dreams
  - dream_read_report(dream_id)                       → pull report into chat
  - dream_stop(dream_id)                              → kill switch

Dreams run in background threads. State is persisted to disk under
~/.mcp-dream/dreams/<id>/ so even if the server restarts, past reports
remain readable.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import load_config, BudgetConfig
from .daemon import DreamRun, dream


mcp = FastMCP("mcp-dream")

# In-memory registry of running/finished dreams keyed by dream_id.
# Persistence lives on disk; this is just for fast status lookups within
# a single server lifetime.
_RUNS: dict[str, DreamRun] = {}
_RUNS_LOCK = threading.Lock()
# Per-dream "stop requested" flags. The dream loop doesn't currently check
# these, but we record the request so dream_stop is honest about what it does.
_STOP_REQUESTS: dict[str, bool] = {}


def _spawn_dream(goal: str, overrides: dict | None = None) -> str:
    """Launch a dream in a background thread. Returns the dream_id immediately."""
    cfg = load_config()

    # Apply per-call overrides for budget caps (callers can ask for shorter dreams).
    if overrides:
        cfg.budget = BudgetConfig(
            max_hours=float(overrides.get("max_hours", cfg.budget.max_hours)),
            max_dollars=float(overrides.get("max_dollars", cfg.budget.max_dollars)),
            max_iterations=int(overrides.get("max_iterations", cfg.budget.max_iterations)),
        )

    dream_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _target():
        try:
            run = dream(cfg, goal, dream_id=dream_id)
        except Exception as e:  # noqa: BLE001
            # Last-resort: record an error stub run so status calls don't return nothing.
            run = DreamRun(
                dream_id=dream_id,
                goal=goal,
                started_at=datetime.now(),
                dream_dir=cfg.dreams_dir / dream_id,
            )
            run.status = "error"
            run.error = str(e)
        with _RUNS_LOCK:
            _RUNS[dream_id] = run

    # Register a placeholder run immediately so status checks work right away.
    with _RUNS_LOCK:
        _RUNS[dream_id] = DreamRun(
            dream_id=dream_id,
            goal=goal,
            started_at=datetime.now(),
            dream_dir=cfg.dreams_dir / dream_id,
        )
        _RUNS[dream_id].status = "starting"

    t = threading.Thread(target=_target, daemon=True, name=f"dream-{dream_id}")
    t.start()
    return dream_id


@mcp.tool()
def dream_start(
    goal: str,
    max_hours: float | None = None,
    max_dollars: float | None = None,
    max_iterations: int | None = None,
) -> str:
    """Start a dream. The agent will explore your goal in the background until
    a budget cap is hit, then write a morning report.

    Args:
        goal: What you want the AI to dream about. Be vague or specific —
              "surprise me with ideas about X" works as well as a specific
              question.
        max_hours: Optional override for the time cap (defaults to config).
        max_dollars: Optional override for the spend cap.
        max_iterations: Optional override for the iteration cap.
    """
    overrides = {}
    if max_hours is not None:
        overrides["max_hours"] = max_hours
    if max_dollars is not None:
        overrides["max_dollars"] = max_dollars
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations

    dream_id = _spawn_dream(goal, overrides=overrides or None)
    return (
        f"Dream started 🌙\n"
        f"dream_id: {dream_id}\n"
        f"goal: {goal}\n\n"
        f"Use dream_status('{dream_id}') to check on it, or dream_read_report('{dream_id}') "
        f"once it's done. Sweet dreams."
    )


@mcp.tool()
def dream_status(dream_id: str | None = None) -> str:
    """Check on a dream. If no dream_id is given, returns the most recent."""
    with _RUNS_LOCK:
        if not _RUNS:
            return "No dreams in flight or in memory."
        if dream_id is None:
            dream_id = max(_RUNS.keys())
        run = _RUNS.get(dream_id)

    if run is None:
        return f"No dream with id {dream_id} found in memory. (Check ~/.mcp-dream/dreams/ for past reports.)"

    parts = [
        f"dream_id: {run.dream_id}",
        f"status: {run.status}",
        f"started_at: {run.started_at.isoformat(timespec='seconds')}",
        f"goal: {run.goal}",
        f"notes_so_far: {len(run.scratch_notes)}",
    ]
    if run.tracker is not None:
        parts.append(f"stats: {json.dumps(run.tracker.summary())}")
    if run.stop_reason:
        parts.append(f"stop_reason: {run.stop_reason}")
    if run.error:
        parts.append(f"error: {run.error}")
    if run.report_path:
        parts.append(f"report: {run.report_path}")
    return "\n".join(parts)


@mcp.tool()
def dream_list() -> str:
    """List recent dreams (from disk, not just memory)."""
    cfg = load_config()
    if not cfg.dreams_dir.exists():
        return "No dreams yet."
    dirs = sorted([p for p in cfg.dreams_dir.iterdir() if p.is_dir()], reverse=True)
    if not dirs:
        return "No dreams yet."
    out = []
    for d in dirs[:20]:
        meta = d / "metadata.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                goal = m.get("goal", "(unknown goal)")
                stop = m.get("stop_reason", "?")
                out.append(f"- {d.name}  [{stop}]  {goal[:80]}")
            except Exception:  # noqa: BLE001
                out.append(f"- {d.name}  (metadata unreadable)")
        else:
            out.append(f"- {d.name}  (in progress?)")
    return "\n".join(out)


@mcp.tool()
def dream_read_report(dream_id: str) -> str:
    """Read the morning report for a finished dream."""
    cfg = load_config()
    report_path = cfg.dreams_dir / dream_id / "report.md"
    if not report_path.exists():
        # Maybe still running.
        with _RUNS_LOCK:
            run = _RUNS.get(dream_id)
        if run and run.status != "done":
            return (
                f"Dream {dream_id} is still {run.status}. "
                f"Notes so far: {len(run.scratch_notes)}. Try again later."
            )
        return f"No report found for dream_id {dream_id}."
    return report_path.read_text()


@mcp.tool()
def dream_stop(dream_id: str) -> str:
    """Request a dream to stop. (Currently advisory — the dream finishes its
    current LLM turn before honoring the request.)"""
    with _RUNS_LOCK:
        if dream_id not in _RUNS:
            return f"No active dream with id {dream_id}."
        _STOP_REQUESTS[dream_id] = True
        run = _RUNS[dream_id]
        # The loop checks budget caps, not stop requests directly. As a
        # cheap-but-effective stop, we zero out the iteration cap on the
        # live tracker so the next check() returns "iterations".
        if run.tracker is not None:
            run.tracker.config.max_iterations = run.tracker.iterations
    return f"Stop requested for dream {dream_id}. It will end after the current turn."


def main() -> None:
    """Entry point: `mcp-dream-server`."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
