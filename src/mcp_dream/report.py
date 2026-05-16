"""Morning report generation.

After the dream loop finishes, this module synthesizes:
  - the user's goal
  - the agent's scratch notes
  - the conversation trace

…into a beautiful markdown report that the user can read in the morning.
The report is the *product* — the dream loop is the engine. We invest here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_PROMPT = """\
You just spent the night dreaming on the user's behalf. Your task: write the
morning report the user will read with their coffee.

The user's goal was:
---
{goal}
---

Your notes from the dream (in chronological order):
---
{notes}
---

Stop summary: {stop_reason}
Stats: {stats}

Write a markdown report with these sections (use these emoji headers exactly):

# 🌙 Morning report

## 🎯 What I dreamed about
One short paragraph restating the user's goal in your own words, so they
remember what they asked for.

## 💡 Ideas I had
Three to five concrete ideas, takes, or hypotheses you came up with. Each
should be 1–3 sentences. Be specific, not generic. If you found nothing
worth saying, say so honestly — fluff erodes trust.

## 🔍 Things I noticed
Two to four observations from your research. Cite URLs or file paths when
relevant. Surprises are gold; flag them.

## ❓ Questions for you
Two or three questions you'd want the user to answer to make the next dream
more useful. These should be specific, not philosophical.

## 📌 What I'd do next
One short paragraph: if I dreamed on this again tomorrow, here's the thread
I'd pull.

Rules:
- Be warm but not saccharine. You're a thoughtful colleague, not a chatbot.
- No corporate-speak. No "I hope this helps."
- It's okay to say "I struck out on X" if you did.
- Markdown only. No HTML.
"""


def render_pre_report(
    goal: str,
    started_at: datetime,
    stop_reason: str,
    stats: dict[str, Any],
    notes: list[str],
) -> str:
    """A fallback report if synthesis fails — at minimum the user gets the raw notes."""
    out = [
        "# 🌙 Morning report (raw)",
        "",
        f"_Dreamed at {started_at.isoformat(timespec='seconds')}, stopped because: **{stop_reason}**_",
        "",
        "## 🎯 What I dreamed about",
        "",
        goal.strip(),
        "",
        "## 📓 Raw notes",
        "",
    ]
    if not notes:
        out.append("_(no notes recorded — the dream may have ended too early)_")
    else:
        for i, n in enumerate(notes, 1):
            out.append(f"{i}. {n}")
    out += ["", "## 📊 Stats", "", "```json", json.dumps(stats, indent=2), "```", ""]
    return "\n".join(out)


def build_synthesis_prompt(
    goal: str,
    notes: list[str],
    stop_reason: str,
    stats: dict[str, Any],
) -> str:
    """Build the prompt that asks the LLM to synthesize the final report."""
    notes_block = "\n".join(f"- {n}" for n in notes) if notes else "(no notes recorded)"
    return REPORT_PROMPT.format(
        goal=goal.strip(),
        notes=notes_block,
        stop_reason=stop_reason,
        stats=json.dumps(stats),
    )


def save_report(
    report_text: str,
    dream_dir: Path,
    goal: str,
    started_at: datetime,
    stop_reason: str,
    stats: dict[str, Any],
    notes: list[str],
) -> Path:
    """Write the report + a sidecar metadata file. Returns the report path."""
    dream_dir.mkdir(parents=True, exist_ok=True)
    report_path = dream_dir / "report.md"

    # Prepend a small header so the file is self-describing.
    header = (
        f"<!-- mcp-dream report\n"
        f"started_at: {started_at.isoformat(timespec='seconds')}\n"
        f"stop_reason: {stop_reason}\n"
        f"-->\n\n"
    )
    report_path.write_text(header + report_text)

    metadata = {
        "goal": goal,
        "started_at": started_at.isoformat(),
        "stop_reason": stop_reason,
        "stats": stats,
        "notes": notes,
    }
    (dream_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return report_path
