"""Tools the dreaming agent can invoke.

Each tool is independently toggleable in config. Safe defaults: only web
search and read-only file access in explicitly-allowed paths.

We expose tools in the Anthropic tool-use JSON-schema format so the agent
loop can pass them straight through to messages.create(tools=...).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import ToolsConfig


# Tool implementations ────────────────────────────────────────────────────────


def _tool_web_search(query: str, max_results: int = 5) -> str:
    """Free DuckDuckGo lite scrape. Good enough for an overnight dream loop.

    No API key required, which keeps the bar to first dream low.
    """
    try:
        r = httpx.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (mcp-dream)"},
            timeout=15.0,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"[web_search error] {e}"

    # Very light parsing — we want titles and URLs, not perfection.
    # The HTML structure is stable enough for the lite endpoint.
    import re

    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    matches = pattern.findall(r.text)[:max_results]
    if not matches:
        return f"[web_search] no results for: {query}"

    lines = []
    tag_re = re.compile(r"<[^>]+>")
    for url, title in matches:
        clean_title = tag_re.sub("", title).strip()
        lines.append(f"- {clean_title}\n  {url}")
    return "\n".join(lines)


def _tool_read_file(path: str, allowed_roots: list[Path], max_chars: int = 12000) -> str:
    """Read a file, but only if it lives under an allowed root."""
    if not allowed_roots:
        return "[read_file error] no allowed_read_paths configured"

    target = Path(path).expanduser().resolve()
    if not any(_is_within(target, root) for root in allowed_roots):
        return f"[read_file error] {target} is not under any allowed root"

    if not target.exists():
        return f"[read_file error] {target} does not exist"
    if not target.is_file():
        return f"[read_file error] {target} is not a file"

    try:
        text = target.read_text(errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"[read_file error] {e}"

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[...truncated, file is {len(text)} chars]"
    return text


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _tool_list_dir(path: str, allowed_roots: list[Path]) -> str:
    """List entries in a directory under an allowed root."""
    if not allowed_roots:
        return "[list_dir error] no allowed_read_paths configured"
    target = Path(path).expanduser().resolve()
    if not any(_is_within(target, root) for root in allowed_roots):
        return f"[list_dir error] {target} is not under any allowed root"
    if not target.exists() or not target.is_dir():
        return f"[list_dir error] {target} is not a directory"

    entries = sorted(target.iterdir())
    out = []
    for e in entries[:200]:
        marker = "/" if e.is_dir() else ""
        out.append(f"{e.name}{marker}")
    if len(entries) > 200:
        out.append(f"... and {len(entries) - 200} more")
    return "\n".join(out)


def _tool_shell(command: str, timeout: int = 30) -> str:
    """Run a shell command. DANGEROUS — only enabled if user opts in."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[shell] timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"[shell error] {e}"

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out[:4000]}")
    if err:
        parts.append(f"stderr:\n{err[:2000]}")
    return "\n".join(parts)


def _tool_run_python(code: str, timeout: int = 30) -> str:
    """Run a Python snippet in a subprocess. DANGEROUS — opt-in only."""
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[run_python] timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"[run_python error] {e}"
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out[:4000]}")
    if err:
        parts.append(f"stderr:\n{err[:2000]}")
    return "\n".join(parts)


def _tool_note(content: str, scratch: list[str]) -> str:
    """Append a note to the dream's scratch pad.

    This is how the agent saves observations during the dream that should
    surface in the morning report.
    """
    scratch.append(content.strip())
    return f"noted ({len(scratch)} notes so far)"


# Tool registry ───────────────────────────────────────────────────────────────


# Each entry: (anthropic_schema, implementation_factory)
# We build implementations as closures so they capture config-derived state
# like allowed roots and the scratch pad.


def build_toolset(
    tools_config: ToolsConfig, scratch_notes: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]]]:
    """Build the tool schemas and implementations for a given dream.

    Returns:
      (anthropic_tool_schemas, name -> callable)
    The schemas can be passed directly as `tools=` to messages.create.
    """
    schemas: list[dict[str, Any]] = []
    impls: dict[str, Callable[..., str]] = {}

    # Note is always available — the agent needs somewhere to record findings.
    schemas.append(
        {
            "name": "note",
            "description": (
                "Save a finding, idea, or observation that should appear in the "
                "morning report. Write it as a single complete thought."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note to save. One thought per call.",
                    }
                },
                "required": ["content"],
            },
        }
    )
    impls["note"] = lambda content: _tool_note(content, scratch_notes)

    if tools_config.web_search:
        schemas.append(
            {
                "name": "web_search",
                "description": "Search the web. Returns a list of titles and URLs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            }
        )
        impls["web_search"] = _tool_web_search

    allowed_roots = [Path(p).expanduser().resolve() for p in tools_config.allowed_read_paths]

    if tools_config.read_files and allowed_roots:
        schemas.append(
            {
                "name": "read_file",
                "description": (
                    "Read a text file. Restricted to configured allowed roots: "
                    + ", ".join(str(r) for r in allowed_roots)
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        )
        impls["read_file"] = lambda path: _tool_read_file(path, allowed_roots)

        schemas.append(
            {
                "name": "list_dir",
                "description": "List the contents of a directory under an allowed root.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        )
        impls["list_dir"] = lambda path: _tool_list_dir(path, allowed_roots)

    if tools_config.shell:
        schemas.append(
            {
                "name": "shell",
                "description": "Run a shell command. Output is captured and returned.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30},
                    },
                    "required": ["command"],
                },
            }
        )
        impls["shell"] = _tool_shell

    if tools_config.code_execution:
        schemas.append(
            {
                "name": "run_python",
                "description": "Execute a Python snippet and return stdout/stderr.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30},
                    },
                    "required": ["code"],
                },
            }
        )
        impls["run_python"] = _tool_run_python

    return schemas, impls


def dispatch(name: str, args: dict[str, Any], impls: dict[str, Callable[..., str]]) -> str:
    """Run a tool by name with the given args dict. Always returns a string."""
    fn = impls.get(name)
    if fn is None:
        return f"[unknown tool: {name}]"
    try:
        return fn(**args)
    except TypeError as e:
        return f"[tool {name} bad args] {e}: got {json.dumps(args)}"
    except Exception as e:  # noqa: BLE001
        return f"[tool {name} error] {e}"
