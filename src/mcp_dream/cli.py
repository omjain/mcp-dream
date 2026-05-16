"""Command-line interface for mcp-dream.

Subcommands:
  mcp-dream init       — create ~/.mcp-dream/config.toml
  mcp-dream install    — print/install Claude Desktop MCP config snippet
  mcp-dream dream      — run a dream synchronously (no MCP)
  mcp-dream list       — list past dreams
  mcp-dream report ID  — print a report
  mcp-dream config     — show effective config
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import DEFAULT_HOME, load_config, write_default_config, config_to_dict
from .daemon import dream as run_dream


console = Console()


@click.group(help="🌙 mcp-dream — let AI dream on your problems while you sleep.")
@click.version_option(__version__)
def main() -> None:
    pass


@main.command()
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
def init(force: bool) -> None:
    """Create ~/.mcp-dream/config.toml with safe defaults."""
    path = write_default_config(overwrite=force)
    console.print(f"[green]✓[/green] Config at [bold]{path}[/bold]")
    console.print(
        "\nNext steps:\n"
        "  1. (optional) Edit the config to add allowed_read_paths, opt into shell, etc.\n"
        "  2. Set ANTHROPIC_API_KEY in your environment (or set provider.name = 'ollama').\n"
        "  3. Run [cyan]mcp-dream install[/cyan] to wire it into Claude Desktop.\n"
        "  4. Or try it now: [cyan]mcp-dream dream \"surprise me with three startup ideas\"[/cyan]"
    )


@main.command(name="config")
def show_config() -> None:
    """Show the effective configuration."""
    cfg = load_config()
    console.print(Panel.fit(json.dumps(config_to_dict(cfg), indent=2), title="config"))


@main.command()
@click.argument("goal", nargs=-1, required=True)
@click.option("--hours", type=float, default=None, help="Override max_hours for this run.")
@click.option("--dollars", type=float, default=None, help="Override max_dollars for this run.")
@click.option("--iters", type=int, default=None, help="Override max_iterations for this run.")
def dream(goal: tuple[str, ...], hours: float | None, dollars: float | None, iters: int | None) -> None:
    """Run a dream synchronously, right here in your terminal.

    Example:
      mcp-dream dream "find three weird papers I'd like about diffusion models"
    """
    goal_text = " ".join(goal)
    cfg = load_config()
    if hours is not None:
        cfg.budget.max_hours = hours
    if dollars is not None:
        cfg.budget.max_dollars = dollars
    if iters is not None:
        cfg.budget.max_iterations = iters

    console.print(Panel.fit(f"🌙 dreaming on:\n\n[italic]{goal_text}[/italic]", title="mcp-dream"))
    console.print(
        f"[dim]caps: {cfg.budget.max_hours}h, ${cfg.budget.max_dollars}, "
        f"{cfg.budget.max_iterations} iters · provider: {cfg.provider.name}[/dim]\n"
    )

    with console.status("[bold cyan]dreaming...[/bold cyan]"):
        run = run_dream(cfg, goal_text)

    if run.error:
        console.print(f"[red]error during dream:[/red] {run.error}")

    console.print(
        f"\n[green]✓[/green] dream complete · stop_reason: [bold]{run.stop_reason}[/bold] · "
        f"notes: {len(run.scratch_notes)}"
    )
    if run.tracker:
        console.print(f"[dim]{json.dumps(run.tracker.summary())}[/dim]\n")

    if run.report_path and run.report_path.exists():
        console.print(Markdown(run.report_path.read_text()))
        console.print(f"\n[dim]saved to: {run.report_path}[/dim]")


@main.command(name="list")
def list_dreams() -> None:
    """List past dreams from disk."""
    cfg = load_config()
    if not cfg.dreams_dir.exists():
        console.print("[dim]no dreams yet — run [cyan]mcp-dream dream \"...\"[/cyan][/dim]")
        return
    dirs = sorted([p for p in cfg.dreams_dir.iterdir() if p.is_dir()], reverse=True)
    if not dirs:
        console.print("[dim]no dreams yet[/dim]")
        return

    table = Table(title="🌙 dreams")
    table.add_column("dream_id", style="cyan")
    table.add_column("stop", style="magenta")
    table.add_column("notes", justify="right")
    table.add_column("goal")

    for d in dirs[:30]:
        meta_path = d / "metadata.json"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                table.add_row(
                    d.name,
                    str(m.get("stop_reason", "?")),
                    str(len(m.get("notes", []))),
                    (m.get("goal") or "").strip()[:80],
                )
            except Exception:  # noqa: BLE001
                table.add_row(d.name, "?", "?", "(unreadable metadata)")
        else:
            table.add_row(d.name, "?", "?", "(in progress?)")
    console.print(table)


@main.command()
@click.argument("dream_id")
def report(dream_id: str) -> None:
    """Print a dream's morning report."""
    cfg = load_config()
    path = cfg.dreams_dir / dream_id / "report.md"
    if not path.exists():
        console.print(f"[red]no report at {path}[/red]")
        sys.exit(1)
    console.print(Markdown(path.read_text()))


@main.command()
def install() -> None:
    """Print a Claude Desktop MCP-config snippet you can paste."""
    # We don't auto-edit the user's Claude config — too risky across platforms.
    # Print a copy-pasteable snippet instead.
    snippet = {
        "mcpServers": {
            "mcp-dream": {
                "command": "mcp-dream-server",
                "env": {
                    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
                },
            }
        }
    }
    console.print(
        Panel.fit(
            "[bold]Paste this into your Claude Desktop config[/bold]\n"
            "(macOS: ~/Library/Application Support/Claude/claude_desktop_config.json)\n"
            "(Windows: %APPDATA%\\Claude\\claude_desktop_config.json)\n\n"
            + json.dumps(snippet, indent=2),
            title="install",
        )
    )
    console.print(
        "\n[dim]Then restart Claude Desktop. You should see a 🛠️  with 'mcp-dream' "
        "in the tools list.[/dim]"
    )


if __name__ == "__main__":
    main()
