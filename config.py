"""Configuration loading and defaults.

Config lives at ~/.mcp-dream/config.toml. We bake in conservative defaults
so an out-of-the-box install can never run away with someone's API budget.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# Where everything lives on disk.
DEFAULT_HOME = Path.home() / ".mcp-dream"
CONFIG_FILENAME = "config.toml"
DREAMS_DIRNAME = "dreams"


@dataclass
class BudgetConfig:
    """Hard caps that protect users from runaway dreams.

    All three caps run simultaneously — whichever trips first wins.
    These defaults are intentionally conservative.
    """

    max_hours: float = 2.0
    max_dollars: float = 2.0
    max_iterations: int = 30


@dataclass
class ToolsConfig:
    """Which tools the dreaming agent is allowed to use.

    The safe defaults are read-only: web search and reading files in
    explicitly-allowed directories. Shell and code execution are off
    by default and require an explicit opt-in.
    """

    web_search: bool = True
    read_files: bool = True
    allowed_read_paths: list[str] = field(default_factory=list)
    shell: bool = False
    code_execution: bool = False


@dataclass
class ProviderConfig:
    """Which LLM provider to use for the dream loop."""

    # "anthropic" or "ollama"
    name: str = "anthropic"
    # Anthropic-specific
    anthropic_model: str = "claude-opus-4-7"
    # Ollama-specific
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"


@dataclass
class Config:
    """Top-level config."""

    home: Path = field(default_factory=lambda: DEFAULT_HOME)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    @property
    def dreams_dir(self) -> Path:
        return self.home / DREAMS_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.home / CONFIG_FILENAME

    def ensure_dirs(self) -> None:
        """Create ~/.mcp-dream and ~/.mcp-dream/dreams if missing."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.dreams_dir.mkdir(parents=True, exist_ok=True)


def _coerce(raw: dict[str, Any], home_override: Path | None = None) -> Config:
    """Build a Config from a parsed TOML dict, falling back to defaults."""
    cfg = Config()
    if home_override is not None:
        cfg.home = home_override

    prov = raw.get("provider", {})
    if isinstance(prov, dict):
        cfg.provider = ProviderConfig(
            name=prov.get("name", cfg.provider.name),
            anthropic_model=prov.get("anthropic_model", cfg.provider.anthropic_model),
            ollama_model=prov.get("ollama_model", cfg.provider.ollama_model),
            ollama_host=prov.get("ollama_host", cfg.provider.ollama_host),
        )

    tools = raw.get("tools", {})
    if isinstance(tools, dict):
        cfg.tools = ToolsConfig(
            web_search=tools.get("web_search", cfg.tools.web_search),
            read_files=tools.get("read_files", cfg.tools.read_files),
            allowed_read_paths=list(tools.get("allowed_read_paths", [])),
            shell=tools.get("shell", cfg.tools.shell),
            code_execution=tools.get("code_execution", cfg.tools.code_execution),
        )

    budget = raw.get("budget", {})
    if isinstance(budget, dict):
        cfg.budget = BudgetConfig(
            max_hours=float(budget.get("max_hours", cfg.budget.max_hours)),
            max_dollars=float(budget.get("max_dollars", cfg.budget.max_dollars)),
            max_iterations=int(budget.get("max_iterations", cfg.budget.max_iterations)),
        )

    return cfg


def load_config(home: Path | None = None) -> Config:
    """Load config from ~/.mcp-dream/config.toml. Returns defaults if missing.

    Pass `home` to use a non-default location (mainly for tests).
    """
    home = home or Path(os.environ.get("MCP_DREAM_HOME", DEFAULT_HOME))
    config_path = home / CONFIG_FILENAME
    if not config_path.exists():
        cfg = Config(home=home)
        return cfg

    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    return _coerce(raw, home_override=home)


def write_default_config(home: Path | None = None, overwrite: bool = False) -> Path:
    """Write an annotated default config to disk. Returns the path written."""
    home = home or Path(os.environ.get("MCP_DREAM_HOME", DEFAULT_HOME))
    home.mkdir(parents=True, exist_ok=True)
    path = home / CONFIG_FILENAME
    if path.exists() and not overwrite:
        return path
    path.write_text(_DEFAULT_CONFIG_TOML)
    return path


# Annotated default config — the file users will actually edit.
# Comments here are the primary documentation surface, so we invest in them.
_DEFAULT_CONFIG_TOML = '''\
# mcp-dream config
# ----------------
# This file controls how your AI dreams. Safe defaults are baked in — feel free
# to tweak, but read the comments first. Caps are ON by default for a reason.

[provider]
# "anthropic" (paid, uses your ANTHROPIC_API_KEY) or "ollama" (free, local)
name = "anthropic"

# Used when provider = "anthropic"
anthropic_model = "claude-opus-4-7"

# Used when provider = "ollama" — make sure the model is pulled locally first:
#   ollama pull llama3.2
ollama_model = "llama3.2"
ollama_host = "http://localhost:11434"


[tools]
# What the dreaming agent is allowed to do. The defaults are READ-ONLY for safety.
# Turn the dangerous knobs on only if you really know what you want.

web_search = true
read_files = true

# Directories the agent may read from. Leave empty to disable file reads entirely
# even if read_files = true. No wildcards — explicit paths only.
# Example:
#   allowed_read_paths = ["/Users/you/Projects/my-repo", "/Users/you/Notes"]
allowed_read_paths = []

# DANGEROUS: lets the agent run arbitrary shell commands. Default off.
shell = false

# DANGEROUS: lets the agent execute Python it writes. Default off.
code_execution = false


[budget]
# Three independent caps. The dream stops the moment ANY cap is hit.
# These are deliberately conservative — you can raise them, but consider why
# the defaults exist before you do.

max_hours = 2.0          # wall-clock hours
max_dollars = 2.0        # API spend estimate (anthropic provider only)
max_iterations = 30      # number of LLM turns
'''


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Useful for debugging / logging."""
    d = asdict(cfg)
    d["home"] = str(cfg.home)
    return d
