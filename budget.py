"""Budget enforcement for dream loops.

Three caps run in parallel:
  - wall-clock time
  - estimated USD spend (Anthropic only; Ollama is free)
  - number of LLM iterations

A BudgetTracker is constructed once per dream and consulted on every loop
turn. The first cap to trip stops the dream gracefully.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .config import BudgetConfig


# Rough per-1M-token pricing in USD for the models we support.
# These are approximations used only for the safety cap — not for billing.
# Users should treat the displayed cost as a ballpark.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    # model_id: (input_per_mtok, output_per_mtok)
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
_FALLBACK_PRICING = (15.0, 75.0)  # assume expensive if unknown


StopReason = Literal["time", "dollars", "iterations", "model_stopped", "user_stopped"]


@dataclass
class BudgetTracker:
    """Tracks usage and decides when to stop."""

    config: BudgetConfig
    model: str = "claude-opus-4-7"

    started_at: float = field(default_factory=time.monotonic)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record_turn(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record one LLM call's usage."""
        self.iterations += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started_at) / 3600.0

    @property
    def estimated_dollars(self) -> float:
        in_rate, out_rate = PRICING_PER_MTOK.get(self.model, _FALLBACK_PRICING)
        cost = (self.input_tokens / 1_000_000.0) * in_rate
        cost += (self.output_tokens / 1_000_000.0) * out_rate
        return cost

    def check(self) -> StopReason | None:
        """Return a stop reason if a cap is hit, else None."""
        if self.elapsed_hours >= self.config.max_hours:
            return "time"
        if self.estimated_dollars >= self.config.max_dollars:
            return "dollars"
        if self.iterations >= self.config.max_iterations:
            return "iterations"
        return None

    def summary(self) -> dict[str, float | int]:
        """A serializable snapshot for reports and status checks."""
        return {
            "elapsed_hours": round(self.elapsed_hours, 3),
            "iterations": self.iterations,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_dollars": round(self.estimated_dollars, 4),
            "cap_hours": self.config.max_hours,
            "cap_dollars": self.config.max_dollars,
            "cap_iterations": self.config.max_iterations,
        }
