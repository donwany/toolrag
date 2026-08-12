"""
Rich console helpers for user-facing CLI output.

Keep Rich usage in entrypoints / presentation layers.
Core logic should log via Loguru instead.
"""

from __future__ import annotations

from typing import Any, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON

console = Console()


def panel(title: str, body: str, *, style: str = "cyan") -> None:
    console.print(Panel(body, title=title, border_style=style))


def tools_table(tools: Iterable[dict[str, Any]], *, title: str = "Retrieved tools") -> None:
    t = Table(title=title, show_lines=False)
    t.add_column("#", style="dim", width=3)
    t.add_column("tool_id", style="bold")
    t.add_column("description")
    t.add_column("when_to_use")
    for i, tool in enumerate(tools, 1):
        t.add_row(str(i), str(tool.get("tool_id", "")), str(tool.get("description", "")), str(tool.get("when_to_use", "")))
    console.print(t)


def scores_table(
    tools: Iterable[dict[str, Any]],
    *,
    title: str = "tool scores"
) -> None:
    t = Table(title=title, show_lines=False)
    t.add_column("#", style="dim", width=3)
    t.add_column("tool_id", style="bold")
    t.add_column("score", justify="right")
    t.add_column("similarity", justify="right")

    for i, tool in enumerate(tools, 1):
        t.add_row(
            str(i),
            str(tool.get("tool_id", "")),
            f"{tool.get('score', 0.0):.4f}",
            f"{tool.get('similarity', 0.0):.4f}",
        )

    console.print(t)


def pretty_json(data: Any, *, title: str | None = None) -> None:
    # JSON expects a string; for dict/list, convert via repr-safe JSON.
    if isinstance(data, str):
        try:
            console.print(JSON(data), title=title)
        except Exception:
            console.print(data)
    else:
        import json

        console.print(JSON(json.dumps(data, default=str, indent=2)), title=title)

