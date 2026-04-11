"""GetTodayDate tool — returns the current calendar date.

Exists so the main agent's system prompt can stay fully date-agnostic:
when the user does not specify an analysis date in their query, the
agent calls this tool once at the start of the run to anchor the
analysis window.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from src.tools.base_tool import BaseTool


class GetTodayDateInput(BaseModel):
    """No inputs. The tool takes no arguments."""


class GetTodayDateTool(BaseTool):
    name = "get_today_date"
    description = (
        "**What**: Returns the current calendar date as an ISO string (YYYY-MM-DD). "
        "**When to use**: Call this exactly once, at the very start of a run, "
        "ONLY when the user's query does not already specify an analysis date. "
        "If the user said 'as of 2023-06-30', DO NOT call this tool. "
        "**Input**: No arguments. "
        "**Output**: A bare ISO date string, e.g. '2026-04-11'. "
        "**Limits**: Do not call more than once per run. Never call after "
        "sub-agent tools have already been invoked."
    )
    input_schema = GetTodayDateInput

    def execute(self) -> str:
        return date.today().isoformat()
