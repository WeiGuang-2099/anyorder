"""Finds the oversell in the naive buyer scenario and prints its trace."""

from __future__ import annotations

import sys

from anyorder import explore
from anyorder import scheduler
from anyorder.scenarios import oversell

_SEEDS = range(100)


def _format_step(index: int, step: scheduler.Step) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in step.args.items())
    return f"  {index}. {step.agent}: {step.tool}({args}) -> {step.result!r}"


def main() -> int:
    """Prints the first failing schedule. Returns 1 if there is one."""
    scenario = oversell.naive()
    failure = explore.explore(scenario, _SEEDS)
    if failure is None:
        print(f"{scenario.name}: no violation in {len(_SEEDS)} schedules")
        return 0
    result = failure.result
    print(f"{scenario.name}: violation found at seed {failure.seed}")
    print(f"  {result.violation.message}")
    print(f"schedule: {' '.join(result.schedule)}")
    for index, step in enumerate(result.steps, start=1):
        print(_format_step(index, step))
    return 1


if __name__ == "__main__":
    sys.exit(main())
