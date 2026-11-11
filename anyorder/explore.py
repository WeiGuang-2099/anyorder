"""Searches schedules for one that breaks an invariant, and replays it."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from anyorder import scheduler


@dataclasses.dataclass(frozen=True)
class Failure:
    """A run that broke an invariant.

    Attributes:
        seed: The seed whose random policy produced the run.
        result: The failing run. Its schedule replays the failure without
            the seed.
    """

    seed: int
    result: scheduler.RunResult


def explore(
    scenario: scheduler.Scenario, seeds: Iterable[int]
) -> Failure | None:
    """Runs one random schedule per seed until an invariant breaks.

    A failure is replayed from its schedule before it is returned, so a
    reported failure is always one that can be reproduced.

    Args:
        scenario: What to run.
        seeds: Seeds to try, in order.

    Returns:
        The first failure, or None if every run passed.

    Raises:
        ReplayDivergenceError: A failing run did not replay to the same
            trace, which means the scenario is not deterministic.
    """
    for seed in seeds:
        result = scheduler.run(scenario, scheduler.RandomPolicy(seed))
        if result.violation is None:
            continue
        if replay(scenario, result.schedule) != result:
            raise scheduler.ReplayDivergenceError(
                f"seed {seed} broke an invariant of {scenario.name!r}, but "
                "its schedule replayed to a different trace")
        return Failure(seed, result)
    return None


def replay(
    scenario: scheduler.Scenario, schedule: Sequence[str]
) -> scheduler.RunResult:
    """Runs a scenario under a recorded schedule.

    Args:
        scenario: What to run.
        schedule: The agent to choose at each step.

    Returns:
        The run's trace and the first violated invariant, if any.

    Raises:
        ReplayDivergenceError: The schedule names an agent that is not
            waiting, ends while agents can still run, or has steps left
            when the run ends.
    """
    result = scheduler.run(scenario, scheduler.ReplayPolicy(schedule))
    if len(result.steps) != len(schedule):
        raise scheduler.ReplayDivergenceError(
            f"the run ended after {len(result.steps)} steps but the "
            f"schedule has {len(schedule)}")
    return result
