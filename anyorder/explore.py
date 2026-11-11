"""Searches schedules, by seed or exhaustively, and replays what it reports."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Sequence

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


def exhaust(scenario: scheduler.Scenario) -> Iterator[scheduler.RunResult]:
    """Runs every complete schedule of a scenario, depth first.

    Waiting agents are tried in name order, so schedules come out sorted.
    Each run is replayed from its schedule before it is yielded, so a
    failure taken from here is one that can be reproduced. Running every
    schedule twice probes that the scenario is deterministic. It does not
    prove it.

    Nothing bounds the search: every agent must finish under every
    schedule, and the number of schedules can grow exponentially with
    the number of calls and factorially with the number of agents.

    Args:
        scenario: What to run.

    Yields:
        The run of each complete schedule, once.

    Raises:
        ReplayDivergenceError: A run did not replay to the same trace, or
            left the start of the schedule it was given. Either means the
            scenario is not deterministic: state outlives a run, or an
            argument, a tool's result or an invariant's message does not
            compare equal from run to run.
    """
    prefix: tuple[str, ...] | None = ()
    while prefix is not None:
        policy = _PrefixPolicy(prefix)
        result = scheduler.run(scenario, policy)
        if len(result.steps) < len(prefix):
            raise scheduler.ReplayDivergenceError(
                f"a run of {scenario.name!r} ended after "
                f"{len(result.steps)} steps, inside the {len(prefix)} steps "
                "it was given to follow")
        if replay(scenario, result.schedule) != result:
            raise scheduler.ReplayDivergenceError(
                f"the schedule {' '.join(result.schedule)!r} of "
                f"{scenario.name!r} replayed to a different trace")
        yield result
        prefix = _next_prefix(result.schedule, policy.seen)


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


class _PrefixPolicy:
    """Follows the start of a schedule, then takes the first waiting agent.

    Attributes:
        seen: The waiting agents at each step so far, which is where the
            search finds the agents it has not tried.
    """

    def __init__(self, prefix: Sequence[str]) -> None:
        self._replay = scheduler.ReplayPolicy(prefix)
        self._length = len(prefix)
        self.seen: list[tuple[str, ...]] = []

    def choose(self, waiting: Sequence[str]) -> str:
        # A copy: the protocol does not promise a fresh list at each step.
        self.seen.append(tuple(waiting))
        if len(self.seen) > self._length:
            return waiting[0]
        return self._replay.choose(waiting)


def _next_prefix(
    schedule: Sequence[str], seen: Sequence[tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Returns the start of the next schedule, or None after the last.

    Past its prefix a run takes the first waiting agent, and a prefix ends
    with the agent after the one tried before. So at every step the agents
    before the chosen one have been tried and those after it have not, and
    the search turns at the deepest step that still has one.

    Args:
        schedule: The schedule that just ran.
        seen: The waiting agents at each of its steps, in name order.
    """
    for position in reversed(range(len(schedule))):
        waiting = seen[position]
        untried = waiting[waiting.index(schedule[position]) + 1:]
        if untried:
            return (*schedule[:position], untried[0])
    return None
