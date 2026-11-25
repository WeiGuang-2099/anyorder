"""Searches schedules, by seed or exhaustively, and replays what it reports."""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Iterable, Iterator, Sequence

from anyorder import configuration
from anyorder import scheduler
from anyorder import values


@dataclasses.dataclass(frozen=True)
class Failure:
    """A run that broke an invariant.

    Attributes:
        seed: The seed whose random policy and whose draws produced the
            run.
        result: The failing run. Its schedule and its draws replay the
            failure without the seed.
    """

    seed: int
    result: scheduler.RunResult


def explore(
    scenario: scheduler.Scenario, seeds: Iterable[int]
) -> Failure | None:
    """Runs one random schedule per seed until an invariant breaks.

    Agents that draw are given fresh draws for each seed, from a seeded
    sampler. A failure is replayed from its schedule and its draws before
    it is returned, so a reported failure is always one that can be
    reproduced.

    Args:
        scenario: What to run.
        seeds: Seeds to try, in order.

    Returns:
        The first failure, or None if every run passed.

    Raises:
        ReplayDivergenceError: A failing run did not replay to the same
            trace, which means the scenario is not deterministic.
        UnsupportedValueError: An agent asked to draw from something that
            is no written distribution.
    """
    for seed in seeds:
        result = scheduler.run(
            scenario, scheduler.RandomPolicy(seed),
            scheduler.SeededSampler(seed))
        if result.violation is None:
            continue
        if replay(scenario, result.steps, result.draws) != result:
            raise scheduler.ReplayDivergenceError(
                f"seed {seed} broke an invariant of {scenario.name!r}, but "
                "its schedule replayed to a different trace")
        return Failure(seed, result)
    return None


def exhaust(
    scenario: scheduler.Scenario, sampler: scheduler.Sampler | None = None
) -> Iterator[scheduler.RunResult]:
    """Runs every complete schedule of a scenario, depth first.

    Waiting agents are tried in name order, so schedules come out sorted.
    Each run is replayed from its steps and its draws before it is
    yielded, so a failure taken from here is one that can be reproduced.
    Running every schedule twice probes that the scenario is
    deterministic. It does not prove it.

    Nothing bounds the search: every agent must finish under every
    schedule, and the number of schedules can grow exponentially with
    the number of calls and factorially with the number of agents.

    A scenario whose agents draw is searched under one sampler, which all
    the runs share. It must give an agent the same draw wherever the agent
    has seen and done the same, as configuration.TableSampler does: then
    the agents are a program with no chance left in it, and its schedules
    are what is run. A sampler that draws in the order it is asked can
    give another draw at the same place in another run, and the search
    refuses the run in which it does.

    Args:
        scenario: What to run.
        sampler: Gives the agents their draws, in every run. With none, a
            scenario whose agents draw is not searched.

    Yields:
        The run of each complete schedule, once.

    Raises:
        ReplayDivergenceError: A run did not replay to the same trace, or
            left the start of the schedule it was given. Either means the
            scenario is not deterministic: state outlives a run, or an
            argument, a tool's result or an invariant's message does not
            compare equal from run to run. Or the sampler gave an agent
            another draw in this run than in one before it, where the
            agent had seen and done the same.
        UnsupportedAwaitError: An agent drew and there is no sampler.
    """
    prefix: tuple[str, ...] | None = ()
    drawn: dict[Hashable, scheduler.Drawn] = {}
    while prefix is not None:
        policy = _PrefixPolicy(prefix)
        result = scheduler.run(scenario, policy, sampler)
        if len(result.steps) < len(prefix):
            raise scheduler.ReplayDivergenceError(
                f"a run of {scenario.name!r} ended after "
                f"{len(result.steps)} steps, inside the {len(prefix)} steps "
                "it was given to follow")
        if result.draws:
            _expect_one_draw_at_each_place(scenario, drawn, result)
        if replay(scenario, result.steps, result.draws) != result:
            raise scheduler.ReplayDivergenceError(
                f"the schedule {' '.join(result.schedule)!r} of "
                f"{scenario.name!r} replayed to a different trace")
        yield result
        prefix = _next_prefix(result.schedule, policy.seen)


def _expect_one_draw_at_each_place(
    scenario: scheduler.Scenario,
    drawn: dict[Hashable, scheduler.Drawn],
    result: scheduler.RunResult,
) -> None:
    """Refuses a run that drew otherwise than a run before it at one place.

    The runs of a search must be runs of one program, and they are if an
    agent is given the same draw wherever it has seen and done the same.
    A replay does not tell: it is given the draws of the run it replays.

    Args:
        scenario: What is searched.
        drawn: The first draw at each place, of the runs so far. Those of
            this run are added.
        result: This run.
    """
    for place, draw in configuration.places(result):
        first = drawn.setdefault(place, draw)
        if values.key(first.value) != values.key(draw.value):
            raise scheduler.ReplayDivergenceError(
                f"agent {draw.agent!r} of {scenario.name!r} drew "
                f"{values.shown(first.value)} in one run and "
                f"{values.shown(draw.value)} in another, where it had seen "
                "and done the same: an exhaustive search needs a sampler "
                "that gives one draw at one place")


def replay(
    scenario: scheduler.Scenario,
    schedule: Sequence[str | scheduler.Step],
    draws: Sequence[scheduler.Drawn] = (),
) -> scheduler.RunResult:
    """Runs a scenario under a recorded schedule and recorded draws.

    Args:
        scenario: What to run.
        schedule: The agent to choose at each step, or the steps of the
            run that is replayed. Given the steps, each call is held
            against its step before it is made, and a run that would
            make another call ends there.
        draws: The draws of the run, in the order they were made. Each is
            given back to the agent that made it. A run that drew nothing
            has none.

    Returns:
        The run's trace, its draws and the first violated invariant, if
        any.

    Raises:
        ReplayDivergenceError: The schedule names an agent that is not
            waiting, ends while agents can still run, or has steps left
            when the run ends. Or an agent waits to make another call
            than the step given for it. Or an agent asks for a draw that is not
            the next of those given: there is none left, another agent
            made it, or it was drawn from something else. Or the run
            ends with draws left, or made one after another step than
            the draw says.
    """
    result = scheduler.run(
        scenario, scheduler.ReplayPolicy(schedule),
        scheduler.ReplaySampler(draws))
    if len(result.steps) != len(schedule):
        raise scheduler.ReplayDivergenceError(
            f"the run ended after {len(result.steps)} steps but the "
            f"schedule has {len(schedule)}")
    if len(result.draws) != len(draws):
        raise scheduler.ReplayDivergenceError(
            f"the run made {len(result.draws)} draws but {len(draws)} were "
            "given")
    for number, (made, held) in enumerate(zip(result.draws, draws), start=1):
        # The sampler is not told where a run stands, so this is seen
        # only now.
        if made.step != held.step:
            raise scheduler.ReplayDivergenceError(
                f"draw {number} was made after {made.step} steps, not after "
                f"{held.step}")
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

    def choose(self, moment: scheduler.Moment) -> str:
        waiting = moment.waiting
        self.seen.append(tuple(waiting))
        if len(self.seen) > self._length:
            return waiting[0]
        return self._replay.choose(moment)


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
