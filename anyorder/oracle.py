"""Judges a run by the serial orders of its agents.

A serial order runs the agents one after another: each makes all its calls
before the next makes one. A run fails when it leaves its world in a state
that no serial order leaves: then no way of taking turns explains what the
agents did together. An invariant needs a rule about one world. This oracle
needs none, and it sees a lost update that breaks no invariant.

What a run leaves is what a snapshot function gives of its world once every
agent has finished, and only that is compared. What an agent read or was
told is not, so an agent's conclusion counts only where a tool call wrote
it to the world. A failure that the serial orders share is not seen either:
the invariants are the oracle for that.

A run does not hold its world, so the oracle runs the schedule again to
look at it. The scheduler is used as it is.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

from anyorder import explore
from anyorder import scheduler

Snapshot = Callable[[Any], Hashable]
Order = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SerialOutcomes:
    """What the serial orders of a scenario's agents leave.

    Attributes:
        scenario: Whose agents were run one after another.
        snapshot: Gives the state of a world. A run is judged with the same
            one.
        orders: The serial orders by their outcome, which is the state of
            the world once every agent has finished.
    """

    scenario: scheduler.Scenario
    snapshot: Snapshot
    orders: Mapping[Hashable, tuple[Order, ...]]


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What the oracle says of one run.

    Attributes:
        outcome: The state the run left its world in.
        orders: The serial orders that end in that state. The run failed
            if there is none.
    """

    outcome: Hashable
    orders: tuple[Order, ...]

    @property
    def failed(self) -> bool:
        """True if no serial order ends as the run ended."""
        return not self.orders


def serial_outcomes(
    scenario: scheduler.Scenario, snapshot: Snapshot
) -> SerialOutcomes:
    """Runs every serial order of a scenario's agents.

    Each order is run twice, and the two runs must give the same trace and
    end in the same state: every run is judged against these outcomes.

    Nothing bounds the work. That is 2 * n! runs for n agents, and every
    agent must finish in every order: an agent that waits for a later one
    by trying again never does.

    The orders are run with no sampler, so a scenario whose agents draw
    has no serial outcomes yet: they would have to be those under the
    draws of the run that is judged.

    Args:
        scenario: Whose agents to run.
        snapshot: Gives the state of a world as a value that compares
            equal for worlds in the same state and can be hashed, such as
            a tuple. It decides what counts as ending the same.

    Returns:
        The serial orders, in name order, by the outcome they end in.

    Raises:
        ReplayDivergenceError: A serial order did not give the same trace
            and the same outcome twice. Either the scenario is not
            deterministic, or the snapshot gives values that are not equal
            for worlds in the same state.
        TypeError: The snapshot gave a value that cannot be hashed.
        UnsupportedAwaitError: An agent drew.
    """
    orders: dict[Hashable, list[Order]] = {}
    for order in itertools.permutations(sorted(scenario.agents)):
        run, outcome = _observe(
            scenario, snapshot,
            functools.partial(scheduler.run, policy=_SerialPolicy(order)))
        again = _replayed(scenario, snapshot, run)
        if again != outcome:
            raise scheduler.ReplayDivergenceError(
                f"the serial order {' '.join(order)!r} of {scenario.name!r} "
                f"ended in another state when it was run again: {outcome!r}, "
                f"then {again!r}")
        orders.setdefault(outcome, []).append(order)
    return SerialOutcomes(
        scenario, snapshot,
        {outcome: tuple(found) for outcome, found in orders.items()})


def judge(serial: SerialOutcomes, result: scheduler.RunResult) -> Verdict:
    """Says which serial orders end as a run ended.

    The run is replayed from its schedule to look at its world, and the
    replay must give the same trace. A run that ends as no serial order
    ends is replayed once more before it is reported, and must give the
    same trace and end in the same state again: a trace does not hold what
    the verdict rests on.

    Args:
        serial: The serial outcomes of the scenario the run is a run of.
        result: The run to judge. Whether it broke an invariant plays no
            part.

    Returns:
        The run's outcome and the serial orders that end in it.

    Raises:
        ReplayDivergenceError: The run did not replay to the same trace, or
            it failed and ended in another state when it was run again.
            Either the run is not one of this scenario, or the scenario is
            not deterministic.
        TypeError: The snapshot gave a value that cannot be hashed.
    """
    scenario = serial.scenario
    outcome = _replayed(scenario, serial.snapshot, result)
    orders = serial.orders.get(outcome, ())
    if not orders:
        again = _replayed(scenario, serial.snapshot, result)
        if again != outcome:
            raise scheduler.ReplayDivergenceError(
                f"the schedule {' '.join(result.schedule)!r} of "
                f"{scenario.name!r} ended as no serial order ends, and "
                f"ended in another state when it was run again: {outcome!r}, "
                f"then {again!r}")
    return Verdict(outcome, orders)


class _SerialPolicy:
    """Runs the agents one after another, in a given order."""

    def __init__(self, order: Sequence[str]) -> None:
        self._order = order

    def choose(self, waiting: Sequence[str]) -> str:
        # The first that is waiting: those before it have finished.
        return next(name for name in self._order if name in waiting)


def _observe(
    scenario: scheduler.Scenario,
    snapshot: Snapshot,
    run: Callable[[scheduler.Scenario], scheduler.RunResult],
) -> tuple[scheduler.RunResult, Hashable]:
    """Makes one run and returns it with the state its world ends in.

    Args:
        scenario: What to run.
        snapshot: Gives the state of a world.
        run: Makes the run, given a copy of the scenario that keeps the
            world it builds.
    """
    worlds: list[Any] = []

    def make_world() -> Any:
        world = scenario.make_world()
        worlds.append(world)
        return world

    result = run(dataclasses.replace(scenario, make_world=make_world))
    # One world: the scheduler builds it once for a run.
    (world,) = worlds
    outcome = snapshot(world)
    try:
        hash(outcome)
    except TypeError as error:
        raise TypeError(
            f"the snapshot of {scenario.name!r} gave {outcome!r}, which "
            f"cannot be hashed: {error}") from None
    return result, outcome


def _replayed(
    scenario: scheduler.Scenario,
    snapshot: Snapshot,
    result: scheduler.RunResult,
) -> Hashable:
    """Replays a run and returns the state its world ends in.

    Raises:
        ReplayDivergenceError: The replay left the run's schedule or gave
            another trace.
    """
    def replay(watched: scheduler.Scenario) -> scheduler.RunResult:
        replayed = explore.replay(watched, result.schedule)
        # Here, before the snapshot is asked: the world of another run
        # may be one the snapshot was not written for.
        if replayed != result:
            raise scheduler.ReplayDivergenceError(
                f"the schedule {' '.join(result.schedule)!r} of "
                f"{scenario.name!r} replayed to a different trace")
        return replayed

    _, outcome = _observe(scenario, snapshot, replay)
    return outcome
