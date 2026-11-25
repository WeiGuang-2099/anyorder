"""Shrinks a failing run to one that is easier to read.

A failing run is smaller when fewer context switches lead to its violation,
or as many and fewer agents. The search moves steps of the schedule it was
given, keeps a move when the run it leads to breaks the same invariant and
is smaller, and stops when no move helps. What it returns has been replayed.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator, Sequence

from anyorder import explore
from anyorder import scheduler


@dataclasses.dataclass(frozen=True)
class Shrunk:
    """A failing run after shrinking.

    Attributes:
        result: The smallest failing run found, replayed from its schedule.
            It is the given run if no move made it smaller.
        candidates: How many candidate runs the search made. Replays are
            not counted.
    """

    result: scheduler.RunResult
    candidates: int


def context_switches(schedule: Sequence[str]) -> int:
    """Counts the steps taken by another agent than the step before."""
    return sum(
        name != previous for previous, name in itertools.pairwise(schedule))


def size(failure: scheduler.RunResult) -> tuple[int, int]:
    """Measures a failing run by the steps that led to its violation.

    Steps after the violation are not counted: the invariant had broken
    without them. An invariant that must hold only at the end is seen to
    break at the end, so there every step counts.

    Args:
        failure: A run that broke an invariant.

    Returns:
        The context switches among those steps, then the agents that took
        one of them. Tuples compare in that order, which is the order a
        smaller failure is judged in.

    Raises:
        ValueError: The run broke no invariant.
    """
    if failure.violation is None:
        raise ValueError("the run broke no invariant, so it has no size")
    seen = failure.schedule[:failure.violation.step]
    return context_switches(seen), len(set(seen))


def shrink(
    scenario: scheduler.Scenario, result: scheduler.RunResult
) -> Shrunk:
    """Searches for a smaller run that breaks the same invariant.

    Each candidate moves steps of the current run's schedule: one agent's
    steps are put off to the end, or two of an agent's blocks are joined,
    where a block is a stretch of steps by one agent. The first candidate
    that breaks the same invariant with a smaller size becomes the current
    run, and the search starts again from it. It ends when no candidate
    does.

    The result is a local minimum: no single move makes it smaller. It is
    not always the smallest failing run: a move takes whole blocks and
    never part of one, and the search does not go back on a run it has
    kept. A run keeps only its first violation, so a candidate that breaks
    another invariant before this one is passed over.

    Nothing bounds a candidate run. Every agent must finish under every
    candidate, as under every schedule of an exhaustive search: an agent
    that waits for another by trying again can keep a candidate from
    ending, and then the search does not return.

    Args:
        scenario: What the run is a run of.
        result: A run that broke an invariant.

    Returns:
        The smallest failing run found and how many candidates were run.

    Raises:
        ValueError: The run broke no invariant, or it made draws. A run
            that drew is not shrunk yet: its draws would have to be held
            fixed while its schedule changes, and nothing says which draw
            an agent is given once it has read something else. Nothing
            has run.
        ReplayDivergenceError: The given run or a smaller one did not
            replay to the same trace. Either the run is not one of this
            scenario, or the scenario is not deterministic.
        UnsupportedAwaitError: An agent drew under a candidate, where it
            did not in the given run.
    """
    if result.violation is None:
        raise ValueError(
            "the run broke no invariant, so there is nothing to shrink")
    if result.draws:
        raise ValueError(
            "the run made draws, and a run that drew is not shrunk yet")
    _confirm(scenario, result)
    candidates = 0
    orders = _orders(result.schedule)
    while (order := next(orders, None)) is not None:
        candidates += 1
        candidate = scheduler.run(scenario, _FollowPolicy(order))
        if _is_smaller(candidate, result):
            _confirm(scenario, candidate)
            result = candidate
            orders = _orders(result.schedule)
    return Shrunk(result, candidates)


class _FollowPolicy:
    """Follows an order as far as the run allows.

    A candidate is an order to prefer and not a schedule: moving an agent's
    steps can change what it reads, and with that how many calls it makes.
    An entry whose agent has finished is skipped, and once the order runs
    out the first waiting agent runs. So no candidate has to be refused for
    leaving the order, and one that ends gives a complete run with a
    schedule of its own. A candidate does not end if the first waiting
    agent keeps calling until another agent acts.
    """

    def __init__(self, order: Sequence[str]) -> None:
        self._order = iter(order)

    def choose(self, waiting: Sequence[str]) -> str:
        for name in self._order:
            if name in waiting:
                return name
        return waiting[0]


def _confirm(scenario: scheduler.Scenario, run: scheduler.RunResult) -> None:
    """Replays a run and refuses it if the trace is not the same."""
    if explore.replay(scenario, run.schedule) != run:
        raise scheduler.ReplayDivergenceError(
            f"the schedule {' '.join(run.schedule)!r} of {scenario.name!r} "
            "replayed to a different trace")


def _is_smaller(
    candidate: scheduler.RunResult, failure: scheduler.RunResult
) -> bool:
    """Tells whether a run breaks a failure's invariant with a smaller size.

    The invariant is compared by name. The message and the step change with
    the schedule.
    """
    return (
        candidate.violation is not None
        and candidate.violation.invariant == failure.violation.invariant
        and size(candidate) < size(failure))


def _orders(schedule: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Yields the orders to try from a schedule, each of them once.

    Each agent put off comes first, in name order, then each two blocks
    joined, in the order of the blocks. A move can give the schedule itself
    or an order another move gave. Those are left out: the run they lead to
    has been seen.
    """
    seen = {tuple(schedule)}
    for order in itertools.chain(_put_off(schedule), _joined(schedule)):
        if order not in seen:
            seen.add(order)
            yield order


def _put_off(schedule: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Yields the schedule with each agent's steps moved to the end.

    This is how an agent leaves the steps that lead to the violation.
    """
    for name in sorted(set(schedule)):
        others = [entry for entry in schedule if entry != name]
        yield (*others, *[name] * (len(schedule) - len(others)))


def _joined(schedule: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Yields the schedule with each two blocks of one agent made one.

    One block is taken out and the other grows by its length, either way
    round, so an agent's earlier steps are put off or its later steps are
    brought forward. That removes at least one context switch from the
    order.
    """
    blocks = [
        (name, len(list(steps)))
        for name, steps in itertools.groupby(schedule)]
    for taken, (name, length) in enumerate(blocks):
        for kept, (other, other_length) in enumerate(blocks):
            if taken == kept or name != other:
                continue
            joined = list(blocks)
            joined[kept] = (name, length + other_length)
            del joined[taken]
            yield tuple(
                entry for entry, count in joined for _ in range(count))
