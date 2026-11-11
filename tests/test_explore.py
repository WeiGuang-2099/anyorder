"""Checks schedule search and replay on the oversell and other scenarios."""

import collections
import itertools
from collections.abc import Collection, Mapping, Sequence

import pytest

from anyorder import explore
from anyorder import scheduler
from anyorder.scenarios import oversell

_SERIAL = ("buyer-1", "buyer-1", "buyer-1", "buyer-2")
_BOTH_READ_FIRST = (
    "buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2")


class _Bell:
    """A world whose one tool changes nothing."""

    def ring(self) -> None:
        pass


def _ringers(
    calls: Mapping[str, int], quiet: Collection[str] = frozenset()
) -> scheduler.Scenario:
    """A scenario whose agents each ring a bell a set number of times.

    Args:
        calls: How many calls each agent makes, by name.
        quiet: Names of agents that make no call. An agent reads it when
            it starts, so a test that adds a name between two runs of one
            search plays the part of state that outlives a run.
    """
    async def ringer(ctx: scheduler.AgentContext) -> None:
        if ctx.name in quiet:
            return
        for _ in range(calls[ctx.name]):
            await ctx.call("ring")

    return scheduler.Scenario(
        name="bell",
        make_world=_Bell,
        agents={name: ringer for name in calls},
        invariants=(),
    )


def _leaky(invariants: Sequence[scheduler.Invariant]) -> scheduler.Scenario:
    """A scenario whose one tool reads state that outlives the run."""
    tickets = itertools.count()

    class Counter:
        def take(self) -> int:
            return next(tickets)

    async def taker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take")

    return scheduler.Scenario(
        name="leaky",
        make_world=Counter,
        agents={"a": taker},
        invariants=invariants,
    )


def _runs(
    scenario: scheduler.Scenario, limit: int = 1000
) -> list[scheduler.RunResult]:
    """Runs a whole exhaustive search, up to a limit.

    A search that stopped advancing would yield without end, and a test
    that drained it would hang where it should fail.
    """
    search = explore.exhaust(scenario)
    runs = list(itertools.islice(search, limit))
    assert next(search, None) is None, f"more than {limit} runs"
    return runs


def _reads_of_one(run: scheduler.RunResult) -> int:
    """Counts the buyers that read a stock of 1."""
    return sum(
        step.tool == "read_stock" and step.result == 1 for step in run.steps)


def test_search_finds_the_oversell():
    failure = explore.explore(oversell.naive(), range(20))
    assert failure is not None
    assert failure.result.violation is not None
    assert "2 orders placed against a stock of 1" in failure.result.violation


def test_failure_replays_from_its_schedule_alone():
    scenario = oversell.naive()
    failure = explore.explore(scenario, range(20))
    assert failure is not None
    assert explore.replay(scenario, failure.result.schedule) == failure.result


def test_serial_buyers_do_not_oversell():
    result = explore.replay(oversell.naive(), _SERIAL)
    assert result.violation is None
    assert result.steps[-1].result == 0


def test_two_reads_before_a_write_oversell():
    result = explore.replay(oversell.naive(), _BOTH_READ_FIRST)
    assert result.violation == (
        "2 orders placed against a stock of 1: buyer-1, buyer-2")


@pytest.mark.parametrize("stock,buyers", [(1, 2), (2, 5), (3, 3)])
def test_control_never_oversells(stock, buyers):
    scenario = oversell.atomic(stock=stock, buyers=buyers)
    assert explore.explore(scenario, range(200)) is None


def test_replay_refuses_steps_left_over():
    with pytest.raises(scheduler.ReplayDivergenceError, match="ended after 4"):
        explore.replay(oversell.naive(), _SERIAL + ("buyer-2",))


def test_search_refuses_a_failure_that_does_not_replay():
    scenario = _leaky((lambda world: "always fails",))
    with pytest.raises(scheduler.ReplayDivergenceError, match="'leaky'"):
        explore.explore(scenario, range(1))


def test_two_naive_buyers_have_16_schedules_and_12_oversell():
    # With a stock of 1 every write stores 0, so a buyer reads 1 exactly
    # when it reads before the first write. It then makes three calls
    # (read, write, order); a buyer that reads 0 makes one.
    # The first call of a run is a read of 1, so at least one buyer reads 1.
    # Both read 1: the two reads come first, in 2 orders, and the four
    # calls left (a write and an order each) interleave in
    # 4! / (2! 2!) = 6 ways. 2 * 6 = 12 schedules, 2 orders for 1 unit.
    # One reads 1: 2 choices. Its read and its write come first, and the
    # other buyer's read goes before or after its order. 2 * 2 = 4
    # schedules, 1 order each.
    # 12 + 4 = 16 schedules, and the 12 fail.
    runs = _runs(oversell.naive())
    failing = [run for run in runs if run.violation is not None]
    assert (len(runs), len(failing)) == (16, 12)
    assert collections.Counter(map(_reads_of_one, runs)) == {1: 4, 2: 12}
    assert all(_reads_of_one(run) == 2 for run in failing)


def test_three_naive_buyers_have_702_schedules_and_684_oversell():
    # As with two buyers, a buyer reads 1 exactly when it reads before the
    # first write, and a run fails exactly when two or more do. No read
    # of 0 comes before the first write, so the reads of 1 come first.
    # Three read 1: the reads in 3! = 6 orders, then six calls (a write
    # and an order each) in 6! / (2! 2! 2!) = 90 ways. 6 * 90 = 540.
    # Two read 1: 3 choices of the buyer that reads 0. The two reads in 2
    # orders, then their four calls in 6 ways, the first of them a write.
    # The read of 0 goes after any of those four. 3 * 2 * 6 * 4 = 144.
    # One reads 1: 3 choices. Its read and its write come first, then its
    # order and the two reads of 0 in 3! = 6 orders. 3 * 6 = 18.
    # 540 + 144 + 18 = 702 schedules, and 540 + 144 = 684 fail.
    runs = _runs(oversell.naive(buyers=3))
    failing = [run for run in runs if run.violation is not None]
    assert (len(runs), len(failing)) == (702, 684)
    assert collections.Counter(map(_reads_of_one, runs)) == {
        1: 18, 2: 144, 3: 540}
    assert all(_reads_of_one(run) >= 2 for run in failing)


@pytest.mark.parametrize("buyers,schedules", [(2, 2), (3, 6)])
def test_every_schedule_of_the_control_passes(buyers, schedules):
    # An atomic buyer makes one call, so the schedules are the orders of
    # the buyers: 2! and 3!.
    runs = _runs(oversell.atomic(buyers=buyers))
    assert len(runs) == schedules
    assert all(run.violation is None for run in runs)


def test_exhaust_yields_each_schedule_once_in_name_order():
    schedules = [run.schedule for run in _runs(oversell.naive())]
    assert schedules[0] == _SERIAL
    assert schedules == sorted(set(schedules))


def test_exhaust_reaches_every_interleaving_of_unequal_agents():
    # Declared out of name order: names decide the order, not the mapping.
    scenario = _ringers({"c": 3, "a": 1, "b": 2})
    schedules = [run.schedule for run in _runs(scenario)]
    # One a, two b and three c can be ordered in 6! / (1! 2! 3!) = 60 ways.
    assert len(schedules) == 60
    assert schedules == sorted(set(itertools.permutations("abbccc")))


def test_agents_that_make_no_call_have_one_empty_schedule():
    async def idle(ctx: scheduler.AgentContext) -> None:
        pass

    scenario = scheduler.Scenario(
        name="idle",
        make_world=_Bell,
        agents={"a": idle, "b": idle},
        invariants=(lambda world: "always fails",),
    )
    # The empty schedule is still a run: it is checked and it can fail.
    assert _runs(scenario) == [scheduler.RunResult((), "always fails")]


@pytest.mark.parametrize(
    "invariants", [(lambda world: "always fails",), ()],
    ids=["failing", "passing"])
def test_exhaust_refuses_a_run_that_does_not_replay(invariants):
    # One next, not a list: the refusal must come before the run is handed
    # over, not when the search is asked for the run after it.
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="schedule 'a' of 'leaky'"):
        next(explore.exhaust(_leaky(invariants)))


def test_exhaust_refuses_a_run_whose_verdict_does_not_replay():
    entries = []

    class Log:
        """A world whose tool returns nothing and whose state is shared."""

        def record(self) -> None:
            entries.append(None)

    async def writer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record")

    def one_entry(world: Log) -> str | None:
        if len(entries) == 1:
            return None
        return f"{len(entries)} entries"

    scenario = scheduler.Scenario(
        name="shared-log",
        make_world=Log,
        agents={"a": writer},
        invariants=(one_entry,),
    )
    # The run and its replay make the same call with the same result. Only
    # the verdict differs: the run passes and its replay fails.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="schedule 'a' of 'shared-log'"):
        next(explore.exhaust(scenario))


def test_exhaust_refuses_a_prefix_whose_agent_is_not_waiting():
    quiet = set()
    runs = explore.exhaust(_ringers({"a": 2, "b": 2}, quiet))
    assert next(runs).schedule == ("a", "a", "b", "b")
    # The next schedule starts a b, and now b makes no call.
    quiet.add("b")
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 2"):
        next(runs)


def test_exhaust_refuses_a_run_that_ends_inside_its_prefix():
    quiet = set()
    runs = explore.exhaust(_ringers({"a": 2, "b": 2}, quiet))
    assert next(runs).schedule == ("a", "a", "b", "b")
    # The next schedule starts a b, and now nobody makes a call.
    quiet.update({"a", "b"})
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="inside the 2 steps"):
        next(runs)


def test_seeded_failure_is_one_of_the_exhaustive_failures():
    scenario = oversell.naive()
    failure = explore.explore(scenario, range(20))
    assert failure is not None
    assert failure.result in _runs(scenario)


def test_exhaust_replays_runs_whose_tool_raised():
    class Door:
        """Opens once. Pushing it again raises."""

        def __init__(self) -> None:
            self.open = False

        def push(self) -> None:
            if self.open:
                raise RuntimeError("already open")
            self.open = True

    async def pusher(ctx: scheduler.AgentContext) -> None:
        await ctx.call("push")

    scenario = scheduler.Scenario(
        name="door",
        make_world=Door,
        agents={"a": pusher, "b": pusher},
        invariants=(),
    )
    runs = _runs(scenario)
    assert [run.schedule for run in runs] == [("a", "b"), ("b", "a")]
    # Whoever pushes second is told so, in both runs and in their replays.
    assert all(
        [step.result for step in run.steps]
        == [None, scheduler.Raised("RuntimeError", "already open")]
        for run in runs)


def test_exhaust_refuses_a_tool_error_that_changes_between_runs():
    tickets = itertools.count()

    class Desk:
        def take(self) -> None:
            raise RuntimeError(f"ticket {next(tickets)} is void")

    async def taker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take")

    scenario = scheduler.Scenario(
        name="desk",
        make_world=Desk,
        agents={"a": taker},
        invariants=(),
    )
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="schedule 'a' of 'desk'"):
        next(explore.exhaust(scenario))


@pytest.mark.parametrize("buyers", [2, 3])
def test_every_schedule_of_the_control_sells_the_one_unit(buyers):
    # A control whose calls all failed would pass too, and show nothing.
    for run in _runs(oversell.atomic(buyers=buyers)):
        results = collections.Counter(step.result for step in run.steps)
        assert results == {True: 1, False: buyers - 1}
