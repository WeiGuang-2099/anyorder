"""Checks the serial-equivalence oracle on the oversell and on small worlds.

The oversell comes first, with its counts worked out by hand. The worlds
after it show what the oversell cannot: a failure that no agent reads, a
control whose schedules are not serial orders, and what the oracle does
not see.
"""

import collections
import dataclasses
import fractions
import functools
import inspect
import itertools
from collections.abc import Coroutine, Hashable
from typing import Any

import pytest

from anyorder import explore
from anyorder import oracle
from anyorder import scheduler
from anyorder import shrink
from anyorder import solve
from anyorder import tracefile
from anyorder.scenarios import lost_update
from anyorder.scenarios import oversell
from anyorder.scenarios import write_skew

_B1_B2 = ("buyer-1", "buyer-2")
_B2_B1 = ("buyer-2", "buyer-1")
_BOTH_READ_FIRST = (
    "buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2")
_ONE_THEN_THE_OTHER = (
    "buyer-1", "buyer-1", "buyer-1", "buyer-2", "buyer-2", "buyer-2")


class _Cells:
    """A world of two cells that are written and never read."""

    def __init__(self) -> None:
        self.x: str | None = None
        self.y: str | None = None

    def put(self, cell: str, value: str) -> None:
        setattr(self, cell, value)


def _cells(cells: _Cells) -> tuple[str | None, str | None]:
    return cells.x, cells.y


def _seen_by(
    scenario: scheduler.Scenario, snapshot: oracle.Snapshot
) -> scheduler.Scenario:
    """The scenario with a snapshot of the test's own."""
    return dataclasses.replace(scenario, snapshot=snapshot)


class _SwapInventory(oversell.Inventory):
    """An inventory whose stock can be written with a compare-and-set."""

    def swap_stock(self, expected: int, value: int) -> bool:
        if self.stock != expected:
            return False
        self.stock = value
        return True


async def _swapping_buyer(ctx: scheduler.AgentContext) -> None:
    """Buys with a compare-and-set, and reads again when it is refused."""
    while True:
        stock = await ctx.call("read_stock")
        if stock <= 0:
            return
        swapped = await ctx.call(
            "swap_stock", expected=stock, value=stock - 1)
        if swapped:
            await ctx.call("place_order", buyer=ctx.name)
            return


def _swapping(stock: int) -> scheduler.Scenario:
    """Two buyers that check and decrement with a compare-and-set."""
    return dataclasses.replace(
        oversell.naive(stock=stock),
        name="oversell-swap",
        make_world=functools.partial(_SwapInventory, stock),
        agents={"buyer-1": _swapping_buyer, "buyer-2": _swapping_buyer})


class _Ledger:
    """Two accounts, and the total an auditor filed."""

    def __init__(self) -> None:
        self.x = 1
        self.y = 0
        self.filed: int | None = None

    def take_x(self) -> None:
        self.x -= 1

    def give_y(self) -> None:
        self.y += 1

    def read_x(self) -> int:
        return self.x

    def read_y(self) -> int:
        return self.y

    def file(self, total: int) -> None:
        self.filed = total


def _ledger(ledger: _Ledger) -> tuple[int, int, int | None]:
    return ledger.x, ledger.y, ledger.filed


def _audit(*, files: bool) -> scheduler.Scenario:
    """A mover takes 1 from x and gives it to y while an auditor reads both.

    Args:
        files: Whether the auditor writes the total it read to the ledger.
    """
    async def mover(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take_x")
        await ctx.call("give_y")

    async def auditor(ctx: scheduler.AgentContext) -> None:
        total = await ctx.call("read_x") + await ctx.call("read_y")
        if files:
            await ctx.call("file", total=total)

    return scheduler.Scenario(
        name="audit",
        make_world=_Ledger,
        agents={"auditor": auditor, "mover": mover},
        invariants=(),
    )


def _wasteful(*, in_trace: bool) -> tuple[scheduler.Scenario, oracle.Snapshot]:
    """Naive buyers of two units, in a world that leaks from run to run.

    The world counts the writes that stored what was there already, over
    every run so far. A serial order makes no such write and a run that
    loses an update makes one, so only such a run comes out otherwise when
    it is run again.

    Args:
        in_trace: Where the count shows. If True, in what an order is
            answered with, which changes at the third such write, and not
            in the snapshot. If False, in the snapshot and in no step.
    """
    wasted = []

    class Inventory(oversell.Inventory):
        def write_stock(self, value: int) -> None:
            if value == self.stock:
                wasted.append(value)
            super().write_stock(value)

        def place_order(self, buyer: str) -> bool | None:
            super().place_order(buyer)
            return len(wasted) >= 3 if in_trace else None

    def snapshot(inventory: Inventory) -> Hashable:
        if in_trace:
            return oversell.snapshot(inventory)
        return inventory.stock, len(wasted)

    scenario = dataclasses.replace(
        oversell.naive(stock=2), make_world=functools.partial(Inventory, 2))
    return scenario, snapshot


def _counted(
    scenario: scheduler.Scenario
) -> tuple[scheduler.Scenario, list[None]]:
    """Returns the scenario and a list that grows by one with each run."""
    built: list[None] = []

    def make_world() -> object:
        built.append(None)
        return scenario.make_world()

    return dataclasses.replace(scenario, make_world=make_world), built


def _runs(
    scenario: scheduler.Scenario, limit: int = 1000
) -> list[scheduler.RunResult]:
    """Runs a whole exhaustive search, up to a limit."""
    search = explore.exhaust(scenario)
    runs = list(itertools.islice(search, limit))
    assert next(search, None) is None, f"more than {limit} runs"
    return runs


def _failed(
    serial: oracle.SerialOutcomes, runs: list[scheduler.RunResult]
) -> list[scheduler.RunResult]:
    """Keeps the runs that end as no serial order ends."""
    return [run for run in runs if oracle.judge(serial, run).failed]


def test_snapshot_of_an_inventory_is_its_stock_and_its_orders_as_placed():
    inventory = oversell.Inventory(5)
    for buyer in ("b", "a", "b"):
        inventory.place_order(buyer)
    inventory.write_stock(3)
    assert oversell.snapshot(inventory) == (3, ("b", "a", "b"))


def test_serial_orders_of_two_buyers_of_two_units_sell_both():
    # One after the other, the first buyer reads 2 and writes 1 and the
    # second reads 1 and writes 0. The orders are placed in that order.
    serial = oracle.serial_outcomes(oversell.naive(stock=2))
    assert serial.orders == {
        (0, _B1_B2): (_B1_B2,),
        (0, _B2_B1): (_B2_B1,),
    }


def test_serial_outcome_names_every_order_that_ends_in_it():
    # With one unit only the first buyer of a serial order buys, so the two
    # orders that start with the same buyer end alike.
    b1, b2, b3 = "buyer-1", "buyer-2", "buyer-3"
    serial = oracle.serial_outcomes(
        oversell.naive(buyers=3))
    assert serial.orders == {
        (0, (b1,)): ((b1, b2, b3), (b1, b3, b2)),
        (0, (b2,)): ((b2, b1, b3), (b2, b3, b1)),
        (0, (b3,)): ((b3, b1, b2), (b3, b2, b1)),
    }


def test_two_buyers_of_one_unit_fail_exactly_where_they_oversell():
    # tests/test_explore.py counts the 16 schedules. In 4 of them one buyer
    # reads 0 and stops: the stock ends at 0 with the other buyer's order,
    # as in the serial order that buyer starts. In 12 both read 1 and both
    # order, and no serial order leaves two orders.
    scenario = oversell.naive()
    serial = oracle.serial_outcomes(scenario)
    runs = _runs(scenario)
    failed = _failed(serial, runs)
    assert (len(runs), len(failed)) == (16, 12)
    assert failed == [run for run in runs if run.violation is not None]


def test_three_buyers_of_one_unit_fail_exactly_where_they_oversell():
    # tests/test_explore.py counts the 702 schedules. In 18 of them one
    # buyer reads 1 and the other two read 0: the stock ends at 0 with that
    # buyer's order, as in a serial order it starts. In 684 two or three
    # buyers order, and no serial order leaves more than one order.
    scenario = oversell.naive(buyers=3)
    serial = oracle.serial_outcomes(scenario)
    runs = _runs(scenario)
    failed = _failed(serial, runs)
    assert (len(runs), len(failed)) == (702, 684)
    assert failed == [run for run in runs if run.violation is not None]


def test_two_buyers_of_two_units_lose_an_update_in_12_of_20_schedules():
    # With a stock of 2 a buyer reads 2 or 1 and makes three calls either
    # way (read, write, order), so the schedules are the
    # 6! / (3! 3!) = 20 interleavings of two buyers.
    # Both reads come before either write: the reads in 2 orders, then the
    # four calls left (a write and an order each) in 4! / (2! 2!) = 6 ways.
    # 2 * 6 = 12 schedules. Both buyers read 2 and write 1, so the stock
    # ends at 1 with two orders placed, and every serial order ends at 0.
    # Otherwise one buyer reads and writes before the other reads, and its
    # order goes into one of 4 places among the other's three calls.
    # 2 * 4 = 8 schedules. The stock ends at 0 with both orders, and
    # whichever way round they were placed, a serial order places them so.
    # No schedule oversells: 2 orders against a stock of 2.
    scenario = oversell.naive(stock=2)
    serial = oracle.serial_outcomes(scenario)
    runs = _runs(scenario)
    failed = _failed(serial, runs)
    assert (len(runs), len(failed)) == (20, 12)
    assert all(run.violation is None for run in runs)
    assert failed == [
        run for run in runs
        if [step.tool for step in run.steps[:2]] == ["read_stock"] * 2]
    assert {oracle.judge(serial, run).outcome for run in failed} == {
        (1, _B1_B2), (1, _B2_B1)}


def test_run_is_judged_by_the_state_it_leaves_and_not_by_what_was_read():
    # buyer-1 reads and writes first and places its order last. The reads
    # are those of the serial order that buyer-1 starts, and the orders are
    # placed as in the one that buyer-2 starts. No serial order gives both,
    # and the run passes: the state it leaves is that of the second.
    scenario = oversell.naive(stock=2)
    serial = oracle.serial_outcomes(scenario)
    run = explore.replay(scenario, (
        "buyer-1", "buyer-1", "buyer-2", "buyer-2", "buyer-2", "buyer-1"))
    assert [
        (step.agent, step.result) for step in run.steps
        if step.tool == "read_stock"] == [("buyer-1", 2), ("buyer-2", 1)]
    assert oracle.judge(serial, run) == oracle.Verdict(
        (0, _B2_B1), (_B2_B1,))


@pytest.mark.parametrize("stock,buyers", [(1, 2), (2, 2), (1, 3), (2, 3)])
def test_every_schedule_of_the_control_ends_as_its_own_serial_order(
        stock, buyers):
    # An atomic buyer makes one call, so every schedule of the control is
    # a serial order. This shows that the oracle knows every serial order.
    # It shows nothing about interleavings: the swapping buyers do that.
    scenario = oversell.atomic(stock=stock, buyers=buyers)
    serial = oracle.serial_outcomes(scenario)
    runs = _runs(scenario)
    assert len(runs) == len(list(itertools.permutations(scenario.agents)))
    for run in runs:
        assert run.schedule in oracle.judge(serial, run).orders


@pytest.mark.parametrize("stock,schedules,refused", [(1, 16, 12), (2, 28, 20)])
def test_buyers_that_swap_and_read_again_pass_under_every_schedule(
        stock, schedules, refused):
    # A run starts with a read. Then either that buyer swaps before the
    # other reads, or both read before either swaps.
    # Stock 1. One swaps first: 2 choices of the buyer, and the other's
    # read of 0 goes before or after its order. 2 * 2 = 4. Both read 1: the
    # reads in 2 orders, either buyer swaps first and wins, and its order
    # goes into one of 3 places among the loser's two calls (the refused
    # swap and a read of 0). 2 * 2 * 3 = 12. 4 + 12 = 16 schedules.
    # Stock 2. One swaps first: 2 choices, no swap is refused, and its
    # order goes into one of 4 places among the other's three calls.
    # 2 * 4 = 8. Both read 2: the reads in 2 orders, either swaps first,
    # and its order goes into one of 5 places among the other's four calls
    # (the refused swap, a read of 1, a swap and an order).
    # 2 * 2 * 5 = 20. 8 + 20 = 28 schedules.
    # A swap is refused in the 12 and in the 20, which happens in no serial
    # order. Each of those runs still ends as a serial order ends: the
    # stock at 0, and the orders of the buyers whose swaps went through.
    scenario = _swapping(stock)
    serial = oracle.serial_outcomes(scenario)
    runs = _runs(scenario)
    assert len(runs) == schedules
    assert sum(
        any(step.result is False for step in run.steps)
        for run in runs) == refused
    assert all(run.violation is None for run in runs)
    assert _failed(serial, runs) == []


def test_writes_that_no_serial_order_leaves_fail_without_any_read():
    async def writer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("put", cell="x", value=ctx.name)
        await ctx.call("put", cell="y", value=ctx.name)

    scenario = scheduler.Scenario(
        name="writers",
        make_world=_Cells,
        agents={"a": writer, "b": writer},
        invariants=(),
    )
    # A cell ends with the name of whoever wrote it last, and in a serial
    # order that is the second agent for both cells. Of the
    # 4! / (2! 2!) = 6 schedules, a b b a and b a a b leave one cell to
    # each agent. No agent reads, so only the world shows it.
    serial = oracle.serial_outcomes(_seen_by(scenario, _cells))
    assert serial.orders == {
        ("b", "b"): (("a", "b"),),
        ("a", "a"): (("b", "a"),),
    }
    runs = _runs(scenario)
    assert len(runs) == 6
    assert {
        run.schedule: oracle.judge(serial, run).outcome
        for run in _failed(serial, runs)
    } == {
        ("a", "b", "b", "a"): ("b", "a"),
        ("b", "a", "a", "b"): ("a", "b"),
    }


def test_reads_that_no_serial_order_gives_are_not_seen():
    # One after the other, the auditor reads 1 and 0 or 0 and 1. Of the
    # 4! / (2! 2!) = 6 schedules, in one it reads x before the take and y
    # after the give, 1 and 1, and in one it reads both between the take
    # and the give, 0 and 0. Every run ends with x at 0 and y at 1, so all
    # six pass: the auditor leaves nothing in the world to judge.
    scenario = _audit(files=False)
    serial = oracle.serial_outcomes(_seen_by(scenario, _ledger))
    runs = _runs(scenario)
    reads = collections.Counter(
        tuple(
            step.result for step in run.steps if step.agent == "auditor")
        for run in runs)
    assert reads == {(1, 0): 2, (0, 1): 2, (1, 1): 1, (0, 0): 1}
    assert _failed(serial, runs) == []


def test_reads_that_are_written_to_the_world_are_seen():
    # The auditor now files the total it read, which is 1 in both serial
    # orders. There are 5! / (2! 3!) = 10 schedules. It files 2 when it
    # reads x before the take and y after the give: 1 schedule. It files 0
    # when it reads both between the take and the give, and the give and
    # the filing then come in either order: 2 schedules.
    scenario = _audit(files=True)
    serial = oracle.serial_outcomes(_seen_by(scenario, _ledger))
    assert set(serial.orders) == {(0, 1, 1)}
    runs = _runs(scenario)
    assert len(runs) == 10
    a, m = "auditor", "mover"
    assert {
        run.schedule: oracle.judge(serial, run).outcome
        for run in _failed(serial, runs)
    } == {
        (a, m, m, a, a): (0, 1, 2),
        (m, a, a, m, a): (0, 1, 0),
        (m, a, a, a, m): (0, 1, 0),
    }


def test_failure_that_every_serial_order_shares_is_not_seen():
    async def hasty_buyer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("write_stock", value=0)
        await ctx.call("place_order", buyer=ctx.name)

    scenario = dataclasses.replace(
        oversell.naive(),
        agents={"buyer-1": hasty_buyer, "buyer-2": hasty_buyer})
    # Buyers that do not look at the stock oversell one after the other
    # too, so all 4! / (2! 2!) = 6 schedules end as a serial order ends.
    # The invariant is the oracle for this.
    serial = oracle.serial_outcomes(scenario)
    runs = _runs(scenario)
    assert len(runs) == 6
    assert all(run.violation is not None for run in runs)
    assert _failed(serial, runs) == []


def test_agents_that_make_no_call_end_as_every_serial_order():
    async def idle(ctx: scheduler.AgentContext) -> None:
        pass

    # Declared out of name order: names decide the order, not the mapping.
    scenario = scheduler.Scenario(
        name="idle",
        make_world=_Cells,
        agents={"b": idle, "a": idle},
        invariants=(),
    )
    orders = (("a", "b"), ("b", "a"))
    serial = oracle.serial_outcomes(_seen_by(scenario, _cells))
    assert serial.orders == {(None, None): orders}
    assert oracle.judge(serial, scheduler.RunResult((), None)) == (
        oracle.Verdict((None, None), orders))


def test_each_serial_order_is_run_twice():
    # 3! = 6 orders, each run and replayed.
    scenario, built = _counted(oversell.naive(buyers=3))
    oracle.serial_outcomes(scenario)
    assert len(built) == 12


def test_passing_run_is_replayed_once_and_failing_run_twice():
    scenario, built = _counted(oversell.naive(stock=2))
    serial = oracle.serial_outcomes(scenario)
    passing = explore.replay(scenario, _ONE_THEN_THE_OTHER)
    failing = explore.replay(scenario, _BOTH_READ_FIRST)
    built.clear()
    assert not oracle.judge(serial, passing).failed
    assert len(built) == 1
    built.clear()
    assert oracle.judge(serial, failing).failed
    assert len(built) == 2


def test_failure_keeps_its_verdict_through_a_trace_file():
    # The file holds no verdict and no violation. The verdict is worked out
    # again from the run that the file replays to.
    scenario = oversell.naive(stock=2)
    serial = oracle.serial_outcomes(scenario)
    run = explore.replay(scenario, _BOTH_READ_FIRST)
    saved = tracefile.loads(tracefile.dumps(scenario, run))
    assert saved.violation is None
    verdict = oracle.judge(serial, tracefile.replay(scenario, saved))
    assert verdict == oracle.judge(serial, run)
    assert verdict == oracle.Verdict((1, _B1_B2), ())


def test_judge_refuses_a_run_of_another_scenario():
    # Under this schedule the same agents call the same tools in both
    # builds. The reads, the values written and the violation differ. The
    # run is held against its steps as it is replayed, so it is refused
    # at the first call that is another: buyer-1 read 1 and not 2, and
    # waits to write 0.
    serial = oracle.serial_outcomes(oversell.naive())
    run = explore.replay(oversell.naive(stock=2), _BOTH_READ_FIRST)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        oracle.judge(serial, run)
    assert str(caught.value) == (
        "step 3 is write_stock(value=1), but agent 'buyer-1' waits to call "
        "write_stock(value=0)")


def test_judge_refuses_a_run_of_another_scenario_that_would_pass():
    # With three units and with two, one buyer after the other ends with
    # both orders placed, and neither run breaks an invariant. Replayed
    # with two units the run ends as a serial order ends, so only its
    # steps tell that it is not a run of this scenario: the reads give 3
    # and 2 in one and 2 and 1 in the other, and so do the writes that
    # follow them.
    serial = oracle.serial_outcomes(
        oversell.naive(stock=2))
    run = explore.replay(oversell.naive(stock=3), _ONE_THEN_THE_OTHER)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        oracle.judge(serial, run)
    assert str(caught.value) == (
        "step 2 is write_stock(value=2), but agent 'buyer-1' waits to call "
        "write_stock(value=1)")


def test_judge_refuses_a_run_before_it_asks_the_snapshot_of_its_world():
    def buyer_of_the_unit(inventory: oversell.Inventory) -> str:
        (buyer,) = inventory.orders
        return buyer

    # Replayed with one unit, the schedule oversells, and this snapshot is
    # not written for two orders. The run is refused for what it is, not
    # a run of this scenario, and the snapshot's own error does not hide
    # that.
    serial = oracle.serial_outcomes(
        _seen_by(oversell.naive(), buyer_of_the_unit))
    run = explore.replay(oversell.naive(stock=2), _BOTH_READ_FIRST)
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 3 is write_stock\\(value=1\\), but agent "
                  "'buyer-1' waits to call"):
        oracle.judge(serial, run)
    # A run whose calls are those of this scenario and whose results are
    # not is told from its record once the replay is over, and still
    # before the snapshot is asked: here the reads are said to give 2.
    own = explore.replay(oversell.naive(), _BOTH_READ_FIRST)
    steps = tuple(
        dataclasses.replace(step, result=2) if step.tool == "read_stock"
        else step for step in own.steps)
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="replayed to a different trace"):
        oracle.judge(serial, dataclasses.replace(own, steps=steps))


def test_judge_refuses_a_run_whose_violation_was_taken_out():
    scenario = oversell.naive()
    serial = oracle.serial_outcomes(scenario)
    run = explore.replay(scenario, _BOTH_READ_FIRST)
    assert run.violation is not None
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="a different trace"):
        oracle.judge(serial, dataclasses.replace(run, violation=None))


def test_judge_refuses_a_schedule_the_run_leaves():
    serial = oracle.serial_outcomes(oversell.naive())
    # With one unit buyer-2 reads 0 and makes no second call: the run has
    # four steps. Given one of them once more, the schedule is longer
    # than the run, and without the last it is shorter.
    run = explore.replay(oversell.naive(), _ONE_THEN_THE_OTHER[:4])
    longer = dataclasses.replace(run, steps=run.steps + run.steps[-1:])
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the run ended after 4 steps but the schedule has 5$"):
        oracle.judge(serial, longer)
    shorter = dataclasses.replace(run, steps=run.steps[:-1])
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the schedule ends after 3 steps but buyer-2 can still "
                  "run$"):
        oracle.judge(serial, shorter)
    # And a run of two units leaves it at the first call that is another.
    other = explore.replay(oversell.naive(stock=2), _ONE_THEN_THE_OTHER)
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 2 is write_stock\\(value=1\\), but agent "
                  "'buyer-1' waits to call write_stock\\(value=0\\)$"):
        oracle.judge(serial, other)


def test_judge_refuses_a_failure_that_ends_otherwise_when_run_again():
    scenario, snapshot = _wasteful(in_trace=False)
    serial = oracle.serial_outcomes(_seen_by(scenario, snapshot))
    run = explore.replay(scenario, _BOTH_READ_FIRST)
    # The trace replays: only the state tells one run from the next.
    assert explore.replay(scenario, _BOTH_READ_FIRST) == run
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="ended in another state when it was run again"):
        oracle.judge(serial, run)


def test_judge_refuses_a_failure_that_replays_otherwise_when_run_again():
    scenario, snapshot = _wasteful(in_trace=True)
    serial = oracle.serial_outcomes(_seen_by(scenario, snapshot))
    # The run makes the first write that changes nothing. The first replay
    # makes the second and gives the same trace. The replay made before
    # the failure is reported makes the third, and its last order is
    # answered otherwise. All three end in the same state.
    run = explore.replay(scenario, _BOTH_READ_FIRST)
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="replayed to a different trace"):
        oracle.judge(serial, run)


def test_judge_takes_a_passing_run_of_a_world_that_leaks_elsewhere():
    # The leak shows only in a run that loses an update, and this run
    # loses none. This snapshot leaves the orders out, so both serial
    # orders end alike.
    scenario, snapshot = _wasteful(in_trace=False)
    serial = oracle.serial_outcomes(_seen_by(scenario, snapshot))
    run = explore.replay(scenario, _ONE_THEN_THE_OTHER)
    assert oracle.judge(serial, run) == oracle.Verdict(
        (0, 0), (_B1_B2, _B2_B1))


def test_serial_outcomes_refuse_a_state_that_outlives_a_run():
    entries = []

    class Log:
        """A world whose tool returns nothing and whose state is shared."""

        def record(self) -> None:
            entries.append(None)

    async def writer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record")

    scenario = scheduler.Scenario(
        name="shared-log",
        make_world=Log,
        agents={"a": writer},
        invariants=(),
    )
    # The serial order and its replay make the same call with the same
    # result. One ends with 1 entry and the other with 2.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="serial order 'a' of 'shared-log' ended in another state"):
        oracle.serial_outcomes(_seen_by(scenario, lambda log: len(entries)))


def test_serial_outcomes_refuse_a_serial_order_that_does_not_replay():
    tickets = itertools.count()

    class Counter:
        def take(self) -> int:
            return next(tickets)

    async def taker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take")

    scenario = scheduler.Scenario(
        name="leaky",
        make_world=Counter,
        agents={"a": taker},
        invariants=(),
    )
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="schedule 'a' of 'leaky' replayed to a different trace"):
        oracle.serial_outcomes(_seen_by(scenario, lambda counter: None))


def test_snapshot_that_cannot_be_hashed_is_refused():
    # The world's own list, which the world goes on to change.
    with pytest.raises(TypeError, match="cannot be hashed"):
        oracle.serial_outcomes(_seen_by(
            oversell.naive(),
            lambda inventory: (inventory.stock, inventory.orders)))


def test_serial_orders_of_a_scenario_that_draws_are_not_run():
    async def writer(ctx: scheduler.AgentContext) -> None:
        value = await ctx.draw((("left", 1), ("right", 1)))
        await ctx.call("put", cell="x", value=value)

    scenario = scheduler.Scenario(
        name="drawn-cells",
        make_world=_Cells,
        agents={"a": writer, "b": writer},
        invariants=(),
    )
    # The serial orders must be run under the draws of the run they judge,
    # and nothing gives those yet to an order that asks for other draws.
    with pytest.raises(
            scheduler.UnsupportedAwaitError,
            match="^agent 'a' awaited a draw, and the run has no sampler$"):
        oracle.serial_outcomes(_seen_by(scenario, _cells))


def test_serial_orders_are_held_against_their_steps_when_they_are_run_again():
    tickets = itertools.count()

    async def writer(ctx: scheduler.AgentContext) -> None:
        # What it writes outlives the run.
        await ctx.call("put", cell="x", value=f"ticket {next(tickets)}")

    scenario = scheduler.Scenario(
        name="ticketed-cells",
        make_world=_Cells,
        agents={"a": writer},
        invariants=(),
    )
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        oracle.serial_outcomes(_seen_by(scenario, _cells))
    assert str(caught.value) == (
        "step 1 is put(cell='x', value='ticket 0'), but agent 'a' waits to "
        "call put(cell='x', value='ticket 1')")


@pytest.mark.parametrize(
    "scenario,snapshot",
    [
        (oversell.naive(), oversell.snapshot),
        (oversell.atomic(), oversell.snapshot),
        (lost_update.naive(), lost_update.snapshot),
        (lost_update.cas(), lost_update.snapshot),
        (write_skew.naive(), write_skew.snapshot),
        (write_skew.cas(), write_skew.snapshot),
        (write_skew.atomic(), write_skew.snapshot),
    ],
    ids=lambda value: getattr(value, "name", ""))
def test_scenario_of_the_package_brings_the_snapshot_of_its_world(
    scenario, snapshot
):
    assert scenario.snapshot is snapshot


@pytest.mark.parametrize(
    "scenario",
    [
        oversell.naive(),
        oversell.atomic(),
        lost_update.naive(),
        write_skew.naive(),
    ],
    ids=lambda scenario: scenario.name)
def test_serial_outcomes_hold_the_scenario_they_were_given(scenario):
    # The oracle goes by the snapshot a scenario brings: every serial
    # order of these has an outcome. The compare-and-set builds and the
    # control of the write skew are not run here, where nothing guards a
    # run: the tests of their scenarios ask for their serial outcomes
    # behind a guard.
    serial = oracle.serial_outcomes(scenario)
    assert serial.scenario is scenario
    assert sum(len(orders) for orders in serial.orders.values()) == 2


def test_scenario_with_no_snapshot_has_no_serial_outcomes():
    built = []

    def make_world() -> _Cells:
        built.append(_Cells())
        return built[-1]

    async def writer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("put", cell="x", value=ctx.name)

    scenario = scheduler.Scenario(
        name="unseen", make_world=make_world, agents={"a": writer},
        invariants=())
    assert scenario.snapshot is None
    with pytest.raises(scheduler.MissingSnapshotError) as caught:
        oracle.serial_outcomes(scenario)
    assert str(caught.value) == (
        "the scenario 'unseen' has no snapshot, so its runs leave no "
        "outcome to compare")
    # Refused before anything ran.
    assert not built


def test_run_is_judged_with_the_snapshot_of_the_scenario_it_is_a_run_of():
    # The serial outcomes hold their scenario, and the scenario its
    # snapshot: one snapshot for the orders and for the run. Seen by its
    # stock alone, the run that loses an update ends as no order ends, as
    # it does when the orders are seen too.
    scenario = _seen_by(
        oversell.naive(stock=2), lambda inventory: inventory.stock)
    serial = oracle.serial_outcomes(scenario)
    assert serial.orders == {0: (_B1_B2, _B2_B1)}
    lost = explore.replay(scenario, _BOTH_READ_FIRST)
    assert oracle.judge(serial, lost) == oracle.Verdict(1, ())


def test_run_that_drew_is_not_judged():
    built = []

    class Dial:
        """A number that can be read, turned up by one and set."""

        def __init__(self) -> None:
            built.append(self)
            self.value = 0

        def read(self) -> int:
            return self.value

        def turn(self) -> None:
            self.value += 1

        def mark(self, value: int) -> None:
            self.value = value

    class Seven:
        def sample(
            self, agent: str, request: object, moment: scheduler.Moment
        ) -> int:
            return 7

    async def watcher(ctx: scheduler.AgentContext) -> None:
        before = await ctx.call("read")
        after = await ctx.call("read")
        # Only when somebody turned the dial between its two reads, which
        # happens in no serial order.
        if before != after:
            await ctx.call("mark", value=await ctx.draw("coin"))

    async def turner(ctx: scheduler.AgentContext) -> None:
        await ctx.call("turn")

    scenario = scheduler.Scenario(
        name="dial", make_world=Dial, agents={"a": watcher, "b": turner},
        invariants=(), snapshot=lambda dial: dial.value)
    # No agent draws in a serial order, so the scenario has its serial
    # outcomes. A run of it can still draw.
    serial = oracle.serial_outcomes(scenario)
    run = scheduler.run(scenario, scheduler.ReplayPolicy("abaa"), Seven())
    assert len(run.draws) == 1
    built.clear()
    # The serial orders would have to be run under the draws of the run,
    # which nothing gives them yet. Said so, and before anything runs: a
    # replay without the draws would call it a run that left its draws.
    with pytest.raises(
            ValueError,
            match="^the run made draws, and a run that drew is not judged "
                  "yet$"):
        oracle.judge(serial, run)
    assert not built


_NO_SERIAL_ORDER = "serial-outcome"


def _judged_two_units() -> tuple[scheduler.Scenario, scheduler.Scenario]:
    """Two buyers of two units, as they are and with the oracle's check."""
    base = oversell.naive(stock=2, buyers=2)
    return base, oracle.judged(base)


def test_judged_scenario_is_the_scenario_with_one_more_check_at_the_end():
    base, judged = _judged_two_units()
    assert judged.name == base.name
    # Its agents are the scenario's, each behind a watch for a draw.
    assert list(judged.agents) == list(base.agents)
    assert judged.make_world is base.make_world
    assert judged.snapshot is base.snapshot
    *own, added = judged.invariants
    assert tuple(own) == tuple(base.invariants)
    assert added.name == _NO_SERIAL_ORDER
    assert added.every_step is False
    # The serial orders were run for it, each twice, and no run since.
    assert added.check(base.make_world()) == (
        "the run left (2, ()), which no serial order leaves")


def test_judged_scenario_fails_exactly_where_the_oracle_says_so():
    base, judged = _judged_two_units()
    serial = oracle.serial_outcomes(base)
    runs = list(explore.exhaust(judged))
    assert len(runs) == 20
    failing = [run for run in runs if run.violation is not None]
    # The 12 schedules in which both read before either writes: the stock
    # ends at 1, where every serial order ends at 0.
    assert len(failing) == 12
    for run in runs:
        plain = explore.replay(base, run.schedule)
        assert plain.violation is None
        assert plain.steps == run.steps
        assert oracle.judge(serial, plain).failed == (
            run.violation is not None)
    # The check is asked once every agent has finished, and says what the
    # run left.
    assert {run.violation for run in failing} == {
        scheduler.Violation(
            _NO_SERIAL_ORDER,
            f"the run left (1, {orders!r}), which no serial order leaves", 6)
        for orders in (_B1_B2, _B2_B1)}


def test_search_finds_a_failure_that_only_the_oracle_sees():
    base, judged = _judged_two_units()
    assert explore.explore(base, range(2000)) is None
    failure = explore.explore(judged, range(20))
    assert failure is not None
    assert failure.seed == 4
    assert failure.result.schedule == (
        "buyer-1", "buyer-2", "buyer-1", "buyer-2", "buyer-2", "buyer-1")
    assert failure.result.violation == scheduler.Violation(
        _NO_SERIAL_ORDER,
        "the run left (1, ('buyer-2', 'buyer-1')), which no serial order "
        "leaves", 6)


def test_failure_that_only_the_oracle_sees_is_shrunk_and_written():
    _, judged = _judged_two_units()
    failing = [
        run for run in explore.exhaust(judged) if run.violation is not None]
    other_state = 0
    for run in failing:
        shrunk = shrink.shrink(judged, run).result
        # One buyer reads, the other runs to its end, the first finishes:
        # 2 switches and 2 buyers, and no failure is smaller. The check is
        # asked at the end, so every step of the run counts.
        assert shrink.size(shrunk) == (2, 2)
        assert shrunk.violation.invariant == _NO_SERIAL_ORDER
        other_state += shrunk.violation.message != run.violation.message
        # The violation holds the state the verdict rests on, so a trace
        # file of the run holds the verdict.
        text = tracefile.dumps(judged, shrunk)
        assert tracefile.replay(judged, tracefile.loads(text)) == shrunk
    # A shrunk run breaks the same check and need not leave the same
    # state: the two orders are placed the other way round in 4 of the 12.
    assert other_state == 4


def test_trace_file_of_a_judged_failure_replays_on_the_judged_scenario():
    base, judged = _judged_two_units()
    run = explore.replay(judged, _BOTH_READ_FIRST)
    saved = tracefile.loads(tracefile.dumps(judged, run))
    assert tracefile.replay(judged, saved) == run
    # The scenario as it was has no such check, so its run ends otherwise
    # than the file says.
    with pytest.raises(scheduler.ReplayDivergenceError):
        tracefile.replay(base, saved)


def test_solver_values_a_failure_that_only_the_oracle_sees():
    base, judged = _judged_two_units()
    assert solve.adaptive(base).value == 0
    assert solve.adaptive(judged).value == 1
    # After the first read the two wait side by side. With 1/2 the same
    # buyer writes, and what follows is a serial order. With 1/2 the other
    # reads 2 as well, and an update is lost.
    assert solve.uniform(judged) == fractions.Fraction(1, 2)
    assert solve.clairvoyant(judged) == 1


def test_judged_scenario_reports_its_own_invariant_first():
    base = oversell.naive(stock=1, buyers=2)
    judged = oracle.judged(base)
    # With one unit the same 12 schedules oversell, and a run keeps its
    # first violation: the scenario's own, asked after every step.
    assert [run.violation for run in explore.exhaust(judged)] == [
        run.violation for run in explore.exhaust(base)]


@pytest.mark.parametrize(
    "base",
    [oversell.atomic(stock=2, buyers=2), lost_update.naive(editors=2),
     write_skew.naive(doctors=2)],
    ids=lambda scenario: scenario.name)
def test_judged_scenario_adds_no_failure_where_the_two_oracles_agree(base):
    assert list(explore.exhaust(oracle.judged(base))) == list(
        explore.exhaust(base))


def test_scenario_is_not_judged_without_a_snapshot_or_with_draws():
    with pytest.raises(
            scheduler.MissingSnapshotError,
            match="^the scenario 'oversell-naive' has no snapshot, so its "
                  "runs leave no outcome to compare$"):
        oracle.judged(dataclasses.replace(oversell.naive(), snapshot=None))

    async def flipper(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("coin")

    drawing = scheduler.Scenario(
        name="flipping", make_world=_Cells, agents={"a": flipper},
        invariants=(), snapshot=_cells)
    # The serial orders are run with no sampler: with draws they would
    # have to be those of the run that is judged. They are run with the
    # agents as they were given, so the refusal is the scheduler's own.
    with pytest.raises(
            scheduler.UnsupportedAwaitError,
            match="^agent 'a' awaited a draw, and the run has no sampler$"):
        oracle.judged(drawing)


_ONE_OR_FIFTY = ((1, 1), (50, 1))


class _Cell:
    """One number that is read, raised by one and put."""

    def __init__(self) -> None:
        self.value = 0

    def read(self) -> int:
        return self.value

    def inc(self) -> None:
        self.value += 1

    def put(self, value: int) -> None:
        self.value = value


def _torn(between_the_reads: scheduler.Agent) -> scheduler.Scenario:
    """A reader that does more only where an inc came between its reads.

    One after another the two leave 1, whichever comes first, and the
    reader does no more than read.
    """
    async def reader(ctx: scheduler.AgentContext) -> None:
        first = await ctx.call("read")
        if first != await ctx.call("read"):
            await between_the_reads(ctx)

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("inc")

    return scheduler.Scenario(
        name="torn", make_world=_Cell, agents={"a": reader, "b": adder},
        invariants=(), snapshot=lambda cell: cell.value)


async def _puts_a_draw(ctx: scheduler.AgentContext) -> None:
    await ctx.call("put", value=await ctx.draw(_ONE_OR_FIFTY))


async def _puts_a_draw_made_by_hand(ctx: scheduler.AgentContext) -> None:
    await ctx.call("put", value=await scheduler.Draw(_ONE_OR_FIFTY))


@pytest.mark.parametrize(
    "between_the_reads", [_puts_a_draw, _puts_a_draw_made_by_hand],
    ids=["through-the-context", "by-hand"])
def test_run_of_a_judged_scenario_ends_where_an_agent_asks_for_a_draw(
    between_the_reads
):
    torn = _torn(between_the_reads)
    # No serial order draws, so the serial orders are run and nothing
    # refuses the scenario.
    judged = oracle.judged(torn)
    for schedule in (["a", "a", "b"], ["b", "a", "a"]):
        assert explore.replay(judged, schedule).violation is None
    # With the inc between its reads the reader draws, and the scenario
    # as it was goes on with what it drew: it can leave 50, which no
    # serial order leaves.
    sampler = scheduler.SeededSampler(0)
    drawn = scheduler.run(
        torn, scheduler.ReplayPolicy(["a", "b", "a", "a"]), sampler)
    assert [draw.request for draw in drawn.draws] == [_ONE_OR_FIFTY]
    # The oracle does not judge a run that drew, and its check as an
    # invariant sees the world alone. So a run of the judged scenario ends
    # at the draw, before the sampler is asked.
    refusal = (
        "^agent 'a' of 'torn' awaited a draw, and a run that drew is not "
        "judged yet$")
    with pytest.raises(scheduler.UnsupportedAwaitError, match=refusal):
        scheduler.run(
            judged, scheduler.ReplayPolicy(["a", "b", "a"]), sampler)
    with pytest.raises(scheduler.UnsupportedAwaitError, match=refusal):
        explore.explore(judged, range(20))
    for value in (solve.uniform, solve.adaptive, solve.clairvoyant):
        with pytest.raises(scheduler.UnsupportedAwaitError, match=refusal):
            value(judged)


class _Listening:
    """A sampler that keeps who asked it for a draw, and gives 1."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def sample(
        self, agent: str, request: object, moment: scheduler.Moment
    ) -> int:
        self.asked.append(agent)
        return 1


def test_sampler_is_not_asked_for_a_draw_of_a_judged_scenario():
    sampler = _Listening()
    judged = oracle.judged(_torn(_puts_a_draw))
    with pytest.raises(scheduler.UnsupportedAwaitError):
        scheduler.run(judged, scheduler.ReplayPolicy(["a", "b", "a"]), sampler)
    assert sampler.asked == []


def test_every_agent_of_a_judged_scenario_is_watched_as_it_is_listed():
    async def reader(ctx: scheduler.AgentContext) -> None:
        first = await ctx.call("read")
        if first != await ctx.call("read"):
            await ctx.draw(_ONE_OR_FIFTY)

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("inc")

    # The agent that draws has the last name here, and is listed first.
    judged = oracle.judged(scheduler.Scenario(
        name="torn", make_world=_Cell, agents={"b": reader, "a": adder},
        invariants=(), snapshot=lambda cell: cell.value))
    assert list(judged.agents) == ["b", "a"]
    with pytest.raises(
            scheduler.UnsupportedAwaitError,
            match="^agent 'b' of 'torn' awaited a draw, and a run that drew "
                  "is not judged yet$"):
        scheduler.run(
            judged, scheduler.ReplayPolicy(["b", "a", "b"]),
            scheduler.SeededSampler(0))


def test_draw_of_a_kind_of_its_own_is_a_draw_for_the_watch():
    class Marked(scheduler.Draw):
        """A draw that is more than a draw."""

    async def puts_a_marked_draw(ctx: scheduler.AgentContext) -> None:
        await ctx.call("put", value=await Marked(_ONE_OR_FIFTY))

    torn = _torn(puts_a_marked_draw)
    sampler = scheduler.SeededSampler(0)
    # The scheduler takes it for a draw, so the watch must as well.
    drawn = scheduler.run(
        torn, scheduler.ReplayPolicy(["a", "b", "a", "a"]), sampler)
    assert len(drawn.draws) == 1
    with pytest.raises(
            scheduler.UnsupportedAwaitError, match="is not judged yet$"):
        scheduler.run(
            oracle.judged(torn), scheduler.ReplayPolicy(["a", "b", "a"]),
            sampler)


def test_draw_after_a_call_that_gave_nothing_ends_a_judged_run():
    async def incs_and_draws(ctx: scheduler.AgentContext) -> None:
        # The answer before the draw is None, where a read gives a number.
        assert await ctx.call("inc") is None
        await ctx.call("put", value=await ctx.draw(_ONE_OR_FIFTY))

    judged = oracle.judged(_torn(incs_and_draws))
    with pytest.raises(
            scheduler.UnsupportedAwaitError, match="is not judged yet$"):
        scheduler.run(
            judged, scheduler.ReplayPolicy(["a", "b", "a", "a"]),
            scheduler.SeededSampler(0))


def test_judged_scenario_takes_a_call_that_is_also_a_draw_for_a_call():
    class Both(scheduler.Effect, scheduler.Draw):
        """A call, since the scheduler asks first whether it is one."""

    async def by_hand(ctx: scheduler.AgentContext) -> None:
        await Both("inc", {})

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("inc")

    base = scheduler.Scenario(
        name="cell", make_world=_Cell, agents={"a": by_hand, "b": adder},
        invariants=(), snapshot=lambda cell: cell.value)
    sampler = _Listening()
    run = scheduler.run(base, scheduler.ReplayPolicy(["a", "b"]), sampler)
    assert [step.tool for step in run.steps] == ["inc", "inc"]
    assert run.draws == ()
    # So the agent drew nothing, and behind the watch it acts as it did.
    assert scheduler.run(
        oracle.judged(base), scheduler.ReplayPolicy(["a", "b"]),
        sampler) == run
    assert sampler.asked == []


class _Turning:
    """Something that says another thing of its class each time it is asked.

    The scheduler asks what an agent awaits whether it is a call and then
    whether it is a draw, each time by its class. This answers with the
    classes it was given, one for each asking, and with the last of them
    from then on.
    """

    request = _ONE_OR_FIFTY
    tool = "inc"
    args: dict[str, object] = {}

    def __init__(self, *says: type) -> None:
        self._says = list(says)

    @property
    def __class__(self) -> type:
        if len(self._says) > 1:
            return self._says.pop(0)
        return self._says[0]

    def __await__(self):
        return (yield self)


def test_judged_scenario_ends_at_what_the_scheduler_takes_for_a_draw():
    async def awaits_what_is_late(ctx: scheduler.AgentContext) -> None:
        await _Turning(object, scheduler.Draw)

    torn = _torn(awaits_what_is_late)
    sampler = _Listening()
    # Asked the second time, it is a draw: the scheduler makes it.
    run = scheduler.run(torn, scheduler.ReplayPolicy(["a", "b", "a"]), sampler)
    assert [drawn.agent for drawn in run.draws] == ["a"]
    assert sampler.asked == ["a"]
    # The watch asks as the scheduler does, in the same order.
    with pytest.raises(
            scheduler.UnsupportedAwaitError, match="is not judged yet$"):
        scheduler.run(
            oracle.judged(torn), scheduler.ReplayPolicy(["a", "b", "a"]),
            sampler)
    assert sampler.asked == ["a"]


def test_judged_scenario_hands_on_nothing_that_could_be_a_draw_later():
    async def awaits_what_is_later(ctx: scheduler.AgentContext) -> None:
        await _Turning(object, object, scheduler.Draw)

    torn = _torn(awaits_what_is_later)
    sampler = _Listening()
    # Asked twice, it is neither a call nor a draw, and the scheduler
    # refuses it.
    with pytest.raises(scheduler.UnsupportedAwaitError) as plain:
        scheduler.run(torn, scheduler.ReplayPolicy(["a", "b", "a"]), sampler)
    # The watch has asked twice as well. It hands nothing of the agent's
    # on for the scheduler to ask again, and refuses in the same words.
    with pytest.raises(scheduler.UnsupportedAwaitError) as judged:
        scheduler.run(
            oracle.judged(torn), scheduler.ReplayPolicy(["a", "b", "a"]),
            sampler)
    assert str(judged.value) == str(plain.value) == (
        "agent 'a' awaited _Turning; an agent may only await "
        "AgentContext.call and AgentContext.draw")
    assert sampler.asked == []


def test_judged_scenario_hands_on_a_call_of_its_own_making():
    async def awaits_what_turns(ctx: scheduler.AgentContext) -> None:
        await _Turning(scheduler.Effect, scheduler.Draw)

    torn = _torn(awaits_what_turns)
    sampler = _Listening()
    schedule = ["a", "b", "a", "a"]
    # Asked the first time, it is a call, and the scheduler makes it.
    run = scheduler.run(torn, scheduler.ReplayPolicy(schedule), sampler)
    assert [step.tool for step in run.steps] == ["read", "inc", "read", "inc"]
    assert run.draws == ()
    # The watch asks once as well, and hands on a call of its own with
    # that tool and those arguments. Asked again, what the agent awaited
    # would say that it is a draw.
    judged = scheduler.run(
        oracle.judged(torn), scheduler.ReplayPolicy(schedule), sampler)
    assert judged.steps == run.steps
    assert judged.draws == ()
    assert sampler.asked == []
    # The run drew nothing, so it is judged: two incs where every serial
    # order has one.
    assert judged.violation == scheduler.Violation(
        _NO_SERIAL_ORDER, "the run left 2, which no serial order leaves", 4)


def test_judged_scenario_asks_no_third_time_whether_it_is_a_call():
    async def awaits_a_late_call(ctx: scheduler.AgentContext) -> None:
        await _Turning(object, object, scheduler.Effect)

    torn = _torn(awaits_a_late_call)
    for scenario in (torn, oracle.judged(torn)):
        with pytest.raises(
                scheduler.UnsupportedAwaitError,
                match="^agent 'a' awaited _Turning; an agent may only "
                      "await AgentContext.call and AgentContext.draw$"):
            scheduler.run(
                scenario, scheduler.ReplayPolicy(["a", "b", "a"]),
                _Listening())


def test_judged_scenario_refuses_an_await_of_nothing_as_the_scheduler():
    class Nothing:
        """Something that awaits None."""

        def __await__(self):
            return (yield None)

    async def waits(ctx: scheduler.AgentContext) -> None:
        await Nothing()

    torn = _torn(waits)
    for scenario in (torn, oracle.judged(torn)):
        with pytest.raises(
                scheduler.UnsupportedAwaitError,
                match="^agent 'a' awaited NoneType; an agent may only "
                      "await AgentContext.call and AgentContext.draw$"):
            scheduler.run(scenario, scheduler.ReplayPolicy(["a", "b", "a"]))


def test_judged_scenario_reads_the_tool_and_the_arguments_once():
    reads = []

    class Shifting(scheduler.Effect):
        """A call that names another tool and other arguments later."""

        def __init__(self) -> None:
            pass

        @property
        def tool(self) -> str:
            reads.append("tool")
            return "inc" if reads.count("tool") == 1 else "put"

        @property
        def args(self) -> dict[str, object]:
            reads.append("args")
            return {} if reads.count("args") == 1 else {"value": 40}

    async def shifts(ctx: scheduler.AgentContext) -> None:
        await Shifting()

    torn = _torn(shifts)
    schedule = ["a", "b", "a", "a"]
    run = scheduler.run(torn, scheduler.ReplayPolicy(schedule))
    assert [step.tool for step in run.steps] == ["read", "inc", "read", "inc"]
    # The scheduler reads the tool and then the arguments, each once.
    assert reads == ["tool", "args"]
    judged = oracle.judged(torn)
    reads.clear()
    # So does the watch, and what it hands on is read in its place.
    assert scheduler.run(
        judged, scheduler.ReplayPolicy(schedule)).steps == run.steps
    assert reads == ["tool", "args"]


class _Name(str):
    """A name that is more than a str."""


@pytest.mark.parametrize(
    "tool,args,text",
    [
        (_Name("inc"), {},
         "agent 'a' cannot call a tool named 'inc', of type _Name: a "
         "tool's name is a string"),
        # Arguments that are no mapping end a run, also where dict could
        # make something of them.
        ("inc", [("value", 1)],
         "agent 'a' cannot call 'inc': its arguments are of type list, not "
         "a mapping"),
    ],
    ids=["tool", "args"])
def test_judged_scenario_refuses_a_call_as_the_scheduler_refuses_it(
    tool, args, text
):
    async def calls(ctx: scheduler.AgentContext) -> None:
        await scheduler.Effect(tool, args)

    torn = _torn(calls)
    # The watch hands the tool and the arguments on as the agent gave
    # them, so the run ends behind it as it ends without it.
    for scenario in (torn, oracle.judged(torn)):
        with pytest.raises(scheduler.UnsupportedValueError) as caught:
            scheduler.run(scenario, scheduler.ReplayPolicy(["a", "b", "a"]))
        assert str(caught.value) == text


def test_judged_scenario_reads_no_arguments_of_a_refused_tool():
    class Made(scheduler.Effect):
        """A call of a tool with no plain name, whose arguments raise."""

        def __init__(self) -> None:
            pass

        @property
        def tool(self) -> str:
            return _Name("inc")

        @property
        def args(self) -> dict[str, object]:
            raise LookupError("the arguments were read")

    async def calls(ctx: scheduler.AgentContext) -> None:
        await Made()

    torn = _torn(calls)
    for scenario in (torn, oracle.judged(torn)):
        # The scheduler refuses the tool and never reads the arguments.
        with pytest.raises(
                scheduler.UnsupportedValueError,
                match="a tool's name is a string$"):
            scheduler.run(scenario, scheduler.ReplayPolicy(["a", "b", "a"]))


def test_judged_scenario_takes_what_says_it_is_a_call_as_the_scheduler_does():
    class Says:
        """Something that says by its class that it is a call."""

        tool = "inc"
        args: dict[str, object] = {}

        @property
        def __class__(self) -> type:
            return scheduler.Effect

        def __await__(self):
            return (yield self)

    async def by_hand(ctx: scheduler.AgentContext) -> None:
        await Says()

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("inc")

    base = scheduler.Scenario(
        name="cell", make_world=_Cell, agents={"a": by_hand, "b": adder},
        invariants=(), snapshot=lambda cell: cell.value)
    runs = list(explore.exhaust(base))
    assert [run.schedule for run in runs] == [("a", "b"), ("b", "a")]
    assert list(explore.exhaust(oracle.judged(base))) == runs


def test_error_of_an_agent_ends_a_judged_run_as_it_ends_any_run():
    async def fails(ctx: scheduler.AgentContext) -> None:
        raise LookupError("the agent's own")

    torn = _torn(fails)
    for scenario in (torn, oracle.judged(torn)):
        with pytest.raises(LookupError, match="^the agent's own$"):
            scheduler.run(scenario, scheduler.ReplayPolicy(["a", "b", "a"]))


def test_error_of_a_tool_reaches_the_agent_of_a_judged_scenario():
    class Brittle(_Cell):
        """A number that cannot be put."""

        def put(self, value: int) -> None:
            raise ValueError("no")

    async def putter(ctx: scheduler.AgentContext) -> None:
        if isinstance(await ctx.call("put", value=5), scheduler.Raised):
            await ctx.call("inc")

    base = scheduler.Scenario(
        name="brittle", make_world=Brittle, agents={"a": putter},
        invariants=(), snapshot=lambda cell: cell.value)
    (run,) = explore.exhaust(oracle.judged(base))
    assert [step.tool for step in run.steps] == ["put", "inc"]
    assert run == explore.replay(base, ["a", "a"])


def test_judged_run_ends_at_a_draw_whatever_the_agent_raises_as_it_closes():
    def clinging(error: BaseException) -> scheduler.Agent:
        async def clings(ctx: scheduler.AgentContext) -> None:
            try:
                await ctx.draw(_ONE_OR_FIFTY)
            finally:
                raise error

        return clings

    # An error of the agent's own is dropped as the scheduler drops it,
    # whatever its kind: the refusal came first.
    judged = oracle.judged(_torn(clinging(ValueError("mine"))))
    with pytest.raises(
            scheduler.UnsupportedAwaitError, match="is not judged yet$"):
        scheduler.run(judged, scheduler.ReplayPolicy(["a", "b", "a"]))
    # An interrupt is not dropped, as the scheduler does not drop one.
    judged = oracle.judged(_torn(clinging(KeyboardInterrupt())))
    with pytest.raises(KeyboardInterrupt):
        scheduler.run(judged, scheduler.ReplayPolicy(["a", "b", "a"]))


def test_judged_scenario_can_be_judged_again():
    once = oracle.judged(oversell.naive(stock=2, buyers=2))
    twice = oracle.judged(once)
    # An agent behind a watch is an agent, and can be watched.
    assert [
        (run.steps, run.violation is None) for run in explore.exhaust(twice)
    ] == [
        (run.steps, run.violation is None) for run in explore.exhaust(once)]


def test_agents_are_closed_when_a_judged_run_ends_at_a_draw():
    held = []

    def kept(agent: scheduler.Agent) -> scheduler.Agent:
        """Wraps an agent so that the test keeps the agent as it runs."""
        def start(ctx: scheduler.AgentContext) -> Coroutine[Any, Any, None]:
            held.append(agent(ctx))
            return held[-1]

        return start

    async def reader(ctx: scheduler.AgentContext) -> None:
        first = await ctx.call("read")
        if first != await ctx.call("read"):
            await ctx.draw(_ONE_OR_FIFTY)

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("inc")
        await ctx.call("inc")

    judged = oracle.judged(scheduler.Scenario(
        name="torn", make_world=_Cell,
        agents={"a": kept(reader), "b": kept(adder)},
        invariants=(), snapshot=lambda cell: cell.value))
    held.clear()
    with pytest.raises(
            scheduler.UnsupportedAwaitError, match="is not judged yet$"):
        scheduler.run(
            judged, scheduler.ReplayPolicy(["a", "b", "a"]),
            scheduler.SeededSampler(0))
    # The reader was stopped where it asked, and the adder where it waits
    # for its second call. The test still holds both, so nothing but the
    # watch has closed them.
    assert [inspect.getcoroutinestate(agent) for agent in held] == [
        inspect.CORO_CLOSED, inspect.CORO_CLOSED]


def test_judged_run_ends_at_a_draw_of_an_agent_that_does_not_close():
    async def clings(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.draw(_ONE_OR_FIFTY)
        finally:
            # No agent may make a call while it is closed. The error of
            # this one is dropped: the refusal came first.
            await ctx.call("inc")

    judged = oracle.judged(_torn(clings))
    with pytest.raises(
            scheduler.UnsupportedAwaitError,
            match="^agent 'a' of 'torn' awaited a draw, and a run that drew "
                  "is not judged yet$"):
        scheduler.run(judged, scheduler.ReplayPolicy(["a", "b", "a"]))


def test_judged_scenario_refuses_any_other_await_as_the_scheduler_does():
    class Renamed(type):
        """A metaclass whose classes say that they are named Draw."""

        @property
        def __name__(cls) -> str:
            return "Draw"

    class Other(metaclass=Renamed):
        """Something that is neither a call nor a draw."""

        def __await__(self):
            return (yield self)

    async def strays(ctx: scheduler.AgentContext) -> None:
        await Other()

    torn = _torn(strays)
    with pytest.raises(scheduler.UnsupportedAwaitError) as plain:
        scheduler.run(torn, scheduler.ReplayPolicy(["a", "b", "a"]))
    with pytest.raises(scheduler.UnsupportedAwaitError) as judged:
        scheduler.run(
            oracle.judged(torn), scheduler.ReplayPolicy(["a", "b", "a"]))
    # Both name the class as type holds its name, whatever it says.
    assert str(judged.value) == str(plain.value) == (
        "agent 'a' awaited Other; an agent may only await "
        "AgentContext.call and AgentContext.draw")


def test_agent_of_a_judged_scenario_can_be_awaited_as_any_agent():
    judged = oracle.judged(oversell.naive(stock=2, buyers=2))
    held = dict(judged.agents)

    def behind(name: str) -> scheduler.Agent:
        async def agent(ctx: scheduler.AgentContext) -> None:
            await held[name](ctx)

        return agent

    # An agent that is made of another awaits it. What it is handed for
    # each call comes back through the watch as the scheduler gave it.
    wrapped = dataclasses.replace(
        judged, agents={name: behind(name) for name in held})
    assert list(explore.exhaust(wrapped)) == list(explore.exhaust(judged))


def test_judged_violation_shows_no_more_than_the_start_of_a_long_state():
    async def writer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("put", cell="x", value="short")

    judged = oracle.judged(scheduler.Scenario(
        name="cells", make_world=_Cells, agents={"a": writer}, invariants=(),
        snapshot=_cells))
    # No run of the one agent fails, so the check is asked by hand, of a
    # world that no run leaves.
    assert all(run.violation is None for run in explore.exhaust(judged))
    world = _Cells()
    world.put("y", "y" * 500)
    shown = repr((None, "y" * 500))
    assert judged.invariants[-1].check(world) == (
        f"the run left {shown[:200]}... ({len(shown)} characters), which no "
        "serial order leaves")
