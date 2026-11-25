"""Checks the serial-equivalence oracle on the oversell and on small worlds.

The oversell comes first, with its counts worked out by hand. The worlds
after it show what the oversell cannot: a failure that no agent reads, a
control whose schedules are not serial orders, and what the oracle does
not see.
"""

import collections
import dataclasses
import functools
import itertools
from collections.abc import Hashable

import pytest

from anyorder import explore
from anyorder import oracle
from anyorder import scheduler
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
