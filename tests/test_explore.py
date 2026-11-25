"""Checks schedule search and replay on the oversell and other scenarios."""

import collections
import dataclasses
import itertools
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest

from anyorder import configuration
from anyorder import explore
from anyorder import scheduler
from anyorder.scenarios import oversell

_SERIAL = ("buyer-1", "buyer-1", "buyer-1", "buyer-2")
_BOTH_READ_FIRST = (
    "buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2")
_ALWAYS_FAILS = scheduler.Invariant(
    "always-fails", lambda world: "always fails", every_step=False)


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
    assert "2 orders placed against a stock of 1" in (
        failure.result.violation.message)


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
    assert result.violation == scheduler.Violation(
        "no-oversell",
        "2 orders placed against a stock of 1: buyer-1, buyer-2",
        6)


@pytest.mark.parametrize("stock,buyers", [(1, 2), (2, 5), (3, 3)])
def test_control_never_oversells(stock, buyers):
    scenario = oversell.atomic(stock=stock, buyers=buyers)
    assert explore.explore(scenario, range(200)) is None


def test_replay_refuses_steps_left_over():
    with pytest.raises(scheduler.ReplayDivergenceError, match="ended after 4"):
        explore.replay(oversell.naive(), _SERIAL + ("buyer-2",))


def test_search_refuses_a_failure_that_does_not_replay():
    scenario = _leaky((_ALWAYS_FAILS,))
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
        invariants=(_ALWAYS_FAILS,),
    )
    # The empty schedule is still a run: it is checked and it can fail.
    assert _runs(scenario) == [scheduler.RunResult(
        (), scheduler.Violation("always-fails", "always fails", 0))]


@pytest.mark.parametrize(
    "invariants", [(_ALWAYS_FAILS,), ()],
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
        invariants=(
            scheduler.Invariant("one-entry", one_entry, every_step=False),
        ),
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


def test_exhaust_refuses_a_violation_seen_at_another_step_in_replay():
    asked = itertools.count(1)

    def patient(world: _Bell) -> str | None:
        """Fails from the third time it is asked, whatever the world."""
        return "asked too often" if next(asked) >= 3 else None

    scenario = dataclasses.replace(
        _ringers({"a": 2}),
        invariants=(
            scheduler.Invariant("patient", patient, every_step=True),))
    # The run is asked on the fresh world and after each of its two steps,
    # and fails at step 2. Its replay fails at once: the same steps and
    # the same message, at step 0.
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="schedule 'a a' of 'bell'"):
        next(explore.exhaust(scenario))


def test_two_buyer_oversell_is_seen_at_the_last_step():
    # The invariant first breaks when the second order is placed, and in a
    # failing run of two buyers that is the last of its six calls.
    failing = [
        run for run in _runs(oversell.naive()) if run.violation is not None]
    assert {run.violation.invariant for run in failing} == {"no-oversell"}
    assert {run.violation.step for run in failing} == {6}


def test_three_buyer_oversell_is_seen_at_the_second_order():
    # Two read 1 (144 runs of 7 steps): after the two reads, the second
    # order is the last of the four calls of those two buyers. The read of
    # 0 comes after it in 3 * 2 * 6 = 36 runs, which see the break at step
    # 6, and before it in the other 108, at step 7.
    # Three read 1 (540 runs of 9 steps): after the three reads, the second
    # order is the fourth of the six calls left when the first four are
    # those of two buyers: 3 pairs of buyers in 6 ways each, so
    # 6 * 18 = 108 runs, at step 7. In the other 6 * 72 = 432 it is the
    # fifth, at step 8. It is never the sixth: a third order follows it.
    # 36 at step 6, 108 + 108 = 216 at step 7, 432 at step 8.
    failing = [
        run for run in _runs(oversell.naive(buyers=3))
        if run.violation is not None]
    assert collections.Counter(run.violation.step for run in failing) == {
        6: 36, 7: 216, 8: 432}
    for run in failing:
        seen = run.steps[:run.violation.step]
        assert seen[-1].tool == "place_order"
        assert sum(step.tool == "place_order" for step in seen) == 2
        # The message is that of the moment the invariant broke.
        assert run.violation.message.startswith("2 orders placed")
    # A run goes on to its end after the violation.
    assert collections.Counter(len(run.steps) for run in failing) == {
        7: 144, 9: 540}


# A counter starts at 0. b and c each add 1 in one call. a reads r and
# then draws whether it sets the counter to r + 10: one time in four if r
# is 0, one time in two otherwise. Run one after another, in any order,
# the three leave 12 whenever a sets: r incs came before its read and the
# other 2 - r come after its set. An inc between the read and the set is
# lost, and the counter then ends below 12.
_RARELY = ((True, 1), (False, 3))
_EVENLY = ((True, 1), (False, 1))
_LOSES_AN_INC = ("a", "c", "a", "b")


class _Counter:
    """A counter that remembers whether it was ever set."""

    def __init__(self) -> None:
        self.value = 0
        self.was_set = False

    def read(self) -> int:
        return self.value

    def inc(self) -> None:
        self.value += 1

    def set(self, value: int) -> None:
        self.value = value
        self.was_set = True


async def _setter(ctx: scheduler.AgentContext) -> None:
    r = await ctx.call("read")
    if await ctx.draw(_RARELY if r == 0 else _EVENLY):
        await ctx.call("set", value=r + 10)


async def _adder(ctx: scheduler.AgentContext) -> None:
    await ctx.call("inc")


def _ends_at_12(counter: _Counter) -> str | None:
    if counter.was_set and counter.value != 12:
        return f"the counter ends at {counter.value}, not 12"
    return None


def _counting() -> scheduler.Scenario:
    return scheduler.Scenario(
        name="counting",
        make_world=_Counter,
        agents={"a": _setter, "b": _adder, "c": _adder},
        invariants=(
            scheduler.Invariant("ends-at-12", _ends_at_12, every_step=False),
        ),
    )


def test_search_draws_and_finds_the_lost_inc():
    failure = explore.explore(_counting(), range(20))
    # At seed 0 both incs come before the read. At seed 1 a reads 0, draws
    # that it sets, and c adds before it does.
    assert failure == explore.Failure(1, scheduler.RunResult(
        steps=(
            scheduler.Step("a", "read", {}, 0),
            scheduler.Step("c", "inc", {}, None),
            scheduler.Step("a", "set", {"value": 10}, None),
            scheduler.Step("b", "inc", {}, None),
        ),
        violation=scheduler.Violation(
            "ends-at-12", "the counter ends at 11, not 12", 4),
        draws=(scheduler.Drawn("a", _RARELY, True, 1),)))
    assert failure.result.schedule == _LOSES_AN_INC


def test_failure_that_drew_replays_from_its_schedule_and_its_draws():
    scenario = _counting()
    failure = explore.explore(scenario, range(20))
    assert failure is not None
    result = failure.result
    assert explore.replay(scenario, result.schedule, result.draws) == result


def test_schedule_alone_does_not_replay_a_run_that_drew():
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the draws end after 0 but agent 'a' asks for one more$"):
        explore.replay(_counting(), _LOSES_AN_INC)


def test_other_draw_under_the_same_schedule_is_another_run():
    scenario = _counting()
    stops = [scheduler.Drawn("a", _RARELY, False, 1)]
    # With the other draw a stops after its read, so the schedule that has
    # it set cannot even be followed.
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="^step 3 names 'a'"):
        explore.replay(scenario, _LOSES_AN_INC, stops)
    passed = explore.replay(scenario, ("a", "c", "b"), stops)
    assert passed.violation is None
    assert passed.draws == tuple(stops)


def test_replay_refuses_draws_left_over():
    held = [scheduler.Drawn("a", _RARELY, False, 1)] * 2
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the run made 1 draws but 2 were given$"):
        explore.replay(_counting(), ("a", "b", "c"), held)
    # A run that draws nothing leaves every draw over.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the run made 0 draws but 1 were given$"):
        explore.replay(oversell.naive(), _SERIAL, held[:1])


def test_replay_refuses_a_draw_held_at_another_step():
    # a draws once its read has returned, which is after step 1 here.
    for step in (0, 2):
        held = [scheduler.Drawn("a", _RARELY, False, step)]
        with pytest.raises(
                scheduler.ReplayDivergenceError,
                match=f"^draw 1 was made after 1 steps, not after {step}$"):
            explore.replay(_counting(), ("a", "b", "c"), held)


def test_replay_gives_a_draw_back_to_its_agent():
    held = [scheduler.Drawn("a", _EVENLY, True, 2)]
    run = explore.replay(_counting(), ("b", "a", "a", "c"), held)
    # a reads 1, is given that it sets, and sets 11 before the last inc.
    assert [step.tool for step in run.steps] == ["inc", "read", "set", "inc"]
    assert run.steps[2].args == {"value": 11}
    assert run.draws == tuple(held)
    assert run.violation is None


def test_uniform_search_loses_an_inc_five_times_in_36():
    # A random policy chooses evenly among the waiting agents, so a's read
    # is the first, second or third call of a run with a chance of 1/3
    # each, and it reads 0, 1 or 2.
    # * It reads 0 and sets one time in four. Its set then waits beside
    #   both incs and is chosen before either with 1/3: an inc is lost
    #   with 2/3.
    # * It reads 1 and sets one time in two. Its set waits beside one
    #   inc, which comes first with 1/2.
    # * It reads 2, and no inc is left to lose.
    # That is 1/3 * (1/4 * 2/3 + 1/2 * 1/2) = 1/3 * 5/12 = 5/36. Of 3600
    # seeds it is 500, with a standard deviation of 21. The count is
    # written out: the same seeds must fail under every Python.
    scenario = _counting()
    failed = sum(
        explore.explore(scenario, [seed]) is not None for seed in range(3600))
    assert failed == 501


def test_search_refuses_a_failure_whose_draws_do_not_replay():
    tickets = itertools.count()

    async def drawer(ctx: scheduler.AgentContext) -> None:
        # What it draws from outlives the run.
        await ctx.draw(((next(tickets), 1),))
        await ctx.call("ring")

    scenario = scheduler.Scenario(
        name="leaky-draw",
        make_world=_Bell,
        agents={"a": drawer},
        invariants=(_ALWAYS_FAILS,),
    )
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        explore.explore(scenario, range(1))
    assert str(caught.value) == (
        "draw 1 is from ((0, 1),), but agent 'a' asks to draw from "
        "((1, 1),)")


def test_exhaustive_search_does_not_run_a_scenario_that_draws():
    # Given no sampler, that is. Its runs have to share their draws, which
    # a table sampler gives them.
    with pytest.raises(
            scheduler.UnsupportedAwaitError,
            match="^agent 'a' awaited a draw, and the run has no sampler$"):
        next(explore.exhaust(_counting()))


def _ticketed(
    invariants: Sequence[scheduler.Invariant] = (),
) -> scheduler.Scenario:
    """A scenario whose one call carries a number that outlives the run."""
    tickets = itertools.count()

    class Board:
        def post(self, ticket: int) -> None:
            pass

    async def clerk(ctx: scheduler.AgentContext) -> None:
        await ctx.call("post", ticket=next(tickets))

    return scheduler.Scenario(
        name="ticketed",
        make_world=Board,
        agents={"a": clerk},
        invariants=invariants,
    )


def test_replay_takes_the_steps_of_a_run_for_its_schedule():
    scenario = oversell.naive()
    run = explore.replay(scenario, _BOTH_READ_FIRST)
    assert explore.replay(scenario, run.steps) == run
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the run ended after 6 steps but the schedule has 7$"):
        explore.replay(scenario, run.steps + run.steps[-1:])


def test_replay_of_steps_ends_a_run_at_the_first_call_that_is_another():
    scenario = oversell.naive()
    run = explore.replay(oversell.naive(stock=2), _BOTH_READ_FIRST)
    # With one unit both buyers read 1 and not 2, which a replay is told
    # only by the calls that come of it: the first write stores 0.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        explore.replay(scenario, run.steps)
    assert str(caught.value) == (
        "step 3 is write_stock(value=1), but agent 'buyer-1' waits to call "
        "write_stock(value=0)")
    # Given the names it makes all six calls, and is another run.
    assert explore.replay(scenario, run.schedule) != run


def test_search_holds_a_failure_against_its_steps_as_it_replays_it():
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        explore.explore(_ticketed((_ALWAYS_FAILS,)), range(1))
    assert str(caught.value) == (
        "step 1 is post(ticket=0), but agent 'a' waits to call "
        "post(ticket=1)")


def test_exhaustive_search_holds_each_run_against_its_steps():
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        next(explore.exhaust(_ticketed()))
    assert str(caught.value) == (
        "step 1 is post(ticket=0), but agent 'a' waits to call "
        "post(ticket=1)")


class _SetsAfter:
    """A table of the tests' own for the counter: whether a sets, by what
    it read."""

    def __init__(self, *sets: bool) -> None:
        self._sets = sets

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> bool:
        (read,) = (step for step in moment.steps if step.agent == agent)
        return self._sets[read.result]


@pytest.mark.parametrize(
    "sets", list(itertools.product([False, True], repeat=3)), ids=str)
def test_every_schedule_of_the_counter_under_each_table(sets):
    # a reads first, second or third among the three first calls, and
    # reads 0, 1 or 2. Where it sets, its set is one more call.
    # * It reads 0. If it sets, the set and the two incs come in 3! = 6
    #   orders, and an inc is lost in the 4 that do not start with the
    #   set. If not, the incs come in 2.
    # * It reads 1, after either inc: 2 ways. If it sets, the set and
    #   the other inc come in 2 orders, and the inc is lost in 1: 4
    #   schedules, 2 lose. If not, 2 schedules.
    # * It reads 2, after both incs in 2 orders, and nothing is left to
    #   lose whether it sets or not: 2 schedules.
    at_0, at_1, _ = sets
    scenario = _counting()
    runs = list(explore.exhaust(scenario, _SetsAfter(*sets)))
    assert len(runs) == (6 if at_0 else 2) + (4 if at_1 else 2) + 2
    lost = [run for run in runs if run.violation is not None]
    assert len(lost) == (4 if at_0 else 0) + (2 if at_1 else 0)
    # Each run was replayed from its steps and its draws before it came.
    for run in runs:
        assert len(run.draws) == 1
        assert explore.replay(scenario, run.steps, run.draws) == run
    assert len({run.schedule for run in runs}) == len(runs)


def test_exhaustive_search_under_a_table_needs_three_draws_for_all_it_runs():
    found = []
    for seed in range(8):
        table = configuration.TableSampler(seed)
        runs = list(explore.exhaust(_counting(), table))
        # One draw for each of the three places a can stand at, however
        # many schedules come to them.
        assert len(table.held) == 3
        found.append(
            (len(runs), sum(run.violation is not None for run in runs)))
    # Written out, as the draws of a seed are: tables 3 and 4 have a set
    # after it read 0 and after it read 1, and tables 0, 2 and 6 after
    # neither.
    assert found == [
        (6, 0), (8, 2), (6, 0), (12, 6), (12, 6), (8, 2), (6, 0), (10, 4)]


def test_exhaustive_search_refuses_a_sampler_that_is_no_table():
    # A seeded sampler draws in the order it is asked, and the search
    # asks it again in every run. With seed 0 it has a stop after it read
    # 0 in one run and set in another.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        list(explore.exhaust(_counting(), scheduler.SeededSampler(0)))
    assert str(caught.value) == (
        "agent 'a' of 'counting' drew False in one run and True in "
        "another, where it had seen and done the same: an exhaustive "
        "search needs a sampler that gives one draw at one place")
    # With seed 4 the run that draws otherwise cannot follow the start
    # of its schedule, and ends there.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 2 names 'a', which is not waiting"):
        list(explore.exhaust(_counting(), scheduler.SeededSampler(4)))
    # With seed 3 it happens to draw the same at each place, and then its
    # runs are those of one table: a stops wherever it reads.
    runs = list(explore.exhaust(_counting(), scheduler.SeededSampler(3)))
    assert len(runs) == 6
    assert not any(draw.value for run in runs for draw in run.draws)


def test_exhaustive_search_tells_draws_that_python_takes_for_equal_apart():
    class Loose:
        """Gives True at first and 1 from then on."""

        def __init__(self) -> None:
            self._given = itertools.chain([True], itertools.repeat(1))

        def sample(
            self, agent: str, request: Any, moment: scheduler.Moment
        ) -> Any:
            return next(self._given)

    # The agent sets either way, so every run is equal to the run a
    # table would give, as Python compares. An agent can tell True from
    # 1, and the search holds the draws apart as a trace file does.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^agent 'a' of 'counting' drew True in one run and 1 in "
                  "another"):
        list(explore.exhaust(_counting(), Loose()))
