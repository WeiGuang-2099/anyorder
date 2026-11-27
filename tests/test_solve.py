"""Checks the exact values of small scenarios.

The plan's pinned example comes first, a counter with one agent that draws,
with each value worked out by hand. Small worlds of the tests' own follow,
then the examples of the package, held against a count over every schedule.
"""

import fractions
import itertools
from typing import Any

import pytest

from anyorder import explore
from anyorder import scheduler
from anyorder import solve
from anyorder.scenarios import lost_update
from anyorder.scenarios import oversell
from anyorder.scenarios import write_skew

Fraction = fractions.Fraction

# A counter starts at 0. Two agents each add 1 in one call. The third reads
# r and then draws whether it sets the counter to r + 10: one time in four
# if r is 0, one time in two otherwise. Run one after another, in any
# order, the three leave 12 whenever the third sets: r incs came before its
# read and the other 2 - r come after its set. An inc between the read and
# the set is lost, and the counter then ends below 12.
_RARELY = ((True, 1), (False, 3))
_EVENLY = ((True, 1), (False, 1))


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


def _counting(setter: str = "a") -> scheduler.Scenario:
    """The counter with its three agents, the one that sets under a name."""
    agents = {name: _adder for name in "abc"}
    agents[setter] = _setter
    return scheduler.Scenario(
        name="counting",
        make_world=_Counter,
        agents=agents,
        invariants=(
            scheduler.Invariant("ends-at-12", _ends_at_12, every_step=False),
        ),
        snapshot=lambda counter: (counter.value, counter.was_set),
    )


def test_no_serial_order_of_the_counter_loses_an_inc():
    scenario = _counting()
    # One after another, no inc comes between the read and the set.
    for order in itertools.permutations("abc"):
        assert solve.serial(scenario, order) == 0


def test_uniform_scheduling_loses_an_inc_five_times_in_36():
    # A uniform choice among the waiting agents makes a's read the first,
    # second or third call of a run with a chance of 1/3 each, so it reads
    # 0, 1 or 2.
    # * It reads 0 and sets one time in four. Its set then waits beside
    #   both incs and is chosen before either with 1/3: an inc is lost
    #   with 2/3.
    # * It reads 1 and sets one time in two. Its set waits beside one
    #   inc, which comes first with 1/2.
    # * It reads 2, and no inc is left to lose.
    # That is 1/3 * (1/4 * 2/3 + 1/2 * 1/2) = 1/3 * 5/12 = 5/36.
    assert solve.uniform(_counting()) == Fraction(5, 36)


def test_round_robin_in_name_order_loses_an_inc_one_time_in_four():
    # a reads 0, b adds, c adds, and it is a's turn again: it sets 10 one
    # time in four, after both incs, and the counter ends at 10.
    assert solve.round_robin(_counting()) == Fraction(1, 4)


def test_round_robin_goes_by_the_names_of_the_agents():
    # The same three agents under other names are another round. With the
    # setter second: an inc, the read of 1, an inc, then the set of 11 one
    # time in two, and the counter ends at 11. With the setter last: both
    # incs, the read of 2, and nothing is left to lose.
    assert solve.round_robin(_counting("b")) == Fraction(1, 2)
    assert solve.round_robin(_counting("c")) == 0
    # Uniform scheduling does not look at a name.
    assert solve.uniform(_counting("b")) == Fraction(5, 36)
    assert solve.uniform(_counting("c")) == Fraction(5, 36)


class _Tally:
    """A sum that agents add to, and whether it was ever odd."""

    def __init__(self) -> None:
        self.total = 0
        self.was_odd = False

    def add(self, amount: int) -> int:
        self.total += amount
        self.was_odd = self.was_odd or self.total % 2 == 1
        return self.total


def _tallying(
    invariant: scheduler.Invariant | None = None, **agents: scheduler.Agent
) -> scheduler.Scenario:
    return scheduler.Scenario(
        name="tally",
        make_world=_Tally,
        agents=agents,
        invariants=() if invariant is None else (invariant,),
        snapshot=lambda tally: (tally.total, tally.was_odd),
    )


def _never(total: int, *, every_step: bool = False) -> scheduler.Invariant:
    """The sum is never the given total, asked at the end or at every step."""
    def check(tally: _Tally) -> str | None:
        return f"the sum is {total}" if tally.total == total else None

    return scheduler.Invariant(f"never-{total}", check, every_step=every_step)


def test_draw_before_the_first_call_counts_by_its_weight():
    async def gambler(ctx: scheduler.AgentContext) -> None:
        if await ctx.draw(_RARELY):
            await ctx.call("add", amount=1)

    scenario = _tallying(_never(1), a=gambler)
    assert solve.uniform(scenario) == Fraction(1, 4)
    assert solve.serial(scenario, ["a"]) == Fraction(1, 4)
    assert solve.round_robin(scenario) == Fraction(1, 4)


def test_two_draws_in_a_row_count_by_the_product_of_their_weights():
    async def roller(ctx: scheduler.AgentContext) -> None:
        first = await ctx.draw(((1, 1), (2, 2)))
        second = await ctx.draw(((1, 1), (2, 2)))
        await ctx.call("add", amount=first + second)

    # The sum is 4 when both draws give 2, which is 2/3 of 2/3.
    assert solve.uniform(_tallying(_never(4), a=roller)) == Fraction(4, 9)
    # And 3 when the two differ, either way round: 2 * 1/3 * 2/3.
    assert solve.uniform(_tallying(_never(3), a=roller)) == Fraction(4, 9)
    assert solve.uniform(_tallying(_never(2), a=roller)) == Fraction(1, 9)


def test_outcome_that_is_written_twice_counts_twice():
    async def roller(ctx: scheduler.AgentContext) -> None:
        await ctx.call(
            "add", amount=await ctx.draw(((1, 1), (2, 1), (1, 2))))

    assert solve.uniform(_tallying(_never(1), a=roller)) == Fraction(3, 4)


def test_run_that_broke_an_invariant_on_its_way_has_failed():
    async def there_and_back(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=1)
        await ctx.call("add", amount=-1)

    # The sum is 1 after the first call and 0 at the end. What must hold
    # after every step was broken, though the world is whole again.
    scenario = _tallying(_never(1, every_step=True), a=there_and_back)
    assert solve.uniform(scenario) == 1
    assert solve.serial(scenario, ["a"]) == 1
    # Asked only at the end, the same rule holds.
    assert solve.uniform(_tallying(_never(1), a=there_and_back)) == 0


def test_scenario_with_no_agent_fails_as_its_fresh_world_does():
    assert solve.uniform(_tallying()) == 0
    assert solve.uniform(_tallying(_never(0))) == 1
    assert solve.uniform(_tallying(_never(0, every_step=True))) == 1
    assert solve.round_robin(_tallying(_never(0))) == 1
    assert solve.serial(_tallying(_never(0)), []) == 1


def test_serial_order_names_each_agent_once():
    scenario = _counting()
    for order in (["a", "b"], ["a", "b", "b"], ["a", "b", "c", "d"], "abcc"):
        with pytest.raises(
                ValueError,
                match="^a serial order of 'counting' names each of a, b, c "
                      "once$"):
            solve.serial(scenario, order)


def test_request_that_is_no_written_distribution_is_refused():
    async def guesser(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=1)
        await ctx.draw("coin")

    scenario = _tallying(a=guesser)
    for value in (
            solve.uniform, solve.round_robin,
            lambda given: solve.serial(given, ["a"])):
        with pytest.raises(
                scheduler.UnsupportedValueError,
                match="^agent 'a' cannot draw from 'coin': a written "
                      "distribution is pairs of an outcome and a weight, "
                      "each weight a positive integer$"):
            value(scenario)


def test_scenario_that_is_not_the_same_twice_is_refused():
    made = []

    class Drifting:
        """Answers with the number of worlds made so far."""

        def __init__(self) -> None:
            made.append(self)

        def ask(self) -> int:
            return len(made)

        def note(self, seen: int) -> None:
            pass

    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("note", seen=await ctx.call("ask"))

    async def teller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("note", seen=len(made))
        await ctx.call("note", seen=len(made))

    def drifting(agent: scheduler.Agent) -> scheduler.Scenario:
        return scheduler.Scenario(
            name="drifting", make_world=Drifting, agents={"a": agent},
            invariants=())

    # A point of a run is reached by running its start again. Here the
    # start is answered otherwise the second time.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^a run of 'drifting' replayed its start to a different "
                  "trace$"):
        solve.uniform(drifting(asker))
    made.clear()
    # And here the agent waits to make another call than the one it made.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 1 is note\\(seen=2\\), but agent 'a' waits to call "
                  "note\\(seen=3\\)$"):
        solve.uniform(drifting(teller))


def _caller(calls: int) -> scheduler.Agent:
    async def agent(ctx: scheduler.AgentContext) -> None:
        for _ in range(calls):
            await ctx.call("add", amount=0)

    return agent


def test_run_is_followed_for_as_many_choices_and_draws_as_the_limit():
    scenario = _tallying(a=_caller(5))
    assert solve.uniform(scenario, limit=5) == 0
    with pytest.raises(
            solve.TooLongError,
            match="^a run of 'tally' went past 4 choices and draws, and the "
                  "solver follows no run further: every run must end$"):
        solve.uniform(scenario, limit=4)
    # A draw counts as a choice does.
    async def flipper(ctx: scheduler.AgentContext) -> None:
        for _ in range(3):
            await ctx.draw(_EVENLY)
        await ctx.call("add", amount=0)

    assert solve.uniform(_tallying(a=flipper), limit=4) == 0
    with pytest.raises(solve.TooLongError):
        solve.uniform(_tallying(a=flipper), limit=3)


@pytest.mark.parametrize(
    "value",
    [
        solve.uniform,
        solve.round_robin,
        lambda scenario: solve.serial(scenario, ["a"]),
    ],
    ids=["uniform", "round-robin", "serial"])
def test_agent_that_never_ends_is_refused(value):
    async def caller(ctx: scheduler.AgentContext) -> None:
        while True:
            await ctx.call("add", amount=0)

    async def flipper(ctx: scheduler.AgentContext) -> None:
        while True:
            await ctx.draw(_EVENLY)

    for agent in (caller, flipper):
        with pytest.raises(solve.TooLongError, match="went past 200 choices"):
            value(_tallying(a=agent))


def test_too_long_is_a_refusal_of_the_scheduler():
    assert issubclass(solve.TooLongError, scheduler.SchedulerError)


class _Counted:
    """A policy that follows a schedule and keeps how many agents waited."""

    def __init__(self, schedule: tuple[str, ...]) -> None:
        self._schedule = iter(schedule)
        self.waited: list[int] = []

    def choose(self, moment: scheduler.Moment) -> str:
        self.waited.append(len(moment.waiting))
        return next(self._schedule)


def _uniform_over_schedules(scenario: scheduler.Scenario) -> Fraction:
    """The chance of a failing schedule, summed over every schedule.

    A uniform choice takes a schedule with the product of one over the
    number of agents that waited at each of its steps. For a scenario that
    does not draw and that cannot run for ever.
    """
    total = Fraction(0)
    for run in explore.exhaust(scenario):
        policy = _Counted(run.schedule)
        assert scheduler.run(scenario, policy) == run
        if run.violation is not None:
            chance = Fraction(1)
            for waited in policy.waited:
                chance /= waited
            total += chance
    return total


@pytest.mark.parametrize(
    "scenario,value",
    [
        (oversell.naive(stock=1, buyers=2), Fraction(1, 2)),
        (oversell.naive(stock=1, buyers=3), Fraction(2, 3)),
        (oversell.naive(stock=2, buyers=2), Fraction(0)),
        (lost_update.naive(editors=2), Fraction(1, 2)),
        (lost_update.naive(editors=3), Fraction(5, 6)),
        (write_skew.naive(doctors=2), Fraction(1, 2)),
        (write_skew.naive(doctors=3), Fraction(11, 18)),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_uniform_value_of_an_example_is_the_sum_over_its_schedules(
    scenario, value
):
    # For two buyers by hand: after the first read the two wait side by
    # side. With 1/2 the same buyer writes, and the other then reads 0
    # and buys nothing. With 1/2 the other reads 1 as well, and both buy.
    assert solve.uniform(scenario) == value
    assert _uniform_over_schedules(scenario) == value
    # One after another nothing is lost, whichever comes first.
    for order in itertools.permutations(sorted(scenario.agents)):
        assert solve.serial(scenario, order) == 0


@pytest.mark.parametrize(
    "scenario,calls,value",
    [
        # The limit is the number of calls of the longest run, so an agent
        # that tried again for ever would be refused and not hang the test.
        (oversell.atomic(stock=1, buyers=3), 3, Fraction(0)),
        (lost_update.cas(editors=2), 6, Fraction(0)),
        (lost_update.cas(editors=3), 12, Fraction(0)),
        (write_skew.cas(doctors=2), 4, Fraction(1, 2)),
        (write_skew.cas(doctors=3), 6, Fraction(11, 18)),
        (write_skew.atomic(doctors=2), 2, Fraction(0)),
        (write_skew.atomic(doctors=3), 3, Fraction(0)),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_uniform_value_of_a_control_and_of_the_fix_that_fails(
    scenario, calls, value
):
    assert solve.uniform(scenario, limit=calls) == value
    assert solve.round_robin(scenario, limit=calls) in (0, 1)
    with pytest.raises(solve.TooLongError):
        solve.uniform(scenario, limit=calls - 1)
