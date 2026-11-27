"""Checks the exact values of small scenarios.

The plan's pinned example comes first, a counter with one agent that draws,
with each value worked out by hand. Small worlds of the tests' own follow,
then the examples of the package, held against a count over every schedule.
"""

import dataclasses
import fractions
import inspect
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


def test_serial_order_of_other_things_than_names_is_refused():
    class Loud(str):
        """A name that is more than a string."""

    scenario = _counting()
    # Python cannot put a number in order beside a name, and what is more
    # than a string is no name of an agent.
    for order in (["a", "b", 3], ["a", "b", None], ["a", "b", Loud("c")]):
        with pytest.raises(
                ValueError,
                match="^a serial order of 'counting' names each of a, b, c "
                      "once$"):
            solve.serial(scenario, order)


def test_serial_order_is_walked_once():
    # An order that can be walked only once is an order all the same.
    assert solve.serial(_counting(), iter("abc")) == 0
    assert solve.serial(_counting(), (name for name in "cab")) == 0


@pytest.mark.parametrize(
    "order", [{"a", "b"}, frozenset({"a", "b"})], ids=["set", "frozenset"])
def test_serial_order_with_no_first_and_no_last_is_refused(order):
    async def adds_one(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=1)

    async def adds_two(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=2)

    # With a first the sum is 1 after the first call, with b first never.
    scenario = _tallying(
        _never(1, every_step=True), a=adds_one, b=adds_two)
    # A set of the names holds each of them once, and Python walks it in
    # an order that changes from one process to the next: the value would.
    with pytest.raises(ValueError) as caught:
        solve.serial(scenario, order)
    assert str(caught.value) == (
        "a serial order of 'tally' is walked first to last, and a set has "
        "no first: give a list or a tuple")
    # A dict has an order of its own, the one its names were put in.
    assert solve.serial(scenario, {"a": None, "b": None}) == 1
    assert solve.serial(scenario, {"b": None, "a": None}) == 0


def test_serial_order_for_no_agents_is_empty():
    assert solve.serial(_tallying(), ()) == 0
    with pytest.raises(
            ValueError,
            match="^a serial order of 'tally' names nobody: the scenario "
                  "has no agents$"):
        solve.serial(_tallying(), ["a"])


def test_request_that_is_no_written_distribution_is_refused():
    async def guesser(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=1)
        await ctx.draw("coin")

    scenario = _tallying(a=guesser)
    for value in (
            solve.uniform, solve.round_robin, solve.adaptive,
            solve.clairvoyant, lambda given: solve.serial(given, ["a"])):
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
            invariants=(), snapshot=lambda world: None)

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
    for value in (solve.adaptive, solve.clairvoyant):
        made.clear()
        with pytest.raises(
                scheduler.ReplayDivergenceError,
                match="^a run of 'drifting' replayed its start"):
            value(drifting(asker))


def _caller(calls: int) -> scheduler.Agent:
    async def agent(ctx: scheduler.AgentContext) -> None:
        for _ in range(calls):
            await ctx.call("add", amount=0)

    return agent


@pytest.mark.parametrize(
    "value",
    [
        solve.uniform,
        solve.round_robin,
        lambda scenario: solve.serial(scenario, ["a", "b"]),
        solve.adaptive,
        solve.clairvoyant,
    ],
    ids=["uniform", "round-robin", "serial", "adaptive", "clairvoyant"])
def test_run_whose_chosen_agent_no_longer_waits_is_refused(value):
    made = []

    class Numbered(_Tally):
        """A sum whose worlds are counted as they are made."""

        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    async def fading(ctx: scheduler.AgentContext) -> None:
        if len(made) <= 1:
            await ctx.call("add", amount=0)

    # In the first run a waits beside b. The second run is to go on with
    # a, which makes no call this time: nobody of that name waits.
    scenario = scheduler.Scenario(
        name="fading", make_world=Numbered,
        agents={"a": fading, "b": _caller(2)},
        invariants=(), snapshot=lambda tally: tally.total)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        value(scenario)
    assert str(caught.value) == (
        "a run of 'fading' left the start it was given to follow: agent 'a' "
        "does not wait where it waited before")


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
        solve.adaptive,
        solve.clairvoyant,
    ],
    ids=["uniform", "round-robin", "serial", "adaptive", "clairvoyant"])
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


@pytest.mark.parametrize(
    "value",
    [
        solve.uniform,
        solve.round_robin,
        lambda scenario, **how: solve.serial(scenario, ["a"], **how),
        solve.adaptive,
        solve.clairvoyant,
    ],
    ids=["uniform", "round-robin", "serial", "adaptive", "clairvoyant"])
def test_search_is_one_call_deeper_for_each_choice_of_a_run(value):
    depths = []

    class Deep(_Tally):
        """A sum that keeps how deep Python's stack is at each call."""

        def add(self, amount: int) -> int:
            depths.append(len(inspect.stack(0)))
            return super().add(amount)

    value(scheduler.Scenario(
        name="deep", make_world=Deep, agents={"a": _caller(4)},
        invariants=(), snapshot=lambda tally: tally.total), limit=4)
    # The run that makes the first call is made one decision into the
    # search, the run that makes the second call two, and so on. Each
    # decision costs the stack one call under every Python, so a run of n
    # choices is solved within n calls of where the search started, and
    # the limit can be set near Python's own on the depth of its stack.
    reached = sorted(set(depths))
    assert len(reached) == 4
    assert [deeper - before
            for before, deeper in zip(reached, reached[1:])] == [1, 1, 1]


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


def test_adaptive_scheduler_loses_an_inc_one_time_in_two():
    scenario = _counting()
    # A scheduler that sees everything so far chooses what a reads before
    # a draws. After a read of 0, a sets one time in four. After a read of
    # 1 it sets one time in two, and the other inc still waits beside the
    # set and is put before it: 1/2. After a read of 2 nothing is left to
    # lose. So the best scheduler runs one inc first, and no scheduler
    # that cannot see the draw to come does better.
    solved = solve.adaptive(scenario)
    assert solved.value == Fraction(1, 2)
    # With the value comes one failing run: the first the search met, with
    # the agents tried in name order and the outcomes as they are written.
    # Here a reads 0, draws that it sets, and b adds before it does.
    assert solved.failure == scheduler.RunResult(
        steps=(
            scheduler.Step("a", "read", {}, 0),
            scheduler.Step("b", "inc", {}, None),
            scheduler.Step("a", "set", {"value": 10}, None),
            scheduler.Step("c", "inc", {}, None),
        ),
        violation=scheduler.Violation(
            "ends-at-12", "the counter ends at 11, not 12", 4),
        draws=(scheduler.Drawn("a", _RARELY, True, 1),))
    failure = solved.failure
    assert explore.replay(scenario, failure.steps, failure.draws) == failure


def test_adaptive_value_is_at_least_that_of_every_scheduler_it_can_be():
    for setter in "abc":
        scenario = _counting(setter)
        # The names do not matter to a scheduler that may choose anybody.
        assert solve.adaptive(scenario).value == Fraction(1, 2)
        assert solve.uniform(scenario) < Fraction(1, 2)
        assert solve.round_robin(scenario) <= Fraction(1, 2)


@pytest.mark.parametrize(
    "scenario",
    [
        oversell.naive(stock=1, buyers=2),
        oversell.naive(stock=1, buyers=3),
        lost_update.naive(editors=2),
        lost_update.naive(editors=3),
        write_skew.naive(doctors=2),
        write_skew.naive(doctors=3),
    ],
    ids=lambda scenario: scenario.name)
def test_adaptive_value_of_an_example_that_fails_is_1(scenario):
    solved = solve.adaptive(scenario)
    assert solved.value == 1
    # Nothing is drawn, so the search goes through the schedules as an
    # exhaustive search does and meets the same failing run first.
    assert solved.failure == next(
        run for run in explore.exhaust(scenario) if run.violation is not None)


@pytest.mark.parametrize(
    "scenario,calls,value",
    [
        (oversell.naive(stock=2, buyers=2), 6, 0),
        (oversell.atomic(stock=1, buyers=3), 3, 0),
        (lost_update.cas(editors=2), 6, 0),
        (lost_update.cas(editors=3), 12, 0),
        (write_skew.cas(doctors=2), 4, 1),
        (write_skew.cas(doctors=3), 6, 1),
        (write_skew.atomic(doctors=2), 2, 0),
        (write_skew.atomic(doctors=3), 3, 0),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_adaptive_value_of_a_control_and_of_the_fix_that_fails(
    scenario, calls, value
):
    solved = solve.adaptive(scenario, limit=calls)
    assert solved.value == value
    # A failing run comes with a value above 0 and with no other.
    assert (solved.failure is None) == (value == 0)
    with pytest.raises(solve.TooLongError):
        solve.adaptive(scenario, limit=calls - 1)


def test_failing_run_is_replayed_before_it_is_returned():
    made = []

    class Numbered(_Tally):
        """A sum whose world knows how many were made before it."""

        def __init__(self) -> None:
            super().__init__()
            made.append(self)
            self.number = len(made)

    def numbered(tally: Numbered) -> str | None:
        return f"world {tally.number} ended at {tally.total}"

    scenario = scheduler.Scenario(
        name="numbered", make_world=Numbered, agents={"a": _caller(1)},
        invariants=(
            scheduler.Invariant("numbered", numbered, every_step=False),),
        snapshot=lambda tally: tally.total)
    # Every run fails, and each in other words than the one before. The
    # steps of a run replay, so the search goes through, and the failure
    # it would return does not.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^a failing run of 'numbered' replayed to a different "
                  "trace$"):
        solve.adaptive(scenario)
    assert solve.uniform(scenario) == 1


class _SetsAfter:
    """A table of the tests' own: whether a sets, by what it read."""

    def __init__(self, *sets: bool) -> None:
        self._sets = sets

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> bool:
        (read,) = (step for step in moment.steps if step.tool == "read")
        return self._sets[read.result]


def test_clairvoyant_scheduler_loses_an_inc_five_times_in_eight():
    scenario = _counting()
    # A table of draws holds one for each place a can stand at when it
    # draws: after a read of 0, of 1 and of 2. A scheduler that knows the
    # table lets a read 0 if the table has it set there, and 1 if it sets
    # there, and in both cases an inc is left to put before the set. A
    # set after a read of 2 loses nothing. So some schedule fails unless
    # the table has a stop at 0 and at 1: 1 - 3/4 * 1/2 = 5/8.
    assert solve.clairvoyant(scenario) == Fraction(5, 8)
    # The same from the eight tables, each searched through.
    total = Fraction(0)
    for sets in itertools.product([True, False], repeat=3):
        chance = Fraction(1 if sets[0] else 3, 4) / 2 / 2
        runs = list(explore.exhaust(scenario, _SetsAfter(*sets)))
        fails = any(run.violation is not None for run in runs)
        assert fails == (sets[0] or sets[1])
        total += chance * fails
    assert total == Fraction(5, 8)
    # Knowing the draws to come is worth more than seeing those made.
    assert solve.adaptive(scenario).value < Fraction(5, 8)


def test_clairvoyant_value_does_not_go_by_the_names():
    assert solve.clairvoyant(_counting("b")) == Fraction(5, 8)
    assert solve.clairvoyant(_counting("c")) == Fraction(5, 8)


def test_with_one_agent_every_scheduler_is_the_same():
    async def roller(ctx: scheduler.AgentContext) -> None:
        first = await ctx.draw(((1, 1), (2, 2)))
        second = await ctx.draw(((1, 1), (2, 2)))
        await ctx.call("add", amount=first + second)

    for total, chance in ((4, Fraction(4, 9)), (3, Fraction(4, 9)),
                          (2, Fraction(1, 9)), (5, Fraction(0))):
        scenario = _tallying(_never(total), a=roller)
        assert solve.clairvoyant(scenario) == chance
        assert solve.adaptive(scenario).value == chance
        assert solve.uniform(scenario) == chance


def test_table_holds_one_draw_for_each_place_and_not_for_each_draw():
    async def twice(ctx: scheduler.AgentContext) -> None:
        # Two draws from the same thing, at two places: the second comes
        # after the first.
        first = await ctx.draw(_EVENLY)
        second = await ctx.draw(_EVENLY)
        await ctx.call("add", amount=first + second)

    async def other(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=await ctx.draw(_EVENLY))

    # a adds 0, 1 or 2 and b adds 0 or 1, each draw apart from the others.
    # The sum is 3 when all three give True: 1/8. No schedule changes it.
    scenario = _tallying(_never(3), a=twice, b=other)
    assert solve.clairvoyant(scenario) == Fraction(1, 8)
    assert solve.adaptive(scenario).value == Fraction(1, 8)


@pytest.mark.parametrize(
    "scenario,calls",
    [
        (oversell.naive(stock=1, buyers=2), 6),
        (oversell.naive(stock=1, buyers=3), 9),
        (lost_update.naive(editors=3), 6),
        (write_skew.naive(doctors=3), 6),
        (write_skew.cas(doctors=3), 6),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_clairvoyant_value_of_an_example_that_fails_is_1(scenario, calls):
    # Nothing is drawn, so there is one table and nothing to know ahead:
    # the value is whether a schedule fails.
    assert solve.clairvoyant(scenario, limit=calls) == 1
    # Every schedule was tried for that, also after the first that fails,
    # and the longest among them.
    with pytest.raises(solve.TooLongError):
        solve.clairvoyant(scenario, limit=calls - 1)


@pytest.mark.parametrize(
    "scenario,calls",
    [
        (oversell.naive(stock=2, buyers=2), 6),
        (oversell.atomic(stock=1, buyers=3), 3),
        (lost_update.cas(editors=2), 6),
        (lost_update.cas(editors=3), 12),
        (write_skew.atomic(doctors=3), 3),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_clairvoyant_value_of_a_control_is_0(scenario, calls):
    assert solve.clairvoyant(scenario, limit=calls) == 0
    # Every schedule was tried for that, the longest among them.
    with pytest.raises(solve.TooLongError):
        solve.clairvoyant(scenario, limit=calls - 1)


class _Readable(_Tally):
    """A sum that can also be read."""

    def read(self) -> int:
        return self.total


def _read_and_added(**agents: scheduler.Agent) -> scheduler.Scenario:
    """A sum that must never be 1, with agents that add to it and read it."""
    return scheduler.Scenario(
        name="tally", make_world=_Readable, agents=agents,
        invariants=(_never(1, every_step=True),),
        snapshot=lambda tally: tally.total)


async def _adds_one(ctx: scheduler.AgentContext) -> None:
    await ctx.call("add", amount=1)


@pytest.mark.parametrize("lookup", [True, False], ids=["looked-up", "all"])
def test_clairvoyant_search_tries_every_schedule_under_a_table(lookup):
    async def strays_at_0(ctx: scheduler.AgentContext) -> None:
        if await ctx.call("read") == 0:
            await ctx.call("no_such_tool")

    async def loops_at_0(ctx: scheduler.AgentContext) -> None:
        while await ctx.call("read") == 0:
            pass

    # With the adder first the sum is 1 and the run has failed. With the
    # reader first it reads 0, and its run ends in a call that the world
    # has no tool for, or does not end. A search that stopped at the first
    # failing schedule would give 1 under the names that put the failing
    # schedule first, and under no other names. Every schedule is tried,
    # so what ends a run ends this search as it ends the others, whatever
    # the agents are called.
    for agents in (
            {"a": _adds_one, "b": strays_at_0},
            {"a": strays_at_0, "b": _adds_one}):
        scenario = _read_and_added(**agents)
        with pytest.raises(scheduler.UnknownToolError):
            list(explore.exhaust(scenario))
        for value in _VALUES:
            with pytest.raises(scheduler.UnknownToolError):
                value(scenario, lookup=lookup)
    for agents in (
            {"a": _adds_one, "b": loops_at_0},
            {"a": loops_at_0, "b": _adds_one}):
        for value in _VALUES:
            with pytest.raises(solve.TooLongError):
                value(_read_and_added(**agents), limit=30, lookup=lookup)


_VALUES = [
    solve.uniform,
    lambda scenario, **how: solve.adaptive(scenario, **how).value,
    solve.clairvoyant,
]
_VALUE_IDS = ["uniform", "adaptive", "clairvoyant"]


def _runs_made(scenario: scheduler.Scenario, value: Any, **how: Any) -> int:
    """Returns how many runs a value of a scenario takes to work out."""
    made = []

    def make_world() -> Any:
        made.append(None)
        return scenario.make_world()

    value(dataclasses.replace(scenario, make_world=make_world), **how)
    return len(made)


def test_looking_runs_up_gives_the_values_of_the_counter_in_fewer_runs():
    scenario = _counting()
    assert solve.uniform(scenario, lookup=False) == Fraction(5, 36)
    assert solve.adaptive(scenario, lookup=False).value == Fraction(1, 2)
    assert solve.clairvoyant(scenario, lookup=False) == Fraction(5, 8)
    assert solve.adaptive(scenario, lookup=False) == solve.adaptive(scenario)
    # Without looking up, a run is made for every choice and every outcome
    # on every path. Where two paths come to one configuration, the runs
    # from there on are made once.
    assert _runs_made(scenario, solve.uniform, lookup=False) == 51
    assert _runs_made(scenario, solve.uniform) == 41
    # The clairvoyant search starts again under each table, and under a
    # table it looks runs up as well.
    assert _runs_made(scenario, solve.clairvoyant) < _runs_made(
        scenario, solve.clairvoyant, lookup=False)


@pytest.mark.parametrize("lookup", [True, False], ids=["looked-up", "all"])
@pytest.mark.parametrize(
    "scenario,calls",
    [
        (oversell.naive(stock=1, buyers=2), 6),
        (oversell.naive(stock=2, buyers=2), 6),
        (lost_update.naive(editors=2), 4),
        (lost_update.cas(editors=2), 6),
        (write_skew.cas(doctors=3), 6),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_clairvoyant_search_of_one_table_makes_the_runs_of_the_uniform_one(
    scenario, calls, lookup
):
    # Nothing is drawn, so there is one table, and under it the search
    # makes one run for each choice, whether a schedule has failed or not:
    # the runs that the uniform value takes, looked up or not.
    assert _runs_made(
        scenario, solve.clairvoyant, limit=calls, lookup=lookup
    ) == _runs_made(scenario, solve.uniform, limit=calls, lookup=lookup)


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
@pytest.mark.parametrize(
    "scenario,calls",
    [
        (oversell.naive(stock=1, buyers=3), 9),
        (oversell.naive(stock=2, buyers=2), 6),
        (oversell.atomic(stock=1, buyers=3), 3),
        (lost_update.naive(editors=3), 6),
        (lost_update.cas(editors=3), 12),
        (write_skew.naive(doctors=3), 6),
        (write_skew.cas(doctors=3), 6),
        (write_skew.atomic(doctors=3), 3),
    ],
    ids=lambda given: getattr(given, "name", str(given)))
def test_value_of_an_example_is_the_same_with_and_without_looking_up(
    scenario, calls, value
):
    looked_up = value(scenario, limit=calls)
    assert looked_up == value(scenario, limit=calls, lookup=False)
    assert _runs_made(scenario, value, limit=calls) <= _runs_made(
        scenario, value, limit=calls, lookup=False)


def test_adaptive_value_comes_with_the_same_failing_run_either_way():
    for scenario in (
            oversell.naive(stock=1, buyers=3), lost_update.naive(editors=3),
            write_skew.naive(doctors=3)):
        assert solve.adaptive(scenario) == solve.adaptive(
            scenario, lookup=False)


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
def test_run_that_broke_an_invariant_is_not_looked_up_as_one_that_did_not(
    value
):
    async def maybe_there_and_back(ctx: scheduler.AgentContext) -> None:
        there = 1 if await ctx.draw(_EVENLY) else 0
        await ctx.call("add", amount=there)
        await ctx.call("add", amount=-there)

    # With a done, the sum is 0 and b has done nothing, whichever way a
    # went: one state and one history. But one of the two runs broke what
    # must hold after every step, and it has failed whatever b does.
    scenario = scheduler.Scenario(
        name="tally", make_world=_Tally,
        agents={"a": maybe_there_and_back, "b": _caller(1)},
        invariants=(_never(1, every_step=True),),
        snapshot=lambda tally: tally.total)
    assert value(scenario, lookup=False) == Fraction(1, 2)
    assert value(scenario) == Fraction(1, 2)


class _Slot:
    """Keeps one value, and whether it was ever seen to be True itself."""

    def __init__(self) -> None:
        self.value: Any = None
        self.flagged = False

    def put(self, value: Any) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def flag(self) -> None:
        self.flagged = True


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
def test_state_with_1_is_not_looked_up_as_the_state_with_true(value):
    async def putter(ctx: scheduler.AgentContext) -> None:
        await ctx.call("put", value=await ctx.draw(((1, 1), (True, 1))))

    async def getter(ctx: scheduler.AgentContext) -> None:
        if await ctx.call("get") is True:
            await ctx.call("flag")

    # Python takes 1 and True for equal, and so the two states after the
    # put. The getter does not: it goes on otherwise after each. A state
    # is looked up as a trace file would write it, which keeps the two
    # apart.
    scenario = scheduler.Scenario(
        name="slot", make_world=_Slot, agents={"a": putter, "b": getter},
        invariants=(scheduler.Invariant(
            "never-flagged",
            lambda slot: "flagged" if slot.flagged else None,
            every_step=False),),
        snapshot=lambda slot: (slot.value, slot.flagged))
    without = value(scenario, lookup=False)
    assert without == value(scenario)
    # The slot holds True one time in two, and the getter sees it if the
    # put comes first: always for a scheduler that wants it so, and one
    # time in two for a uniform one.
    assert solve.uniform(scenario) == Fraction(1, 4)
    assert solve.adaptive(scenario).value == Fraction(1, 2)
    assert solve.clairvoyant(scenario) == Fraction(1, 2)


def test_snapshot_that_leaves_something_out_gives_another_value_unseen():
    async def one_way_or_another(ctx: scheduler.AgentContext) -> None:
        first = 1 if await ctx.draw(_EVENLY) else 2
        await ctx.call("add", amount=first)
        await ctx.call("add", amount=2 - first)

    def never_odd(tally: _Tally) -> str | None:
        return "the sum was odd on the way" if tally.was_odd else None

    # With a done the sum is 2 either way, and this snapshot says no more.
    # The check at the end goes by whether the sum was ever odd, which it
    # was in one run of two. Looked up by a state that leaves that out,
    # the two runs are one, and the value is not the scenario's. Nothing
    # tells: this is the limit that the solver's docstring names, and the
    # mode with no lookup is the check.
    scenario = scheduler.Scenario(
        name="tally", make_world=_Tally,
        agents={"a": one_way_or_another, "b": _caller(1)},
        invariants=(scheduler.Invariant(
            "never-odd", never_odd, every_step=False),),
        snapshot=lambda tally: tally.total)
    assert solve.uniform(scenario, lookup=False) == Fraction(1, 2)
    assert solve.uniform(scenario) != Fraction(1, 2)
    whole = dataclasses.replace(
        scenario, snapshot=lambda tally: (tally.total, tally.was_odd))
    assert solve.uniform(whole) == Fraction(1, 2)


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
def test_scenario_with_no_snapshot_is_solved_only_without_looking_up(value):
    scenario = dataclasses.replace(_counting(), snapshot=None)
    with pytest.raises(
            scheduler.MissingSnapshotError,
            match="^the scenario 'counting' has no snapshot to give the "
                  "state of its world$"):
        value(scenario)
    assert value(scenario, lookup=False) == value(_counting())


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
def test_state_that_no_trace_file_writes_is_not_looked_up_by(value):
    scenario = dataclasses.replace(
        _counting(),
        snapshot=lambda counter: frozenset({counter.value}))
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        value(scenario)
    assert str(caught.value) == (
        "the solver cannot look the runs of 'counting' up by the state its "
        "snapshot gives: a call does not carry frozenset({0}), of type "
        "frozenset. Give a state of values a call carries, or solve "
        "with lookup=False")
    assert value(scenario, lookup=False) == value(_counting())
