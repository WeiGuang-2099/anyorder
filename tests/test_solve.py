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


def _adds(*amounts: int) -> scheduler.Agent:
    """An agent that adds each amount in one call, in order."""
    async def agent(ctx: scheduler.AgentContext) -> None:
        for amount in amounts:
            await ctx.call("add", amount=amount)

    return agent


# Every value as a function of the scenario. The serial order is the name
# order, and the adaptive value comes without its failing run.
_EVERY_VALUE = [
    solve.uniform,
    solve.round_robin,
    lambda scenario, **how: solve.serial(
        scenario, sorted(scenario.agents), **how),
    lambda scenario, **how: solve.adaptive(scenario, **how).value,
    solve.clairvoyant,
]
_EVERY_ID = ["uniform", "round-robin", "serial", "adaptive", "clairvoyant"]


def test_serial_order_is_the_order_that_is_given():
    # a adds 1 and b adds 2. With a first the sum is 1 after the first
    # call. With b first it is 2 and then 3, and never 1.
    scenario = _tallying(
        _never(1, every_step=True), a=_adds(1), b=_adds(2))
    assert solve.serial(scenario, ["a", "b"]) == 1
    assert solve.serial(scenario, ["b", "a"]) == 0
    assert solve.serial(scenario, ("b", "a")) == 0
    assert solve.serial(scenario, "ba") == 0


def test_serial_agent_makes_all_its_calls_before_the_next_makes_one():
    # a adds 1 twice and b adds 2. With a first the sums are 1, 2 and 4,
    # with b first 2, 3 and 4, and with one call each in turn, from a, 1,
    # 3 and 4.
    scenario = _tallying(
        _never(3, every_step=True), a=_adds(1, 1), b=_adds(2))
    assert solve.serial(scenario, ["a", "b"]) == 0
    assert solve.serial(scenario, ["b", "a"]) == 1
    assert solve.round_robin(scenario) == 1


def test_serial_order_of_three_is_the_order_that_is_given():
    # a adds 1, b adds 2 and c adds 4. The sum is 3 on the way only where
    # a and b are the first two, in either order: 1, 3 or 2, 3. With c
    # among the first two it goes 1, 5 or 2, 6 or from 4 on.
    scenario = _tallying(
        _never(3, every_step=True), a=_adds(1), b=_adds(2), c=_adds(4))
    assert [
        solve.serial(scenario, order)
        for order in itertools.permutations("abc")] == [1, 0, 1, 0, 0, 0]


def test_name_order_is_the_order_python_puts_names_in():
    # "B" comes before "a" as Python puts strings in order. From B the
    # sum is 2 after the first call. From a it is 1 and then 3.
    scenario = _tallying(
        _never(2, every_step=True), **{"a": _adds(1), "B": _adds(2)})
    assert solve.round_robin(scenario) == 1
    assert solve.serial(scenario, ["B", "a"]) == 1
    assert solve.serial(scenario, ["a", "B"]) == 0


def test_serial_order_that_is_a_set_of_any_kind_is_refused():
    class Names(set):
        """A set that is more than a set."""

    class Frozen(frozenset):
        """A frozenset that is more than a frozenset."""

    one = _tallying(a=_adds(1))
    two = _tallying(a=_adds(1), b=_adds(2))
    # What is refused is a set, and not what it holds: also a set of one
    # name or of none, which could be walked in one way only.
    for scenario, order in [
            (two, Names({"a", "b"})), (two, Frozen({"a", "b"})),
            (one, {"a"}), (one, frozenset({"a"})),
            (_tallying(), set()), (_tallying(), frozenset())]:
        with pytest.raises(ValueError, match="and a set has no first: "):
            solve.serial(scenario, order)


def test_serial_order_can_be_the_keys_of_a_dict():
    scenario = _tallying(
        _never(1, every_step=True), a=_adds(1), b=_adds(2))
    # They come in the order the dict was filled in.
    assert solve.serial(scenario, {"a": 0, "b": 0}.keys()) == 1
    assert solve.serial(scenario, {"b": 0, "a": 0}.keys()) == 0


def test_serial_order_of_nobody_is_told_from_a_scenario_of_nobody():
    with pytest.raises(
            ValueError,
            match="^a serial order of 'tally' names each of a, b once$"):
        solve.serial(_tallying(a=_adds(1), b=_adds(2)), ())
    with pytest.raises(
            ValueError,
            match="^a serial order of 'tally' names nobody: the scenario "
                  "has no agents$"):
        solve.serial(_tallying(), [1])


def test_serial_and_round_robin_meet_only_what_ends_a_run_of_their_own():
    async def strays_after_b(ctx: scheduler.AgentContext) -> None:
        if await ctx.call("add", amount=0) == 1:
            await ctx.call("no_such_tool")

    scenario = _tallying(a=strays_after_b, b=_adds(1))
    # With b first, a calls a tool the world lacks. An exhaustive search
    # and the values that try every choice come to that run, and end.
    with pytest.raises(scheduler.UnknownToolError):
        list(explore.exhaust(scenario))
    for value in (solve.uniform, solve.adaptive, solve.clairvoyant):
        with pytest.raises(scheduler.UnknownToolError):
            value(scenario)
    with pytest.raises(scheduler.UnknownToolError):
        solve.serial(scenario, ["b", "a"])
    # One serial order and round robin follow one schedule each. These
    # two never come to that run, and have a value.
    assert solve.serial(scenario, ["a", "b"]) == 0
    assert solve.round_robin(scenario) == 0


def test_serial_and_round_robin_go_by_the_names_and_not_by_the_listing():
    # The scenario lists b first. The turn starts at a all the same, and
    # the sum is 1 after the first call. From b it would be 2, then 3.
    scenario = _tallying(
        _never(1, every_step=True), b=_adds(2), a=_adds(1))
    assert list(scenario.agents) == ["b", "a"]
    assert solve.round_robin(scenario) == 1
    assert solve.serial(scenario, ["a", "b"]) == 1
    assert solve.serial(scenario, ["b", "a"]) == 0
    with pytest.raises(
            ValueError,
            match="^a serial order of 'tally' names each of a, b once$"):
        solve.serial(scenario, ["b"])


def test_round_robin_passes_over_an_agent_that_has_finished():
    def turns(total: int) -> scheduler.Scenario:
        return _tallying(
            _never(total, every_step=True),
            a=_adds(1, 1, 1), b=_adds(10), c=_adds(100, 100, 100))

    # The calls come as a b c a c a c: after c it is a's turn, and after a
    # the turn passes over b, which has finished, to c. The sums on the
    # way are 1, 11, 111, 112, 212, 213 and 313.
    assert solve.round_robin(turns(213)) == 1
    # And not as a b c a c c a, which a turn by the number of calls made
    # would give: the sum is never 312.
    assert solve.round_robin(turns(312)) == 0
    # The first name can be the last to finish: the turn comes back to it
    # past everybody else.
    assert solve.round_robin(_tallying(a=_adds(0, 0, 0), b=_adds(0))) == 0


@pytest.mark.parametrize("value", _EVERY_VALUE, ids=_EVERY_ID)
def test_run_is_followed_to_its_end_after_it_broke_an_invariant(value):
    async def strays(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=1)
        await ctx.call("no_such_tool")

    # The sum is 1 after the first call, and the run has failed. It is
    # followed to its end all the same, and its end is a call the world
    # has no tool for: there is no value, as an exhaustive search has no
    # run.
    scenario = _tallying(_never(1, every_step=True), a=strays)
    with pytest.raises(scheduler.UnknownToolError):
        value(scenario)
    with pytest.raises(scheduler.UnknownToolError):
        list(explore.exhaust(scenario))


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


def _drifting(
    runs: int, before: scheduler.Agent, after: scheduler.Agent
) -> scheduler.Scenario:
    """A sum whose one agent acts otherwise once some worlds were made."""
    made = []

    class Numbered(_Tally):
        """A sum whose worlds are counted as they are made."""

        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    async def agent(ctx: scheduler.AgentContext) -> None:
        await (before if len(made) <= runs else after)(ctx)

    return scheduler.Scenario(
        name="drifting", make_world=Numbered, agents={"a": agent},
        invariants=(), snapshot=lambda tally: tally.total)


def _script(*acts: Any) -> scheduler.Agent:
    """An agent that adds 0 for each None and draws from anything else."""
    async def agent(ctx: scheduler.AgentContext) -> None:
        for act in acts:
            if act is None:
                await ctx.call("add", amount=0)
            else:
                await ctx.draw(act)

    return agent


_LEFT = "a run of 'drifting' left the start it was given to follow: "
_DRAW_FOR_A_HELD_CALL = (
    3, (None, None, None), (None, _EVENLY, None),
    _LEFT + "it draws where it made a call before")
_ANOTHER_REQUEST = (
    2, (_EVENLY, None), (_RARELY, None),
    "draw 1 is from ((True, 1), (False, 1)), but agent 'a' asks to draw "
    "from ((True, 1), (False, 3))")
_EARLY_END = (
    2, (None, None), (None,),
    "a run of 'drifting' ended at the start it was given to follow, where "
    "it went on before")
_WAYS_TO_LEAVE = [
    # The start drew after its first call, and the run that follows it
    # makes its second call there.
    (3, (None, _EVENLY, None), (None, None),
     _LEFT + "it waits for a choice where it drew before"),
    # The same one run earlier: the draw is the decision to make.
    (2, (None, _EVENLY, None), (None, None),
     _LEFT + "it waits for a choice where it drew before"),
    (3, (None, _EVENLY, None), (_EVENLY, None, None),
     _LEFT + "draw 1 was made after 0 steps, not after 1"),
    _DRAW_FOR_A_HELD_CALL,
    (2, (None, None), (None, _EVENLY, None),
     _LEFT + "it draws where it waited for a choice before"),
    _ANOTHER_REQUEST,
    _EARLY_END,
]
_WAY_IDS = [
    "choice-for-a-held-draw", "choice-for-the-draw", "draw-at-another-step",
    "draw-for-a-held-call", "draw-for-the-choice", "another-request",
    "early-end"]


@pytest.mark.parametrize("runs,before,after,how", _WAYS_TO_LEAVE, ids=_WAY_IDS)
@pytest.mark.parametrize("value", _EVERY_VALUE[:4], ids=_EVERY_ID[:4])
def test_run_that_leaves_its_start_is_refused_where_it_leaves_it(
    value, runs, before, after, how
):
    # The agent is the same for the first runs and another from then on,
    # so a run that follows the start of an earlier one leaves it: with a
    # choice or a draw where the start had the other, a draw after another
    # step, a draw from something else, or an end.
    scenario = _drifting(runs, _script(*before), _script(*after))
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        value(scenario)
    assert str(caught.value) == how


@pytest.mark.parametrize(
    "runs,before,after,how",
    [_DRAW_FOR_A_HELD_CALL, _ANOTHER_REQUEST, _EARLY_END],
    ids=["draw-for-a-held-call", "another-request", "early-end"])
def test_clairvoyant_search_refuses_a_run_that_leaves_its_start(
    runs, before, after, how
):
    # Under a table a draw is no decision, so this search makes other
    # runs than the others and holds a start against fewer of them. These
    # three it meets as they do.
    scenario = _drifting(runs, _script(*before), _script(*after))
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        solve.clairvoyant(scenario)
    assert str(caught.value) == how


@pytest.mark.parametrize(
    "runs,before,after",
    [way[:3] for way in _WAYS_TO_LEAVE
     if way not in (_DRAW_FOR_A_HELD_CALL, _ANOTHER_REQUEST, _EARLY_END)],
    ids=[name for name in _WAY_IDS
         if name not in (
             "draw-for-a-held-call", "another-request", "early-end")])
def test_clairvoyant_search_gives_a_value_where_the_others_refuse(
    runs, before, after
):
    # Each of these leaves a start by a draw, and the other values refuse
    # it. Under a table a draw is no decision and the search starts again
    # at each place the table lacks, so this one holds no run against the
    # one before across the draw, and gives a value. It probes whether a
    # scenario is the same twice, as they do, with other runs.
    for value in _EVERY_VALUE[:4]:
        with pytest.raises(scheduler.ReplayDivergenceError):
            value(_drifting(runs, _script(*before), _script(*after)))
    for lookup in (True, False):
        assert solve.clairvoyant(
            _drifting(runs, _script(*before), _script(*after)),
            lookup=lookup) == 0


@pytest.mark.parametrize("value", _EVERY_VALUE, ids=_EVERY_ID)
def test_solver_can_give_a_value_where_an_exhaustive_search_refuses(value):
    def build() -> scheduler.Scenario:
        # Three calls in the first run that is made, one in every later.
        return _drifting(1, _script(None, None, None), _script(None))

    # An exhaustive search makes its first run to the end and replays it.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the run ended after 1 steps but the schedule has 3$"):
        list(explore.exhaust(build()))
    # The solver's first run stops at the first choice, and every run it
    # holds against another is of the later kind. Neither search proves
    # that a scenario is the same twice.
    assert value(build()) == 0


def test_clairvoyant_search_does_not_end_on_a_new_place_in_every_run():
    made = []

    class Numbered(_Tally):
        """A sum whose worlds are counted as they are made."""

        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    async def restless(ctx: scheduler.AgentContext) -> None:
        await ctx.draw(((len(made), 1), (0, 1)))
        await ctx.call("add", amount=0)

    scenario = scheduler.Scenario(
        name="shifting", make_world=Numbered, agents={"a": restless},
        invariants=(), snapshot=lambda tally: tally.total)
    # The other values hold the request of a run against that of the run
    # before it.
    for value in _EVERY_VALUE[:4]:
        made.clear()
        with pytest.raises(scheduler.ReplayDivergenceError):
            value(scenario)
    made.clear()
    # The clairvoyant search starts again with one more place in its
    # table, without end. What stops it is Python, and no refusal of the
    # package's own: a limit that is written down, and held here so that
    # a change to it is seen.
    with pytest.raises(RecursionError):
        solve.clairvoyant(scenario)


@pytest.mark.parametrize(
    "value", [solve.uniform, solve.adaptive], ids=["uniform", "adaptive"])
def test_outcome_is_not_handed_to_a_draw_from_no_written_distribution(value):
    # The first run draws from a coin and stops there. The second is given
    # an outcome of the coin, and asks to draw from a word.
    scenario = _drifting(1, _script(_EVENLY, None), _script("coin", None))
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot draw from 'coin': "):
        value(scenario)


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


@pytest.mark.parametrize("value", _EVERY_VALUE, ids=_EVERY_ID)
def test_every_value_follows_a_run_for_the_limit_it_is_given(value):
    scenario = _tallying(a=_caller(5))
    assert value(scenario, limit=5) == 0
    with pytest.raises(
            solve.TooLongError,
            match="^a run of 'tally' went past 4 choices and draws, "):
        value(scenario, limit=4)


def test_limit_with_runs_looked_up_goes_by_the_order_of_the_search():
    def drawer(request: Any) -> scheduler.Agent:
        async def one_or_three_draws(ctx: scheduler.AgentContext) -> None:
            if not await ctx.draw(request):
                await ctx.draw(request)
                await ctx.draw(request)

        return one_or_three_draws

    # a draws once or three times and calls nothing, b makes three calls:
    # a run takes 4 or 6 choices and draws, and every long run comes to
    # the configuration the short ones leave, b with nothing done.
    short_first = _tallying(a=drawer(((True, 1), (False, 1))), b=_caller(3))
    long_first = _tallying(a=drawer(((False, 1), (True, 1))), b=_caller(3))
    for scenario in (short_first, long_first):
        assert solve.uniform(scenario, limit=6, lookup=False) == 0
        with pytest.raises(solve.TooLongError):
            solve.uniform(scenario, limit=5, lookup=False)
    # Looked up, a long run is not followed past what a short one worked
    # out, if the short one came first. So the limit holds the runs that
    # are made, and a scenario can have a value at a limit below its
    # longest run.
    assert solve.uniform(short_first, limit=4) == 0
    assert solve.adaptive(short_first, limit=4).value == 0
    with pytest.raises(solve.TooLongError):
        solve.uniform(short_first, limit=3)
    assert solve.uniform(long_first, limit=6) == 0
    with pytest.raises(solve.TooLongError):
        solve.uniform(long_first, limit=5)
    # The clairvoyant value keeps what it looked up for one table only.
    with pytest.raises(solve.TooLongError):
        solve.clairvoyant(short_first, limit=5)


def test_limit_with_runs_looked_up_goes_by_the_names_at_a_choice():
    async def reader(ctx: scheduler.AgentContext) -> None:
        # Three draws, which are no scheduling points, where it reads 1.
        if await ctx.call("add", amount=0) == 1:
            for _ in range(3):
                await ctx.draw(((0, 1),))

    async def toggler(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=1)
        await ctx.call("add", amount=-1)
        for _ in range(3):
            await ctx.call("add", amount=0)

    def named(**agents: scheduler.Agent) -> scheduler.Scenario:
        return scheduler.Scenario(
            name="tally", make_world=_Tally, agents=agents, invariants=(),
            snapshot=lambda tally: tally.total)

    # The longest run is the toggler's first call, the reader's call and
    # its three draws, and the toggler's other four calls: 9.
    for scenario in (named(a=reader, b=toggler), named(a=toggler, b=reader)):
        assert solve.uniform(scenario, limit=9, lookup=False) == 0
        with pytest.raises(solve.TooLongError):
            solve.uniform(scenario, limit=8, lookup=False)
    # Looked up, the waiting agents are tried in name order. With the
    # reader first, "the reader done, the toggler after its first call"
    # is worked out along the short way, and the long way there, 5
    # choices and draws, is followed no further: 6 with the choice it
    # stops at.
    assert solve.uniform(named(a=reader, b=toggler), limit=6) == 0
    # With the toggler first, the long way is the first to come there and
    # is followed one call further, to where the toggler has made two.
    with pytest.raises(solve.TooLongError):
        solve.uniform(named(a=toggler, b=reader), limit=6)
    assert solve.uniform(named(a=toggler, b=reader), limit=7) == 0


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
    # And of no narrower kind: a caller that catches a divergence is told
    # that the scenario is not the same twice, which a run that was cut
    # does not say.
    assert solve.TooLongError.__bases__ == (scheduler.SchedulerError,)


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


@pytest.mark.parametrize("value", _EVERY_VALUE, ids=_EVERY_ID)
def test_search_is_one_call_deeper_for_each_draw_of_a_run(value):
    depths = []

    class Deep(_Tally):
        """A sum that keeps how deep Python's stack is at each call."""

        def add(self, amount: int) -> int:
            depths.append(len(inspect.stack(0)))
            return super().add(amount)

    def drawer(draws: int) -> scheduler.Agent:
        async def agent(ctx: scheduler.AgentContext) -> None:
            for _ in range(draws):
                await ctx.draw(_EVENLY)
            await ctx.call("add", amount=0)

        return agent

    deepest = []
    for draws in range(4):
        depths.clear()
        value(scheduler.Scenario(
            name="deep", make_world=Deep, agents={"a": drawer(draws)},
            invariants=(), snapshot=lambda tally: tally.total))
        deepest.append(max(depths))
    # The one call of a run is made one decision deeper for each draw
    # before it, or under a table one place deeper.
    assert [deeper - before
            for before, deeper in zip(deepest, deepest[1:])] == [1, 1, 1]


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
    # By hand, each from the first read on, which some agent X makes.
    # * Two buyers of one unit: X's write waits beside the other's read.
    #   With 1/2 the write comes first, and the other reads 0 and buys
    #   nothing. With 1/2 the other reads 1 as well, and both buy.
    # * Three buyers: the write waits beside two reads. With 1/3 it comes
    #   first and both others read 0. With 2/3 a second buyer reads 1.
    # * Two units: nobody oversells, whatever is lost.
    # * Two editors: with 1/2 X writes before the other reads. With 1/2
    #   both have read the same lines, and the second write loses a line.
    # * Three editors: with 2/3 another reads before X writes, and a line
    #   is lost. With 1/3 X writes first, one of the two left reads, and
    #   with 1/2 the last reads before that one writes. Nothing is lost
    #   with 1/3 * 1/2, so 5/6.
    # * Two doctors: with 1/2 X leaves before the other reads, and the
    #   other stays. With 1/2 both read a full rota and both leave.
    # * Three doctors: the first two to read leave whatever comes, and the
    #   last, Z, leaves unless both have left before it reads. With 1/3 X
    #   leaves before another reads: Y reads next, and Z reads before Y
    #   leaves with 1/2. With 2/3 Y reads next: Z reads at once with 1/3,
    #   and with 2/3 one of the two leaves and Z reads before the other
    #   does with 1/2, which is 2/3. So 1/3 * 1/2 + 2/3 * 2/3 = 11/18.
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
    # One call each in turn fails no control. It fails the fix that
    # fails: every doctor reads before any of them leaves.
    assert solve.round_robin(scenario, limit=calls) == (
        0 if value == 0 else 1)
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


def test_failing_run_is_replayed_from_its_steps_and_not_from_its_names():
    made = []

    class Numbered(_Tally):
        """A sum whose worlds are counted as they are made."""

        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    async def agent(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=len(made))

    scenario = scheduler.Scenario(
        name="numbered", make_world=Numbered, agents={"a": agent},
        invariants=(scheduler.Invariant(
            "never", lambda tally: "it ended", every_step=False),),
        snapshot=lambda tally: tally.total)
    # The one call of a run adds the number of worlds made so far. The
    # search makes two runs, and the replay of the failing one is the
    # third: its call is held against the step before it is made.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 1 is add\\(amount=2\\), but agent 'a' waits to "
                  "call add\\(amount=3\\)$"):
        solve.adaptive(scenario)


def test_failing_run_takes_the_first_outcome_that_fails_as_written():
    request = ((2, 1), (1, 1), (3, 1))

    async def roller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=await ctx.draw(request))

    def never_odd(tally: _Tally) -> str | None:
        return "the sum was odd" if tally.was_odd else None

    scenario = _tallying(
        scheduler.Invariant("never-odd", never_odd, every_step=False),
        a=roller)
    solved = solve.adaptive(scenario)
    assert solved.value == Fraction(2, 3)
    # Two outcomes fail, and the run that comes with the value drew the
    # one that is written first of the two.
    assert solved.failure.draws == (scheduler.Drawn("a", request, 1, 0),)


def test_adaptive_result_cannot_be_changed():
    solved = solve.adaptive(_counting())
    with pytest.raises(dataclasses.FrozenInstanceError):
        solved.value = Fraction(0)


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
    # on every path: one for each point of the search, the ends among
    # them. With k incs still waiting, the points from one on number
    # * 1, 2 and 5 for k of 0, 1 and 2 once a has finished: 1 + k times
    #   the number for k - 1;
    # * 2, 5 and 16 where a waits to set: 1, a's set and then the number
    #   above, and for each inc the number here for k - 1;
    # * 4, 8 and 22 at a's draw: 1, the number where it sets and the
    #   number where it has finished;
    # * 5, 14 and 51 where a waits to read: 1, the number at its draw, and
    #   for each inc the number here for k - 1.
    assert _runs_made(scenario, solve.uniform, lookup=False) == 51
    # Where two paths come to one configuration, the runs from there on
    # are made once. Seven points are come to a second time. Six have one
    # run below them: a waits to set with both incs made, after a read of
    # 0 and after a read of 1, and b or c waits alone with the counter at
    # 11 and set, or at 1 and not set. One has four below it: a waits to
    # read with both incs made. That saves 6 + 4 runs.
    assert _runs_made(scenario, solve.uniform) == 41
    # The clairvoyant search starts again under each table, and under a
    # table it looks runs up as well.
    assert _runs_made(scenario, solve.clairvoyant) < _runs_made(
        scenario, solve.clairvoyant, lookup=False)
    # The adaptive search makes the runs of the uniform one, and one
    # more: the replay of the failing run it returns.
    assert _runs_made(scenario, solve.adaptive, lookup=False) == 52
    assert _runs_made(scenario, solve.adaptive) == 42
    # With no failing run there is nothing to replay.
    whole = oversell.naive(stock=2, buyers=2)
    assert _runs_made(whole, solve.adaptive) == _runs_made(
        whole, solve.uniform)


class _Slate(_Tally):
    """A sum that can be wiped."""

    def wipe(self) -> None:
        self.total = 0


def test_run_is_looked_up_where_it_stops_at_a_draw():
    async def wiper(ctx: scheduler.AgentContext) -> None:
        await ctx.call("wipe")
        await ctx.draw(_EVENLY)

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=2 if await ctx.draw(_EVENLY) else 4)

    scenario = scheduler.Scenario(
        name="slate", make_world=_Slate, agents={"a": wiper, "b": adder},
        invariants=(), snapshot=lambda slate: slate.total)
    # b draws before its first call, so the first run stops at that draw.
    # For each of its 2 outcomes: 1 run to the choice between a and b.
    # With a first, 1 run to a's draw and for each outcome 1 run to b
    # alone and 1 to the end: 5. With b first, 1 run to a alone, 1 to a's
    # draw and 1 for each outcome: 4. That is 1 + 2 * (1 + 5 + 4) runs.
    assert solve.uniform(scenario) == 0
    assert _runs_made(scenario, solve.uniform, lookup=False) == 21
    # Looked up, b alone at 0 is come to after either outcome of a's draw
    # and its 1 run is saved, for each outcome of b's draw: 2. And a's
    # wipe after b's add leaves 0 whatever b added: a stands at its draw
    # alone with the same things seen and done, and the 2 runs below that
    # draw are made once. A run is looked up where it stops at a draw, too.
    assert _runs_made(scenario, solve.uniform) == 17


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
    # The refusal stands alone, with no other error shown behind it.
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert value(scenario, lookup=False) == value(_counting())


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
def test_state_that_cannot_be_hashed_is_not_looked_up_by(value):
    scenario = dataclasses.replace(
        _tallying(a=_adds(1)), snapshot=lambda tally: [tally.total])
    with pytest.raises(
            TypeError,
            match="^the snapshot of 'tally' gave \\[0\\], which cannot be "
                  "hashed"):
        value(scenario)
    assert value(scenario, lookup=False) == 0


def test_serial_and_round_robin_look_nothing_up():
    # So neither asks for the state of a world: a scenario with no
    # snapshot has these two values, and so has one whose snapshot gives
    # what no run is looked up by.
    for snapshot in (
            None, lambda counter: [counter.value],
            lambda counter: frozenset({counter.value})):
        scenario = dataclasses.replace(_counting(), snapshot=snapshot)
        for order in itertools.permutations("abc"):
            assert solve.serial(scenario, order) == 0
        assert solve.round_robin(scenario) == Fraction(1, 4)


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
def test_request_that_is_refused_is_refused_before_the_state_is_asked(value):
    async def guesser(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("coin")

    # The first thing this run asks for is a draw from no written
    # distribution. That is said, and not that the scenario has no
    # snapshot to look the run up by.
    scenario = dataclasses.replace(_tallying(a=guesser), snapshot=None)
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot draw from 'coin': "):
        value(scenario)
