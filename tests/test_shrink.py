"""Checks shrinking against every failing schedule of the oversell."""

import collections
import dataclasses
import itertools

import pytest

from anyorder import explore
from anyorder import scheduler
from anyorder import shrink
from anyorder.scenarios import oversell

# One buyer reads, the other runs to its end, the first finishes.
_SMALLEST = {
    ("buyer-1", "buyer-2", "buyer-2", "buyer-2", "buyer-1", "buyer-1"),
    ("buyer-2", "buyer-1", "buyer-1", "buyer-1", "buyer-2", "buyer-2"),
}
_THIRD_BUYER_LAST = (
    "buyer-1", "buyer-2", "buyer-2", "buyer-2", "buyer-1", "buyer-1",
    "buyer-3")


class _Roll:
    """A world that keeps the names called out, in order."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def call_out(self, name: str) -> None:
        self.names.append(name)


def _roll_call(
    invariant: scheduler.Invariant, calls: str = "aab"
) -> scheduler.Scenario:
    """A scenario whose agents call out their names.

    Args:
        invariant: What must hold of the roll.
        calls: One letter for each call an agent makes: with "aab", a calls
            out twice and b once.
    """
    async def caller(ctx: scheduler.AgentContext) -> None:
        for _ in range(calls.count(ctx.name)):
            await ctx.call("call_out", name=ctx.name)

    return scheduler.Scenario(
        name="roll-call",
        make_world=_Roll,
        agents={name: caller for name in sorted(set(calls))},
        invariants=(invariant,),
    )


class _Hall:
    """A world that tells each speaker who spoke before it."""

    def __init__(self) -> None:
        self.spoke: list[str] = []

    def speak(self, name: str) -> tuple[str, ...]:
        before = tuple(self.spoke)
        self.spoke.append(name)
        return before


def _hall(speaker: scheduler.Agent, apart: str) -> scheduler.Scenario:
    """Speakers a, b and c, where the two in apart must not both speak."""
    def both_spoke(hall: _Hall) -> str | None:
        if set(apart) <= set(hall.spoke):
            return f"{apart[0]} and {apart[1]} have both spoken"
        return None

    return scheduler.Scenario(
        name="hall",
        make_world=_Hall,
        agents={"a": speaker, "b": speaker, "c": speaker},
        invariants=(
            scheduler.Invariant("apart", both_spoke, every_step=True),
        ),
    )


class _Count:
    """A world that keeps one number."""

    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> None:
        self.value = 0

    def add(self, amount: int) -> None:
        self.value += amount


def _never_three() -> scheduler.Scenario:
    """A scenario where b adds 2 and c resets, then adds 1 twice."""
    async def b(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", amount=2)

    async def c(ctx: scheduler.AgentContext) -> None:
        await ctx.call("reset")
        await ctx.call("add", amount=1)
        await ctx.call("add", amount=1)

    def never_three(count: _Count) -> str | None:
        return "the count is 3" if count.value == 3 else None

    return scheduler.Scenario(
        name="never-three",
        make_world=_Count,
        agents={"b": b, "c": c},
        invariants=(
            scheduler.Invariant("never-three", never_three, every_step=True),
        ),
    )


def _checked_at_the_end(scenario: scheduler.Scenario) -> scheduler.Scenario:
    """The same scenario with every invariant asked only at the end."""
    return dataclasses.replace(scenario, invariants=tuple(
        dataclasses.replace(invariant, every_step=False)
        for invariant in scenario.invariants))


def _failing(scenario: scheduler.Scenario) -> list[scheduler.RunResult]:
    return [
        run for run in explore.exhaust(scenario) if run.violation is not None]


@pytest.mark.parametrize("schedule,switches", [
    ("", 0), ("a", 0), ("aa", 0), ("ab", 1), ("abba", 2), ("abab", 3)])
def test_context_switches_counts_changes_of_agent(schedule, switches):
    assert shrink.context_switches(schedule) == switches


def test_size_stops_at_the_violation():
    failure = explore.replay(oversell.naive(buyers=3), _THIRD_BUYER_LAST)
    # The second order is the sixth call, so buyer-3 and the switch to it
    # are not counted.
    assert failure.violation.step == 6
    assert shrink.size(failure) == (2, 2)
    assert shrink.context_switches(failure.schedule) == 3


def test_size_counts_the_whole_run_when_the_invariant_is_asked_at_the_end():
    scenario = _checked_at_the_end(oversell.naive(buyers=3))
    failure = explore.replay(scenario, _THIRD_BUYER_LAST)
    assert failure.violation.step == 7
    assert shrink.size(failure) == (3, 3)


def test_size_of_a_world_that_starts_broken_is_nothing():
    scenario = _roll_call(scheduler.Invariant(
        "never", lambda roll: "never holds", every_step=True))
    failure = explore.replay(scenario, ("a", "b", "a"))
    assert failure.violation.step == 0
    assert shrink.size(failure) == (0, 0)
    assert shrink.shrink(scenario, failure).result == failure


def test_size_refuses_a_run_that_passed():
    passing = explore.replay(oversell.atomic(), ("buyer-1", "buyer-2"))
    with pytest.raises(ValueError, match="broke no invariant"):
        shrink.size(passing)


def test_demo_failure_shrinks_from_4_context_switches_to_2():
    # The search's failure is buyer-1 buyer-2 buyer-1 buyer-2 buyer-2
    # buyer-1: the agent changes after steps 1, 2, 3 and 5, so 4 switches.
    # Two agents that both act make at least 1 switch, and a run with 1 is
    # serial, which sells one unit and passes. A run with 2 is a, then b,
    # then a. To fail, both must read 1, so a's first block is its read
    # alone: with its write in it b would read 0. b has no other block, so
    # its block is all three of its calls, and a's last block is its write
    # and its order. That is one schedule for each choice of a, so 2 is the
    # least and _SMALLEST holds every failure that reaches it.
    scenario = oversell.naive()
    failure = explore.explore(scenario, range(100))
    assert failure is not None
    assert shrink.size(failure.result) == (4, 2)
    shrunk = shrink.shrink(scenario, failure.result).result
    assert shrink.size(shrunk) == (2, 2)
    assert shrunk.schedule in _SMALLEST
    assert shrunk.violation.invariant == "no-oversell"
    assert explore.replay(scenario, shrunk.schedule) == shrunk


def test_every_two_buyer_failure_shrinks_to_2_context_switches():
    # A failing run starts with both reads, say a then b, and its four
    # other calls (a write and an order each) go in 6 ways:
    #   a b a a b b: 3 switches      a b b a a b: 3 switches
    #   a b a b a b: 5 switches      a b b a b a: 4 switches
    #   a b a b b a: 4 switches      a b b b a a: 2 switches
    # With 2 choices of a that is 2 runs with 2 switches, 4 with 3, 4 with
    # 4 and 2 with 5.
    scenario = oversell.naive()
    failing = _failing(scenario)
    assert collections.Counter(map(shrink.size, failing)) == {
        (2, 2): 2, (3, 2): 4, (4, 2): 4, (5, 2): 2}
    assert {
        run.schedule for run in failing if shrink.size(run) == (2, 2)
    } == _SMALLEST
    for run in failing:
        shrunk = shrink.shrink(scenario, run).result
        assert shrunk.schedule in _SMALLEST
        assert shrunk in failing


def test_every_three_buyer_failure_shrinks_to_two_buyers():
    # Two orders break the invariant and an order is a buyer's third call,
    # so two buyers make 6 calls before it breaks. As with two buyers, 2
    # switches are the least for them. The third buyer is not needed: put
    # off to the end it reads 0, after the violation.
    scenario = oversell.naive(buyers=3)
    failing = _failing(scenario)
    assert min(map(shrink.size, failing)) == (2, 2)
    for run in failing:
        shrunk = shrink.shrink(scenario, run).result
        assert shrink.size(shrunk) == (2, 2)
        assert shrunk.violation.invariant == "no-oversell"
        assert shrunk.violation.step == 6
        assert [step.result for step in shrunk.steps[6:]] == [0]
        assert shrunk in failing


def test_three_buyers_of_two_units_all_stay():
    # Three orders break the invariant, so every buyer makes its three
    # calls before it breaks. Three agents that all act make at least 2
    # switches, and a run with 2 is serial: it sells the two units and
    # passes. 3 are enough: a reads 2, b buys, a buys on its stale read, and
    # c reads 1 and buys.
    scenario = oversell.naive(stock=2, buyers=3)
    failing = _failing(scenario)
    assert min(map(shrink.size, failing)) == (3, 3)
    for run in failing:
        shrunk = shrink.shrink(scenario, run).result
        assert shrink.size(shrunk) == (3, 3)
        assert shrunk.violation.step == 9
        assert shrunk in failing


def test_invariant_asked_at_the_end_keeps_every_buyer():
    # The violation is seen once the run is over, so every step counts and
    # every buyer makes at least its read. Three agents make at least 2
    # switches, and a run with 2 is serial and passes. 3 are enough: a
    # reads, b buys, a buys and c reads 0.
    scenario = _checked_at_the_end(oversell.naive(buyers=3))
    failing = _failing(scenario)
    assert min(map(shrink.size, failing)) == (3, 3)
    for run in failing:
        shrunk = shrink.shrink(scenario, run).result
        assert shrink.size(shrunk) == (3, 3)
        assert shrunk in failing


def test_run_that_breaks_another_invariant_does_not_count():
    def buyer_1_orders_before_buyer_2(
        inventory: oversell.Inventory,
    ) -> str | None:
        if inventory.orders[:1] == ["buyer-2"]:
            return "buyer-2 ordered before buyer-1"
        return None

    scenario = oversell.naive()
    scenario = dataclasses.replace(scenario, invariants=(
        scheduler.Invariant(
            "buyer-1-orders-before-buyer-2", buyer_1_orders_before_buyer_2,
            every_step=True),
        *scenario.invariants))
    # buyer-1 orders at step 5 and buyer-2 at step 6, so only the oversell
    # is broken.
    failure = explore.replay(scenario, (
        "buyer-1", "buyer-2", "buyer-1", "buyer-2", "buyer-1", "buyer-2"))
    assert failure.violation.invariant == "no-oversell"
    shrunk = shrink.shrink(scenario, failure).result
    # In the other schedule of _SMALLEST buyer-2 orders at step 4, before
    # buyer-1, so the other invariant breaks before the oversell. In the
    # serial run that starts with buyer-2, which has 1 switch, buyer-2
    # orders at step 3 and buyer-1 reads 0, so only the other invariant
    # breaks.
    assert shrunk.schedule == (
        "buyer-2", "buyer-1", "buyer-1", "buyer-1", "buyer-2", "buyer-2")
    assert shrunk.violation.invariant == "no-oversell"


def test_as_many_switches_and_fewer_agents_is_smaller():
    def no_rally(roll: _Roll) -> str | None:
        """Fails on three calls in a row with a change of agent each time."""
        last = roll.names[-3:]
        if len(last) == 3 and last[0] != last[1] != last[2]:
            return f"rally: {' '.join(last)}"
        return None

    scenario = _roll_call(
        scheduler.Invariant("no-rally", no_rally, every_step=True), "aabc")
    failure = explore.replay(scenario, ("a", "b", "c", "a"))
    # a b c is a rally at step 3: 2 switches, 3 agents. A rally is 2
    # switches, so no failure has fewer. Putting a off gives b c a, still 3
    # agents. Putting b off gives a c a, a rally of 2 agents, which is kept.
    assert shrink.size(failure) == (2, 3)
    shrunk = shrink.shrink(scenario, failure).result
    assert shrunk.schedule == ("a", "c", "a", "b")
    assert shrink.size(shrunk) == (2, 2)


def test_fewer_switches_is_smaller_even_with_more_agents():
    async def restocker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("write_stock", value=1)

    scenario = oversell.naive()
    scenario = dataclasses.replace(
        scenario, agents={**scenario.agents, "restocker": restocker})
    failure = explore.replay(scenario, (
        "buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2",
        "restocker"))
    # Both buyers read 1 and the second order is step 6: 3 switches and 2
    # agents.
    assert shrink.size(failure) == (3, 2)
    shrunk = shrink.shrink(scenario, failure)
    # The first candidate puts buyer-1 off: buyer-2 buys, the restocker
    # writes 1 and buyer-1 buys. That serial run has 2 switches and 3
    # agents. Fewer switches come first, so it is kept. From it, putting
    # buyer-1 off changes nothing and is not run, with buyer-2 or the
    # restocker put off the second buyer reads 0, and no agent has two
    # blocks: 1 + 2 candidates.
    assert shrunk == shrink.Shrunk(explore.replay(scenario, (
        "buyer-2", "buyer-2", "buyer-2", "restocker",
        "buyer-1", "buyer-1", "buyer-1")), 3)
    assert shrink.size(shrunk.result) == (2, 3)
    # The search does not go back: the failure of the two buyers alone has
    # as many switches and fewer agents, and no move leads to it from here.
    assert shrink.size(explore.replay(scenario, (
        "buyer-1", "buyer-2", "buyer-2", "buyer-2", "buyer-1", "buyer-1",
        "restocker"))) == (2, 2)


def test_shrinking_can_end_above_the_smallest_failure():
    # b has one call, so there are 4 schedules, and the count goes:
    #   b c c c: 2, 0, 1, 2 and passes
    #   c b c c: 0, 2, 3 and fails at step 3 after 2 switches
    #   c c b c: 0, 1, 3 and fails at step 3 after 1 switch
    #   c c c b: 0, 1, 2, 4 and passes
    # From c b c c, putting b off or growing c's first block by its last
    # gives c c c b, and putting c off or growing its last block by its
    # first gives b c c c. Both pass, and each is run once: 2 candidates.
    # The smaller failure moves one step of c's last block and leaves the
    # other, which no move does.
    scenario = _never_three()
    failing = _failing(scenario)
    assert {run.schedule: shrink.size(run) for run in failing} == {
        ("c", "b", "c", "c"): (2, 2), ("c", "c", "b", "c"): (1, 2)}
    for run in failing:
        assert shrink.shrink(scenario, run) == shrink.Shrunk(run, 2)


def test_agents_left_when_the_order_runs_out_run_in_name_order():
    async def speaker(ctx: scheduler.AgentContext) -> None:
        """c speaks twice. a and b speak again unless c spoke before."""
        before = await ctx.call("speak", name=ctx.name)
        if ctx.name == "c" or "c" not in before:
            await ctx.call("speak", name=ctx.name)

    scenario = _hall(speaker, apart="ab")
    failure = explore.replay(scenario, ("c", "c", "a", "b"))
    # c spoke before a and b, so each of them speaks once and the invariant
    # breaks at step 4: 2 switches, 3 agents.
    assert shrink.size(failure) == (2, 3)
    shrunk = shrink.shrink(scenario, failure).result
    # With a or b put off the run still starts with c, and all three speak
    # before the invariant breaks. With c put off the order is a b c c:
    # neither a nor b has heard c, so the invariant breaks at step 2 and
    # each of them has a second call that the order does not hold. They
    # make those after c's two, in name order. 1 switch and 2 agents is the
    # least: a and b must both have spoken.
    assert shrunk.schedule == ("a", "b", "c", "c", "a", "b")
    assert shrink.size(shrunk) == (1, 2)


def test_entry_of_an_agent_that_has_finished_is_skipped():
    async def speaker(ctx: scheduler.AgentContext) -> None:
        """b speaks again if a spoke before it."""
        before = await ctx.call("speak", name=ctx.name)
        if ctx.name == "b" and "a" in before:
            await ctx.call("speak", name=ctx.name)

    scenario = _hall(speaker, apart="bc")
    failure = explore.replay(scenario, ("a", "b", "b", "c"))
    # b heard a, so it speaks twice, and the invariant breaks at step 4:
    # 2 switches, 3 agents.
    assert shrink.size(failure) == (2, 3)
    shrunk = shrink.shrink(scenario, failure).result
    # The first candidate puts a off: b b c a. b has not heard a and is
    # done after one call, so its second entry is skipped and c runs next,
    # not a, the first waiting agent. The invariant breaks at step 2, with
    # 1 switch and 2 agents, which is the least: b and c must both have
    # spoken.
    assert shrunk.schedule == ("b", "c", "a")
    assert shrink.size(shrunk) == (1, 2)


def test_agents_are_put_off_first_in_name_order_and_replays_not_counted():
    scenario = _roll_call(scheduler.Invariant(
        "never", lambda roll: "never holds", every_step=False), "abb")
    failure = explore.replay(scenario, ("b", "a", "b"))
    # Asked at the end, so all 3 steps count: b a b has 2 switches. The
    # first candidate puts a off: b b a has 1 and is kept. Putting b off
    # first, or joining b's blocks first, would give a b b. Two agents make
    # at least 1 switch, so nothing after it is smaller. From b b a,
    # putting a off changes nothing and is not run, putting b off gives
    # a b b, and no agent has two blocks. That is 1 + 1 candidates, and the
    # replay of b b a is not one of them.
    assert shrink.shrink(scenario, failure) == shrink.Shrunk(
        explore.replay(scenario, ("b", "b", "a")), 2)


@pytest.mark.parametrize("schedule", sorted(_SMALLEST))
def test_smallest_failure_is_returned_as_it_is(schedule):
    scenario = oversell.naive()
    failure = explore.replay(scenario, schedule)
    # Each of the two agents put off gives a serial order. Joining the first
    # and the last block, either way, gives those two again, and an order
    # is run once: 2 candidates.
    assert shrink.shrink(scenario, failure) == shrink.Shrunk(failure, 2)


def test_shrink_refuses_a_run_that_passed():
    scenario = oversell.atomic()
    passing = explore.replay(scenario, ("buyer-1", "buyer-2"))
    with pytest.raises(ValueError, match="broke no invariant"):
        shrink.shrink(scenario, passing)


def test_shrink_refuses_a_failure_that_does_not_replay():
    tickets = itertools.count()
    scenario = _roll_call(scheduler.Invariant(
        "ticket", lambda roll: f"ticket {next(tickets)}", every_step=False))
    failure = explore.replay(scenario, ("a", "b", "a"))
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="schedule 'a b a' of 'roll-call'"):
        shrink.shrink(scenario, failure)


def test_shrink_refuses_a_failure_whose_steps_do_not_replay():
    tickets = itertools.count()

    class Ticketed(_Roll):
        def call_out(self, name: str) -> int:
            super().call_out(name)
            return next(tickets)

    scenario = dataclasses.replace(
        _roll_call(scheduler.Invariant(
            "never", lambda roll: "never holds", every_step=False)),
        make_world=Ticketed)
    failure = explore.replay(scenario, ("a", "b", "a"))
    # The violation is the same in every run. Only the results differ.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="schedule 'a b a' of 'roll-call'"):
        shrink.shrink(scenario, failure)


def test_shrink_refuses_a_failure_of_another_scenario():
    failure = explore.replay(oversell.naive(), sorted(_SMALLEST)[0])
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="buyer-3 can still run"):
        shrink.shrink(oversell.naive(buyers=3), failure)


def test_shrink_refuses_a_smaller_run_that_does_not_replay():
    tickets = itertools.count()

    def ticket(roll: _Roll) -> str:
        """Fails with the same message only in the order a, b, a."""
        if roll.names == ["a", "b", "a"]:
            return "no ticket"
        return f"ticket {next(tickets)}"

    scenario = _roll_call(
        scheduler.Invariant("ticket", ticket, every_step=False))
    failure = explore.replay(scenario, ("a", "b", "a"))
    # The given run has 2 switches and replays. The two other orders have
    # 1, so either is smaller, and neither gives the same message twice.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="replayed to a different trace"):
        shrink.shrink(scenario, failure)


def test_run_that_drew_is_not_shrunk():
    built = []

    class Roll(_Roll):
        def __init__(self) -> None:
            super().__init__()
            built.append(self)

    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.draw((("go", 1),))
        await ctx.call("call_out", name=ctx.name)

    scenario = scheduler.Scenario(
        name="drawn-roll",
        make_world=Roll,
        agents={"a": caller, "b": caller},
        invariants=(
            scheduler.Invariant(
                "silence", lambda roll: "somebody called", every_step=False),
        ),
    )
    failure = scheduler.run(
        scenario, scheduler.RandomPolicy(0), scheduler.SeededSampler(0))
    assert failure.violation is not None and len(failure.draws) == 2
    built.clear()
    # Its draws would have to be held fixed while its schedule changes,
    # and nothing says yet which draw an agent is given once it has read
    # something else.
    with pytest.raises(
            ValueError,
            match="^the run made draws, and a run that drew is not shrunk "
                  "yet$"):
        shrink.shrink(scenario, failure)
    # Refused before anything was run.
    assert not built
