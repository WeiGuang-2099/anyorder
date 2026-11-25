"""Checks local histories and configurations on small worlds.

A counter and a log of the tests' own come first, with what a history and
a configuration hold written out. The examples of the package follow, with
a count worked out by hand and a check over every run of each: two runs at
one configuration go on alike.
"""

import dataclasses
from collections.abc import Hashable, Sequence
from typing import Any

import pytest

from anyorder import configuration
from anyorder import explore
from anyorder import scheduler
from anyorder.scenarios import lost_update
from anyorder.scenarios import oversell
from anyorder.scenarios import write_skew

_NO_ARGS = ("dict", ())


class _Counter:
    """A counter and a list of notes."""

    def __init__(self) -> None:
        self.value = 0
        self.notes: list[str] = []

    def inc(self) -> int:
        self.value += 1
        return self.value

    def note(self, text: str) -> None:
        self.notes.append(text)


def _counted(counter: _Counter) -> tuple[int, tuple[str, ...]]:
    return counter.value, tuple(counter.notes)


async def _adder(ctx: scheduler.AgentContext) -> None:
    await ctx.call("inc")


def _counter(**agents: scheduler.Agent) -> scheduler.Scenario:
    return scheduler.Scenario(
        name="counter", make_world=_Counter, agents=agents, invariants=(),
        snapshot=_counted)


class _Following:
    """A policy and a sampler that follow lists and keep what they stood at.

    Attributes:
        stood: The configuration at each choice.
        histories: The local history of the agent that drew, at each draw.
        moment: The moment of the run, which goes on showing the run.
    """

    def __init__(self, schedule: Sequence[str], *given: Any) -> None:
        self._schedule = iter(schedule)
        self._given = iter(given)
        self.stood: list[configuration.Configuration] = []
        self.histories: list[tuple[str, Hashable]] = []
        self.moment: scheduler.Moment | None = None

    def choose(self, moment: scheduler.Moment) -> str:
        self.moment = moment
        self.stood.append(configuration.at(moment))
        return next(self._schedule)

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        self.moment = moment
        self.histories.append(
            (agent, configuration.local_history(moment, agent)))
        return next(self._given)


def _stood(
    scenario: scheduler.Scenario, schedule: Sequence[str]
) -> list[configuration.Configuration]:
    """Returns the configuration at each choice of a run."""
    following = _Following(schedule)
    scheduler.run(scenario, following)
    return following.stood


async def _busy(ctx: scheduler.AgentContext) -> None:
    """Draws, notes what it drew, draws twice and adds."""
    first = await ctx.draw("start")
    await ctx.call("note", text=f"{ctx.name} drew {first}")
    await ctx.draw(("mid", 1))
    await ctx.draw(["more"])
    await ctx.call("inc")


def test_local_history_holds_an_agents_calls_and_draws_as_they_came():
    following = _Following("abab", "x", 2.0, True)
    scheduler.run(_counter(a=_busy, b=_adder_twice), following, following)
    # Looked at once the run is over, the moment shows all of it. The
    # values are held by their keys: a float and a boolean say what they
    # are, and a tuple is not the list of its parts.
    assert configuration.local_history(following.moment, "a") == (
        ("draw", "start", "x"),
        ("call", "note", ("dict", (("text", "a drew x"),)), None),
        ("draw", ("tuple", ("mid", 1)), ("float", (2.0).hex())),
        ("draw", ("list", ("more",)), ("bool", True)),
        ("call", "inc", _NO_ARGS, 2),
    )
    # Its own and nobody else's: b added before and after it.
    assert configuration.local_history(following.moment, "b") == (
        ("call", "inc", _NO_ARGS, 1),
        ("call", "inc", _NO_ARGS, 3),
    )
    assert configuration.local_history(following.moment, "nobody") == ()


async def _adder_twice(ctx: scheduler.AgentContext) -> None:
    await ctx.call("inc")
    await ctx.call("inc")


def test_local_history_at_a_draw_is_without_the_draw_that_is_being_made():
    following = _Following("aab", "x", "y", "z")
    scheduler.run(_counter(a=_busy, b=_adder), following, following)
    note =("call", "note", ("dict", (("text", "a drew x"),)), None)
    assert following.histories == [
        ("a", ()),
        ("a", (("draw", "start", "x"), note)),
        ("a", (
            ("draw", "start", "x"), note,
            ("draw", ("tuple", ("mid", 1)), "y"))),
    ]


def test_local_history_tells_what_python_takes_for_equal_apart():
    async def drawer(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("coin")
        await ctx.call("inc")

    def history(given: Any) -> Hashable:
        following = _Following("a", given)
        result = scheduler.run(_counter(a=drawer), following, following)
        return result, configuration.local_history(following.moment, "a")

    one, of_one = history(1)
    true, of_true = history(True)
    # An agent that is given True can go on otherwise than one that is
    # given 1, as one that asks whether it is True does. The runs are
    # equal as Python compares, and the histories are not.
    assert one == true
    assert of_one != of_true
    assert of_one[0] == ("draw", "coin", 1)
    assert of_true[0] == ("draw", "coin", ("bool", True))


def test_configuration_is_the_state_and_the_history_of_each_that_waits():
    assert _stood(_counter(b=_adder_twice, c=_adder), "bcb") == [
        configuration.Configuration((0, ()), (("b", ()), ("c", ()))),
        configuration.Configuration(
            (1, ()), (("b", (("call", "inc", _NO_ARGS, 1),)), ("c", ()))),
        # c has finished, and is no part of what comes.
        configuration.Configuration(
            (2, ()), (("b", (("call", "inc", _NO_ARGS, 1),)),)),
    ]


def test_runs_that_come_to_one_configuration_by_two_ways_stand_at_one():
    async def noter(ctx: scheduler.AgentContext) -> None:
        await ctx.call("note", text="late")

    async def signer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("note", text=ctx.name)

    scenario = _counter(a=noter, b=_adder, c=_adder)
    b_first = _stood(scenario, "bca")
    c_first = _stood(scenario, "cba")
    # After one inc it matters which agent made it: the other still
    # waits. After both, a waits alone and has done nothing, and the
    # counter stands at 2 either way.
    assert b_first[1] != c_first[1]
    assert b_first[2] == c_first[2] == configuration.Configuration(
        (2, ()), (("a", ()),))
    assert hash(b_first[2]) == hash(c_first[2])
    assert {b_first[2]: "seen"}[c_first[2]] == "seen"
    # Notes by b and by c leave another state for each order, so the two
    # ways do not come together.
    notes = _counter(a=noter, b=signer, c=signer)
    assert _stood(notes, "bca")[2] != _stood(notes, "cba")[2]


def test_configuration_does_not_say_what_an_agent_that_has_finished_did():
    async def quiet(ctx: scheduler.AgentContext) -> None:
        if await ctx.draw("coin"):
            await ctx.call("note", text="loud")

    following = _Following("b", False)
    scheduler.run(_counter(a=quiet, b=_adder), following, following)
    # a drew and finished without a call. What it drew left nothing in
    # the world and a has no part in what comes, so the run stands where
    # a run without a would.
    assert following.stood == _stood(_counter(b=_adder), "b")


def test_configuration_does_not_say_that_an_invariant_broke_on_the_way():
    class Lamp:
        def __init__(self) -> None:
            self.lit = False

        def switch(self) -> None:
            self.lit = not self.lit

    async def switcher(ctx: scheduler.AgentContext) -> None:
        await ctx.call("switch")

    scenario = scheduler.Scenario(
        name="lamp", make_world=Lamp,
        agents={"a": switcher, "b": switcher, "c": switcher},
        invariants=(
            scheduler.Invariant(
                "dark", lambda lamp: "lit" if lamp.lit else None,
                every_step=True),
        ),
        snapshot=lambda lamp: lamp.lit)
    following = _Following("abc")
    result = scheduler.run(scenario, following)
    # The lamp was lit after the first switch and is dark again when c is
    # chosen. The configuration is that of a dark lamp, and the run has
    # failed: whoever looks runs up by where they stand asks the moment
    # for the violation as well.
    assert following.stood[2] == configuration.Configuration(
        False, (("c", ()),))
    assert result.violation == scheduler.Violation("dark", "lit", 1)


def test_scenario_with_no_snapshot_stands_at_no_configuration():
    scenario = dataclasses.replace(_counter(a=_adder), snapshot=None)
    with pytest.raises(
            scheduler.MissingSnapshotError, match="^the scenario 'counter'"):
        scheduler.run(scenario, _Following("a"))
    # A local history is the agent's alone and needs no state.
    following = _Following((), "x")
    scheduler.run(
        dataclasses.replace(_counter(a=_only_draws), snapshot=None),
        following, following)
    assert following.histories == [("a", ())]


async def _only_draws(ctx: scheduler.AgentContext) -> None:
    await ctx.draw("coin")


def _configurations(
    scenario: scheduler.Scenario,
) -> tuple[int, dict[Hashable, Any]]:
    """Runs every schedule and returns what follows each configuration.

    Returns:
        The number of schedules, and for each configuration that a run
        stood at with the rest of its schedule, the steps that followed.
        Two runs at one configuration under one rest of a schedule must
        have made the same steps from there: that is asked here.
    """
    followed: dict[Hashable, Any] = {}
    runs = list(explore.exhaust(scenario))
    for run in runs:
        stood = _stood(scenario, run.schedule)
        assert len(stood) == len(run.steps)
        for index, where in enumerate(stood):
            rest = (where, run.schedule[index:])
            assert followed.setdefault(rest, run.steps[index:]) == (
                run.steps[index:])
    return len(runs), followed


def test_two_naive_buyers_stand_at_17_configurations():
    # While it waits a buyer has done nothing (N), has read 1 (R) or has
    # read 1 and written (W). It has finished once it placed its order
    # (D) or read 0 (Z), which it does only after the other wrote. With
    # at least one buyer waiting, and the two named in either order:
    # N N and R R and W W, 3. N R, N W, R W, 3 * 2 = 6. D beside N, R or
    # W, 3 * 2 = 6. Z beside W, 2: beside D nobody waits. 3 + 6 + 6 + 2 =
    # 17. The state follows from these: the stock is 0 once one wrote,
    # and the orders are those of the buyers at D.
    schedules, followed = _configurations(oversell.naive(stock=1, buyers=2))
    assert schedules == 16
    assert len({where for where, _ in followed}) == 17


@pytest.mark.parametrize(
    "scenario,schedules",
    [
        (oversell.naive(stock=1, buyers=3), 702),
        (oversell.naive(stock=2, buyers=2), 20),
        (oversell.atomic(stock=1, buyers=3), 6),
        (lost_update.naive(editors=3), 90),
        (lost_update.cas(editors=3), 498),
        (write_skew.naive(doctors=3), 90),
        (write_skew.cas(doctors=3), 90),
        (write_skew.atomic(doctors=3), 6),
    ],
    ids=lambda value: getattr(value, "name", None))
def test_runs_at_one_configuration_go_on_alike(scenario, schedules):
    # The counts are those the tests of each example derive by hand.
    # Every run of every schedule is held against every other that stood
    # where it stood: far fewer configurations than places in runs.
    found, followed = _configurations(scenario)
    assert found == schedules
    places = sum(len(run.steps) for run in explore.exhaust(scenario))
    stood_at = {where for where, _ in followed}
    assert len(stood_at) < places
