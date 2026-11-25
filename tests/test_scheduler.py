"""Checks the scheduler's ordering, refusals, tool errors, invariants and draws.

The samplers' tests come last: the seeded one, then the one that replays.
"""

import asyncio
import collections
import dataclasses
import enum
import functools
import inspect
import itertools
import types
from collections.abc import Coroutine, Iterator, Mapping
from typing import Any

import pytest

from anyorder import scheduler
from anyorder import values


class _Log:
    """A world that keeps what agents record, in the order they did."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def record(self, entry: str) -> None:
        self.entries.append(entry)

    def size(self) -> int:
        return len(self.entries)

    def _hidden(self) -> None:
        raise AssertionError("a private method is not a tool")


async def _writer(ctx: scheduler.AgentContext) -> None:
    await ctx.call("record", entry=f"{ctx.name}:1")
    await ctx.call("record", entry=f"{ctx.name}:2")


def _scenario(**agents: scheduler.Agent) -> scheduler.Scenario:
    return scheduler.Scenario(
        name="log",
        make_world=_Log,
        agents=agents or {"a": _writer, "b": _writer},
        invariants=(),
    )


def _entries(result: scheduler.RunResult) -> list[str]:
    return [step.args["entry"] for step in result.steps]


async def _tidier(ctx: scheduler.AgentContext) -> None:
    """Makes a call while it is being closed, which Python refuses."""
    try:
        await ctx.call("record", entry=f"{ctx.name}:1")
    finally:
        await ctx.call("record", entry=f"{ctx.name}:tidy")


def _kept(
    agent: scheduler.Agent, coroutines: dict[str, Coroutine[Any, Any, None]]
) -> scheduler.Agent:
    """Wraps an agent so that a test can look at its coroutine afterwards."""
    def start(ctx: scheduler.AgentContext) -> Coroutine[Any, Any, None]:
        coroutines[ctx.name] = agent(ctx)
        return coroutines[ctx.name]

    return start


class _Till:
    """A world whose tool raises when asked for more than it holds."""

    def __init__(self) -> None:
        self.cash = 5

    def take(self, amount: int) -> int:
        if amount > self.cash:
            raise ValueError(f"only {self.cash} left")
        self.cash -= amount
        return self.cash


async def _taker(ctx: scheduler.AgentContext) -> None:
    """Asks for 4 and, when the till raises, for 1."""
    left = await ctx.call("take", amount=4)
    if isinstance(left, scheduler.Raised):
        await ctx.call("take", amount=1)


def _till(**agents: scheduler.Agent) -> scheduler.Scenario:
    return scheduler.Scenario(
        name="till",
        make_world=_Till,
        agents=agents or {"a": _taker, "b": _taker},
        invariants=(),
    )


class _Lamp:
    """A lamp that each switch turns on or off."""

    def __init__(self, lit: bool) -> None:
        self.lit = lit
        self.switches = 0

    def switch(self) -> None:
        self.lit = not self.lit
        self.switches += 1


def _dark(lamp: _Lamp) -> str | None:
    return "the lamp is lit" if lamp.lit else None


def _switched_once(lamp: _Lamp) -> str | None:
    if lamp.switches < 2:
        return None
    return f"switched {lamp.switches} times"


def _broken(lamp: _Lamp) -> str | None:
    """A check that raises whenever it is asked."""
    raise KeyError("broken check")


def _tripped(lamp: _Lamp) -> str | None:
    """A check that raises once the lamp has been switched."""
    if lamp.switches:
        raise KeyError("broken check")
    return None


_ALWAYS_DARK = scheduler.Invariant("always-dark", _dark, every_step=True)
_DARK_AT_END = scheduler.Invariant("dark-at-end", _dark, every_step=False)


def _lamp(
    switches: int, *invariants: scheduler.Invariant, lit: bool = False
) -> scheduler.Scenario:
    """One agent that switches a lamp a number of times."""
    async def switcher(ctx: scheduler.AgentContext) -> None:
        for _ in range(switches):
            await ctx.call("switch")

    return scheduler.Scenario(
        name="lamp",
        make_world=functools.partial(_Lamp, lit),
        agents={"a": switcher},
        invariants=invariants,
    )


def test_same_seed_gives_same_trace():
    first = scheduler.run(_scenario(), scheduler.RandomPolicy(7))
    second = scheduler.run(_scenario(), scheduler.RandomPolicy(7))
    assert first == second


def test_seeds_reach_every_interleaving():
    # Two agents with two calls each can interleave in 6 ways.
    schedules = {
        scheduler.run(_scenario(), scheduler.RandomPolicy(seed)).schedule
        for seed in range(100)
    }
    assert len(schedules) == 6
    assert all(sorted(schedule) == ["a", "a", "b", "b"]
               for schedule in schedules)


def test_schedule_decides_the_order_of_tool_calls():
    result = scheduler.run(
        _scenario(), scheduler.ReplayPolicy(["b", "a", "a", "b"]))
    assert _entries(result) == ["b:1", "a:1", "a:2", "b:2"]
    assert result.schedule == ("b", "a", "a", "b")


@pytest.mark.parametrize(
    "listed", ["abc", "cab", "cba"], ids=["sorted", "mixed", "reversed"])
def test_agents_are_started_in_name_order_however_they_are_listed(listed):
    started = []

    async def starter(ctx: scheduler.AgentContext) -> None:
        # Before its first call, so this runs when the agent is started.
        started.append(ctx.name)
        await ctx.call("record", entry=ctx.name)

    async def idler(ctx: scheduler.AgentContext) -> None:
        # Makes no call, and is started in its place all the same.
        started.append(ctx.name)

    agents = {name: starter for name in listed}
    scenario = _scenario(**agents, bb=idler)
    assert list(scenario.agents) == [*listed, "bb"]
    scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert started == ["a", "b", "bb", "c"]


def test_agent_receives_the_tool_result():
    async def counter(ctx: scheduler.AgentContext) -> None:
        size = await ctx.call("size")
        await ctx.call("record", entry=f"saw {size}")

    result = scheduler.run(
        _scenario(a=_writer, c=counter),
        scheduler.ReplayPolicy(["a", "c", "a", "c"]))
    assert result.steps[1].result == 1
    assert result.steps[3].args == {"entry": "saw 1"}


def test_replay_refuses_an_agent_that_is_not_waiting():
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 3"):
        scheduler.run(
            _scenario(), scheduler.ReplayPolicy(["a", "a", "a", "b"]))


def test_replay_refuses_a_schedule_that_ends_early():
    with pytest.raises(scheduler.ReplayDivergenceError, match="ends after 2"):
        scheduler.run(_scenario(), scheduler.ReplayPolicy(["a", "b"]))


def test_unknown_tool_is_refused():
    async def lost(ctx: scheduler.AgentContext) -> None:
        await ctx.call("missing")

    with pytest.raises(
            scheduler.UnknownToolError, match="^_Log has no tool 'missing'$"):
        scheduler.run(_scenario(a=lost), scheduler.RandomPolicy(0))


def test_attribute_that_is_no_method_is_not_a_tool():
    async def prying(ctx: scheduler.AgentContext) -> None:
        await ctx.call("entries")

    with pytest.raises(
            scheduler.UnknownToolError, match="^_Log has no tool 'entries'$"):
        scheduler.run(_scenario(a=prying), scheduler.RandomPolicy(0))


def test_private_method_is_not_a_tool():
    async def prying(ctx: scheduler.AgentContext) -> None:
        await ctx.call("_hidden")

    with pytest.raises(scheduler.UnknownToolError, match="'_hidden'"):
        scheduler.run(_scenario(a=prying), scheduler.RandomPolicy(0))


def test_agent_may_only_await_tool_calls():
    async def sleeper(ctx: scheduler.AgentContext) -> None:
        await asyncio.sleep(0)

    with pytest.raises(scheduler.UnsupportedAwaitError, match="'a'"):
        scheduler.run(_scenario(a=sleeper), scheduler.RandomPolicy(0))


def test_refused_call_closes_every_agent():
    async def lost(ctx: scheduler.AgentContext) -> None:
        await ctx.call("missing")

    coroutines = {}
    scenario = _scenario(
        a=_kept(lost, coroutines), b=_kept(_writer, coroutines))
    with pytest.raises(scheduler.UnknownToolError):
        scheduler.run(scenario, scheduler.ReplayPolicy(["a"]))
    # The agent whose call was refused too, not only the one left waiting.
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "b": inspect.CORO_CLOSED}


def test_unsupported_await_closes_its_agent():
    async def sleeper(ctx: scheduler.AgentContext) -> None:
        await asyncio.sleep(0)

    coroutines = {}
    with pytest.raises(scheduler.UnsupportedAwaitError):
        scheduler.run(
            _scenario(a=_kept(sleeper, coroutines)), scheduler.RandomPolicy(0))
    assert inspect.getcoroutinestate(coroutines["a"]) == inspect.CORO_CLOSED


def test_refusal_survives_an_agent_that_raises_while_closing():
    async def lost(ctx: scheduler.AgentContext) -> None:
        await ctx.call("missing")

    async def sore(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.call("record", entry="b:1")
        finally:
            raise OSError("will not close")

    coroutines = {}
    scenario = _scenario(a=lost, b=sore, c=_kept(_writer, coroutines))
    # Any error of the agent's own is dropped, not only Python's complaint
    # about a call made while closing.
    with pytest.raises(scheduler.UnknownToolError, match="'missing'"):
        scheduler.run(scenario, scheduler.ReplayPolicy(["a"]))
    assert inspect.getcoroutinestate(coroutines["c"]) == inspect.CORO_CLOSED


def test_refusal_survives_agents_that_call_a_tool_while_closing():
    async def lost(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.call("missing")
        finally:
            await ctx.call("record", entry="a:tidy")

    coroutines = {}
    scenario = _scenario(a=lost, b=_tidier, c=_kept(_writer, coroutines))
    with pytest.raises(scheduler.UnknownToolError, match="'missing'"):
        scheduler.run(scenario, scheduler.ReplayPolicy(["a"]))
    # The two that would not close did not keep the one after them open.
    assert inspect.getcoroutinestate(coroutines["c"]) == inspect.CORO_CLOSED


def test_divergence_survives_an_agent_that_calls_a_tool_while_closing():
    with pytest.raises(scheduler.ReplayDivergenceError, match="ends after 0"):
        scheduler.run(_scenario(a=_tidier), scheduler.ReplayPolicy([]))


def test_unsupported_await_survives_a_tool_call_while_closing():
    async def sleeper(ctx: scheduler.AgentContext) -> None:
        try:
            await asyncio.sleep(0)
        finally:
            await ctx.call("record", entry="a:tidy")

    with pytest.raises(scheduler.UnsupportedAwaitError, match="'a'"):
        scheduler.run(_scenario(a=sleeper), scheduler.RandomPolicy(0))


def test_tool_that_raises_returns_its_error_to_the_agent():
    result = scheduler.run(_till(), scheduler.ReplayPolicy(["a", "b", "b"]))
    # b is told that the till raised, and goes on to ask for less.
    assert [step.result for step in result.steps] == [
        1, scheduler.Raised("ValueError", "only 1 left"), 0]
    assert result.steps[2].args == {"amount": 1}


def test_tool_error_is_equal_from_run_to_run():
    first = scheduler.run(_till(), scheduler.ReplayPolicy(["a", "b", "b"]))
    second = scheduler.run(_till(), scheduler.ReplayPolicy(["a", "b", "b"]))
    assert first == second


def test_wrong_arguments_are_a_tool_error():
    async def clumsy(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take", sum=4)

    result = scheduler.run(_till(a=clumsy), scheduler.ReplayPolicy(["a"]))
    error = result.steps[0].result
    assert isinstance(error, scheduler.Raised)
    # Only the kind: the text is the interpreter's and changes with it.
    assert error.kind == "TypeError"


@pytest.mark.parametrize(
    "error",
    [
        scheduler.ReplayDivergenceError("from a tool"),
        scheduler.SchedulerError("from a tool"),
        KeyboardInterrupt(),
    ],
    ids=["divergence", "refusal", "interrupt"])
def test_error_that_is_not_the_agents_to_handle_ends_the_run(error):
    class Alarm:
        def pull(self) -> None:
            raise error

    async def puller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("pull")

    scenario = scheduler.Scenario(
        name="alarm",
        make_world=Alarm,
        agents={"a": puller},
        invariants=(),
    )
    with pytest.raises(type(error)):
        scheduler.run(scenario, scheduler.RandomPolicy(0))


def test_invariant_has_to_say_when_it_must_hold():
    with pytest.raises(TypeError, match="every_step"):
        scheduler.Invariant("always-dark", _dark)
    # By keyword, so that the scenario shows which of the two it is.
    with pytest.raises(TypeError, match="positional"):
        scheduler.Invariant("always-dark", _dark, True)


def test_every_step_invariant_sees_a_state_that_is_gone_by_the_end():
    result = scheduler.run(_lamp(2, _ALWAYS_DARK), scheduler.RandomPolicy(0))
    assert result.violation == scheduler.Violation(
        "always-dark", "the lamp is lit", 1)
    # The run goes on to its end after the violation.
    assert len(result.steps) == 2


def test_end_invariant_passes_a_state_that_is_gone_by_the_end():
    result = scheduler.run(_lamp(2, _DARK_AT_END), scheduler.RandomPolicy(0))
    assert result.violation is None


def test_end_invariant_fails_with_the_length_of_the_run():
    # Lit after the first switch too, which an invariant asked only at the
    # end does not see.
    result = scheduler.run(_lamp(3, _DARK_AT_END), scheduler.RandomPolicy(0))
    assert result.violation == scheduler.Violation(
        "dark-at-end", "the lamp is lit", 3)


def test_every_step_invariant_is_checked_on_the_fresh_world():
    result = scheduler.run(
        _lamp(2, _ALWAYS_DARK, lit=True), scheduler.RandomPolicy(0))
    assert result.violation == scheduler.Violation(
        "always-dark", "the lamp is lit", 0)
    assert len(result.steps) == 2


@pytest.mark.parametrize(
    "invariant", [_ALWAYS_DARK, _DARK_AT_END], ids=["every-step", "at-end"])
def test_run_with_no_steps_checks_both_kinds(invariant):
    result = scheduler.run(
        _lamp(0, invariant, lit=True), scheduler.RandomPolicy(0))
    assert result == scheduler.RunResult(
        (), scheduler.Violation(invariant.name, "the lamp is lit", 0))


def test_earlier_violation_beats_earlier_listing():
    once = scheduler.Invariant("once", _switched_once, every_step=True)
    result = scheduler.run(
        _lamp(3, once, _ALWAYS_DARK), scheduler.RandomPolicy(0))
    # Listed second, but the lamp is lit one step before the second switch.
    assert result.violation == scheduler.Violation(
        "always-dark", "the lamp is lit", 1)


def test_violations_at_one_moment_keep_the_order_listed():
    lit = scheduler.Invariant(
        "lit", lambda lamp: "starts lit", every_step=True)
    result = scheduler.run(
        _lamp(1, lit, _ALWAYS_DARK, lit=True), scheduler.RandomPolicy(0))
    assert result.violation == scheduler.Violation("lit", "starts lit", 0)


def test_every_step_invariants_are_asked_before_end_invariants():
    result = scheduler.run(
        _lamp(0, _DARK_AT_END, _ALWAYS_DARK, lit=True),
        scheduler.RandomPolicy(0))
    assert result.violation is not None
    assert result.violation.invariant == "always-dark"


@pytest.mark.parametrize("every_step", [True, False])
def test_nothing_is_asked_after_the_first_violation(every_step):
    # A check can raise on a world that is already broken. Asking it then
    # would lose the violation that came first. The lamp is lit at step 1,
    # where the second check would already raise.
    fragile = scheduler.Invariant("fragile", _tripped, every_step=every_step)
    result = scheduler.run(
        _lamp(2, _ALWAYS_DARK, fragile), scheduler.RandomPolicy(0))
    assert result.violation == scheduler.Violation(
        "always-dark", "the lamp is lit", 1)


def test_every_step_invariant_is_asked_after_a_call_that_raised():
    class Bulb(_Lamp):
        """A lamp whose switch lights it and then raises."""

        def switch(self) -> None:
            super().switch()
            raise RuntimeError("the bulb blew")

    scenario = dataclasses.replace(
        _lamp(1, _ALWAYS_DARK), make_world=functools.partial(Bulb, False))
    result = scheduler.run(scenario, scheduler.RandomPolicy(0))
    # What the tool did before it raised stays done, and is checked.
    assert result.steps[0].result == scheduler.Raised(
        "RuntimeError", "the bulb blew")
    assert result.violation == scheduler.Violation(
        "always-dark", "the lamp is lit", 1)


# _broken raises on the fresh world, _tripped after the first step, when
# the agent that took it is still waiting for its result.
@pytest.mark.parametrize("check", [_broken, _tripped])
@pytest.mark.parametrize("every_step", [True, False])
def test_invariant_that_raises_ends_the_run_with_its_error(check, every_step):
    fragile = scheduler.Invariant("fragile", check, every_step=every_step)
    scenario = _lamp(2, fragile)
    coroutines = {}
    scenario = dataclasses.replace(
        scenario, agents={"a": _kept(scenario.agents["a"], coroutines)})
    # Neither a violation nor a pass: the scenario itself is wrong.
    with pytest.raises(KeyError, match="broken check"):
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert inspect.getcoroutinestate(coroutines["a"]) == inspect.CORO_CLOSED


class _Shelf:
    """A world that hands out its own list and keeps what it is given."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.kept: Any = None
        self.row = ("a", 1)
        self.error = scheduler.Raised("Busy", "later")

    def add(self, item: str) -> list[str]:
        self.items.append(item)
        return self.items

    def count(self, seen: list[str]) -> int:
        return len(seen)

    def keep(self, items: Any) -> None:
        self.kept = items

    def kept_size(self) -> int:
        return len(self.kept)

    def fill(self, into: list[str]) -> None:
        into.append("filled")

    def tag(self, label: str, into: list[str]) -> None:
        into.append(label)

    def read_row(self) -> tuple[str, int]:
        return self.row

    def ask(self) -> scheduler.Raised:
        return self.error

    def mark(self) -> set[int]:
        self.kept = "marked"
        return {1, 2}


def _shelf(
    worlds: list[_Shelf] | None = None, **agents: scheduler.Agent
) -> scheduler.Scenario:
    """A shelf for the agents, kept in worlds for the test to look at."""
    def make_world() -> _Shelf:
        world = _Shelf()
        if worlds is not None:
            worlds.append(world)
        return world

    return scheduler.Scenario(
        name="shelf", make_world=make_world, agents=agents, invariants=())


async def _adder(ctx: scheduler.AgentContext) -> None:
    await ctx.call("add", item=ctx.name)


def _shown(result: scheduler.RunResult) -> list[tuple[Any, ...]]:
    return [(step.tool, step.args, step.result) for step in result.steps]


def test_trace_holds_a_result_as_it_was_when_its_call_returned():
    # The shelf returns its own list, which the second call changes.
    result = scheduler.run(
        _shelf(a=_adder, b=_adder), scheduler.ReplayPolicy(["a", "b"]))
    assert _shown(result) == [
        ("add", {"item": "a"}, ["a"]),
        ("add", {"item": "b"}, ["a", "b"]),
    ]


def test_agent_holds_no_live_part_of_its_world():
    async def watcher(ctx: scheduler.AgentContext) -> None:
        seen = await ctx.call("add", item=ctx.name)
        # No call of its own between: what it holds must not have grown.
        await ctx.call("count", seen=seen)

    result = scheduler.run(
        _shelf(a=watcher, b=_adder), scheduler.ReplayPolicy(["a", "b", "a"]))
    assert _shown(result) == [
        ("add", {"item": "a"}, ["a"]),
        ("add", {"item": "b"}, ["a", "b"]),
        ("count", {"seen": ["a"]}, 1),
    ]


def test_world_holds_no_live_part_of_an_agent():
    async def keeper(ctx: scheduler.AgentContext) -> None:
        items = ["a"]
        await ctx.call("keep", items=items)
        # No call: the agent changes its own list after handing it over.
        items.append("late")
        await ctx.call("kept_size")

    result = scheduler.run(_shelf(a=keeper), scheduler.ReplayPolicy(["a"] * 2))
    assert _shown(result) == [
        ("keep", {"items": ["a"]}, None),
        ("kept_size", {}, 1),
    ]


def test_tool_hands_nothing_back_through_an_argument():
    async def asker(ctx: scheduler.AgentContext) -> None:
        into: list[str] = []
        await ctx.call("fill", into=into)
        await ctx.call("count", seen=into)

    result = scheduler.run(_shelf(a=asker), scheduler.ReplayPolicy(["a"] * 2))
    # The trace shows the argument as the agent gave it, and the agent's
    # own list is as it was: a call is by value.
    assert _shown(result) == [
        ("fill", {"into": []}, None),
        ("count", {"seen": []}, 0),
    ]


def test_tool_gets_its_own_copy_beside_an_argument_that_cannot_change():
    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("tag", label="x", into=[])

    # One argument that cannot change does not make the other the tool's
    # to share with the trace.
    result = scheduler.run(_shelf(a=asker), scheduler.ReplayPolicy(["a"]))
    assert _shown(result) == [("tag", {"label": "x", "into": []}, None)]


def test_agent_that_changes_a_result_changes_nobody_elses():
    async def scribbler(ctx: scheduler.AgentContext) -> None:
        seen = await ctx.call("add", item=ctx.name)
        seen.append("scribble")
        await ctx.call("count", seen=seen)

    worlds: list[_Shelf] = []
    result = scheduler.run(
        _shelf(worlds, a=scribbler, b=_adder),
        scheduler.ReplayPolicy(["a", "a", "b"]))
    assert _shown(result) == [
        ("add", {"item": "a"}, ["a"]),
        ("count", {"seen": ["a", "scribble"]}, 2),
        ("add", {"item": "b"}, ["a", "b"]),
    ]
    assert worlds[0].items == ["a", "b"]


def test_arguments_are_taken_when_the_agent_awaits_the_call():
    shared = ["a"]

    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("keep", items=shared)

    async def meddler(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", item="first")
        # The asker is waiting to make its call, and the list it gave is
        # still in this agent's reach.
        shared.append("meddled")
        await ctx.call("add", item="second")

    worlds: list[_Shelf] = []
    result = scheduler.run(
        _shelf(worlds, a=asker, m=meddler),
        scheduler.ReplayPolicy(["m", "a", "m"]))
    assert _shown(result)[1] == ("keep", {"items": ["a"]}, None)
    assert worlds[0].kept == ["a"]


def test_value_that_cannot_change_is_not_copied():
    received = []

    async def reader(ctx: scheduler.AgentContext) -> None:
        received.append(await ctx.call("read_row"))
        await ctx.call("keep", items=received[0])

    worlds: list[_Shelf] = []
    result = scheduler.run(
        _shelf(worlds, a=reader), scheduler.ReplayPolicy(["a"] * 2))
    row = worlds[0].row
    assert received[0] is row
    assert result.steps[0].result is row
    assert result.steps[1].args["items"] is row
    assert worlds[0].kept is row


def test_tool_error_is_shared_by_nobody():
    received = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        received.append(await ctx.call("ask"))
        received.append(await ctx.call("kept_size"))

    worlds: list[_Shelf] = []
    result = scheduler.run(
        _shelf(worlds, a=asker), scheduler.ReplayPolicy(["a"] * 2))
    # The first error is one the shelf returned and keeps. The second is
    # of a tool that raised: nothing is kept yet, and None has no length.
    assert result.steps[0].result == scheduler.Raised("Busy", "later")
    assert result.steps[1].result.kind == "TypeError"
    for step, mine in zip(result.steps, received):
        assert mine == step.result and mine is not step.result
    assert result.steps[0].result is not worlds[0].error
    assert received[0] is not worlds[0].error


class _Sub(dict):
    """A dict of another type, which says it holds no values."""

    def values(self) -> list[Any]:
        return []


class _Shy(Mapping):
    """A mapping that says it is empty and is not."""

    def __init__(self, held: dict[str, Any]) -> None:
        self._held = held

    def __getitem__(self, key: str) -> Any:
        return self._held[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._held)

    def __len__(self) -> int:
        return 0


@pytest.mark.parametrize(
    "wrap", [types.MappingProxyType, _Sub, _Shy],
    ids=["view", "dict-subclass", "says-it-is-empty"])
def test_call_made_with_another_mapping_is_taken_as_a_dict(wrap):
    given = {"into": ["x"]}

    async def asker(ctx: scheduler.AgentContext) -> None:
        # An agent can await a call it built itself, and keep its parts.
        await scheduler.Effect("fill", wrap(given))
        given["into"].append("late")
        await ctx.call("count", seen=given["into"])

    result = scheduler.run(_shelf(a=asker), scheduler.ReplayPolicy(["a"] * 2))
    assert _shown(result) == [
        ("fill", {"into": ["x"]}, None),
        ("count", {"seen": ["x", "late"]}, 2),
    ]
    assert type(result.steps[0].args) is dict


def test_arguments_of_a_call_built_by_hand_are_taken_out_of_their_dict():
    async def asker(ctx: scheduler.AgentContext) -> None:
        # No value in either dict can change. The dicts can, and they are
        # the agent's to keep.
        empty: dict[str, Any] = {}
        await scheduler.Effect("read_row", empty)
        empty["late"] = 1
        plain = {"item": "x"}
        await scheduler.Effect("add", plain)
        plain["item"] = "late"

    result = scheduler.run(_shelf(a=asker), scheduler.ReplayPolicy(["a"] * 2))
    assert _shown(result) == [
        ("read_row", {}, ("a", 1)),
        ("add", {"item": "x"}, ["x"]),
    ]


def test_arguments_that_cannot_change_are_out_of_reach_while_they_wait():
    empty: dict[str, Any] = {}
    plain = {"item": "x"}

    async def reader(ctx: scheduler.AgentContext) -> None:
        await scheduler.Effect("read_row", empty)

    async def writer(ctx: scheduler.AgentContext) -> None:
        await scheduler.Effect("add", plain)

    async def meddler(ctx: scheduler.AgentContext) -> None:
        await ctx.call("read_row")
        # The two others wait to make their calls, and the dicts they
        # built them with are in this agent's reach.
        empty["item"] = "meddled"
        plain["item"] = "meddled"
        await ctx.call("read_row")

    worlds: list[_Shelf] = []
    result = scheduler.run(
        _shelf(worlds, m=meddler, r=reader, w=writer),
        scheduler.ReplayPolicy(["m", "r", "w", "m"]))
    assert _shown(result)[1:3] == [
        ("read_row", {}, ("a", 1)),
        ("add", {"item": "x"}, ["x"]),
    ]
    assert worlds[0].items == ["x"]


class _Duck:
    """Has what dict asks of a mapping, and is none."""

    def keys(self) -> list[str]:
        return ["items"]

    def __getitem__(self, key: str) -> str:
        return "x"


@pytest.mark.parametrize(
    "given,kind",
    [(5, "int"), (None, "NoneType"), ("ab", "str"), (["ab"], "list"),
     ([], "list"), ((), "tuple"), (set(), "set"), (_Duck(), "_Duck")],
    ids=["5", "None", "'ab'", "['ab']", "[]", "()", "set()", "duck"])
def test_call_whose_arguments_are_no_mapping_ends_the_run(given, kind):
    async def asker(ctx: scheduler.AgentContext) -> None:
        await scheduler.Effect("keep", given)

    worlds: list[_Shelf] = []
    coroutines = {}
    scenario = _shelf(
        worlds, a=_kept(_adder, coroutines), z=_kept(asker, coroutines))
    # dict makes arguments of a list of pairs and of anything empty. An
    # agent that awaits one of these has made no call.
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert str(caught.value) == (
        f"agent 'z' cannot call 'keep': its arguments are of type {kind}, "
        "not a mapping")
    assert worlds[0].kept is None and not worlds[0].items
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


class _Alarm:
    """A value whose repr interrupts whoever asks for it."""

    def __repr__(self) -> str:
        raise KeyboardInterrupt


class _Unreadable(scheduler.Effect):
    """A call whose tool raises when it is read."""

    @property
    def tool(self) -> str:
        raise LookupError("no tool today")


@pytest.mark.parametrize(
    "awaited,error",
    [
        # The value is refused, the refusal asks for its repr, and the
        # interrupt comes from there.
        (lambda: scheduler.Effect("keep", {"items": _Alarm()}),
         KeyboardInterrupt),
        # Made without its fields: a frozen call takes no tool it cannot
        # set.
        (lambda: object.__new__(_Unreadable), LookupError),
    ],
    ids=["interrupt", "tool-raises"])
def test_error_while_a_call_is_taken_closes_every_agent(awaited, error):
    async def asker(ctx: scheduler.AgentContext) -> None:
        await awaited()

    coroutines = {}
    scenario = _shelf(
        a=_kept(_adder, coroutines), z=_kept(asker, coroutines))
    # Not a refusal of the scheduler's, and still no agent is left
    # suspended behind it.
    with pytest.raises(error):
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


def test_error_while_the_agents_copy_is_made_closes_every_agent(monkeypatch):
    # Nothing but the machine makes the copy of a value that was already
    # taken fail, so the test stands in for that: the arguments are the
    # first take, the trace's copy of the list the second and the agent's
    # the third.
    take = values.take
    taken = []

    def failing(value: Any, *room: int) -> Any:
        taken.append(value)
        if len(taken) == 3:
            raise MemoryError("no room for a copy")
        return take(value, *room)

    monkeypatch.setattr(values, "take", failing)
    worlds: list[_Shelf] = []
    coroutines = {}
    scenario = _shelf(worlds, a=_kept(_adder, coroutines))
    with pytest.raises(MemoryError):
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert taken == [{"item": "a"}, ["a"], ["a"]]
    assert taken[1] is worlds[0].items and taken[2] is not worlds[0].items
    assert inspect.getcoroutinestate(coroutines["a"]) == inspect.CORO_CLOSED


def test_argument_no_call_carries_ends_the_run_before_its_tool_runs():
    handled = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.call("keep", items={1, 2})
        except Exception:
            # The refusal is not the agent's to handle.
            handled.append(ctx.name)

    worlds: list[_Shelf] = []
    coroutines = {}
    scenario = _shelf(
        worlds, a=_kept(_adder, coroutines), z=_kept(asker, coroutines))
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.ReplayPolicy(["a", "z"]))
    assert str(caught.value) == (
        "agent 'z' cannot call 'keep': a call does not carry {1, 2}, of "
        "type set")
    # Raised alone: its text holds what the value's own refusal said, so a
    # traceback does not show that one as a second error.
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert not handled
    assert worlds[0].kept is None and not worlds[0].items
    # The agent whose call was refused, and the one that was already
    # waiting to make its own.
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


def test_argument_is_refused_when_it_is_awaited():
    async def late(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", item=ctx.name)
        await ctx.call("keep", items={1, 2})

    async def early(ctx: scheduler.AgentContext) -> None:
        await ctx.call("keep", items={1, 2})

    worlds: list[_Shelf] = []
    coroutines = {}
    scenario = _shelf(
        worlds, a=_kept(_adder, coroutines), z=_kept(late, coroutines))
    # The call is refused as soon as z awaits it, after its first step.
    # The policy is not asked again, so a's call, which the schedule has
    # next and which could still run, is not made.
    with pytest.raises(scheduler.UnsupportedValueError, match="^agent 'z'"):
        scheduler.run(scenario, scheduler.ReplayPolicy(["z", "a"]))
    assert worlds[0].items == ["z"]
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}
    # At the start of a run too, before the policy is asked at all: with
    # no schedule it would refuse the first step.
    with pytest.raises(scheduler.UnsupportedValueError, match="^agent 'z'"):
        scheduler.run(
            _shelf(a=_adder, z=early), scheduler.ReplayPolicy([]))


def test_result_no_call_carries_ends_the_run_after_its_tool_ran():
    handled = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.call("mark")
        except Exception:
            # The refusal is not the agent's to handle.
            handled.append(ctx.name)

    worlds: list[_Shelf] = []
    coroutines = {}
    scenario = _shelf(
        worlds, a=_kept(asker, coroutines), b=_kept(_adder, coroutines))
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.ReplayPolicy(["a", "b"]))
    assert str(caught.value) == (
        "'mark' ran for agent 'a' and its result is refused: a call does "
        "not carry {1, 2}, of type set")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    # The tool ran, so the world is not as it was.
    assert worlds[0].kept == "marked"
    assert not handled
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "b": inspect.CORO_CLOSED}


def test_value_nested_too_deep_ends_the_run():
    deep: Any = 0
    for _ in range(99):
        deep = [deep]

    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("keep", items=deep)

    # The arguments are a dict, which leaves an argument 98 levels.
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot call 'keep': a call does not carry a "
                  "value nested so deep"):
        scheduler.run(_shelf(a=asker), scheduler.RandomPolicy(0))


def test_refusal_of_a_value_is_one_of_the_schedulers():
    assert issubclass(
        scheduler.UnsupportedValueError, scheduler.SchedulerError)


class _Vault:
    """A world whose value is nested: a dict of a list of lists."""

    def __init__(self) -> None:
        self.held: dict[str, Any] = {"rows": [["a"]]}
        self.kept: Any = None

    def read(self) -> dict[str, Any]:
        return self.held

    def grow(self) -> None:
        self.held["rows"][0].append("grown")

    def keep(self, value: dict[str, Any]) -> None:
        self.kept = value

    def spoil(self, value: dict[str, Any]) -> None:
        value["rows"][0].append("spoilt")

    def kept_rows(self) -> int:
        return len(self.kept["rows"][0])

    def gauge(self) -> float:
        return float("nan")


def _vault(**agents: scheduler.Agent) -> scheduler.Scenario:
    return scheduler.Scenario(
        name="vault", make_world=_Vault, agents=agents, invariants=())


def test_trace_copy_of_a_result_reaches_every_depth():
    async def reader(ctx: scheduler.AgentContext) -> None:
        await ctx.call("read")

    async def grower(ctx: scheduler.AgentContext) -> None:
        await ctx.call("grow")

    result = scheduler.run(
        _vault(a=reader, b=grower), scheduler.ReplayPolicy(["a", "b"]))
    # The list that the vault changes afterwards is the innermost one.
    assert result.steps[0].result == {"rows": [["a"]]}


def test_agents_copy_of_a_result_reaches_every_depth():
    async def scribbler(ctx: scheduler.AgentContext) -> None:
        got = await ctx.call("read")
        got["rows"][0].append("scribble")
        await ctx.call("read")

    result = scheduler.run(
        _vault(a=scribbler), scheduler.ReplayPolicy(["a", "a"]))
    # Neither the trace nor the vault holds the list the agent wrote on.
    assert [step.result for step in result.steps] == [{"rows": [["a"]]}] * 2


def test_arguments_are_taken_at_every_depth():
    async def keeper(ctx: scheduler.AgentContext) -> None:
        value = {"rows": [["a"]]}
        await ctx.call("keep", value=value)
        value["rows"][0].append("late")
        await ctx.call("kept_rows")

    result = scheduler.run(
        _vault(a=keeper), scheduler.ReplayPolicy(["a", "a"]))
    assert _shown(result) == [
        ("keep", {"value": {"rows": [["a"]]}}, None),
        ("kept_rows", {}, 1),
    ]


def test_tools_copy_of_an_argument_reaches_every_depth():
    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("spoil", value={"rows": [["a"]]})

    result = scheduler.run(_vault(a=asker), scheduler.ReplayPolicy(["a"]))
    assert _shown(result) == [("spoil", {"value": {"rows": [["a"]]}}, None)]


def test_invariant_is_asked_before_the_agent_is_given_its_result():
    seen = []

    async def reader(ctx: scheduler.AgentContext) -> None:
        await ctx.call("grow")
        seen.append("resumed")

    def asked(vault: _Vault) -> str | None:
        seen.append(f"asked at {len(vault.held['rows'][0])}")
        return None

    scenario = dataclasses.replace(
        _vault(a=reader),
        invariants=(scheduler.Invariant("heard", asked, every_step=True),))
    scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert seen == ["asked at 1", "asked at 2", "resumed"]


def test_interrupt_from_a_tool_closes_every_agent():
    class Alarm:
        def pull(self) -> None:
            raise KeyboardInterrupt

    async def puller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("pull")

    coroutines = {}
    scenario = scheduler.Scenario(
        name="alarm",
        make_world=Alarm,
        agents={
            "a": _kept(puller, coroutines), "b": _kept(puller, coroutines)},
        invariants=())
    # Not an Exception, and it still leaves no agent suspended.
    with pytest.raises(KeyboardInterrupt):
        scheduler.run(scenario, scheduler.ReplayPolicy(["a"]))
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "b": inspect.CORO_CLOSED}


def test_result_that_is_no_finite_number_ends_the_run():
    async def reader(ctx: scheduler.AgentContext) -> None:
        await ctx.call("gauge")

    # A float is a value a call carries, and this one is not: it is not
    # equal to itself, so a run that held it would not be equal to its
    # replay.
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^'gauge' ran for agent 'a' and its result is refused: a "
                  "call does not carry nan, of type float$"):
        scheduler.run(_vault(a=reader), scheduler.RandomPolicy(0))


class _Loud(str):
    pass


def test_tool_error_is_plain_text_whatever_the_error_says_it_is():
    class Boom(Exception):
        def __str__(self) -> str:
            return _Loud("boom")

    class Mine:
        def step(self) -> None:
            raise Boom()

    async def walker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("step")
        await ctx.call("step")

    scenario = scheduler.Scenario(
        name="mine", make_world=Mine, agents={"a": walker}, invariants=())
    result = scheduler.run(scenario, scheduler.ReplayPolicy(["a", "a"]))
    # A tool that raises does not end the run, and no call carries an
    # instance of a subclass of str.
    assert type(str(Boom())) is _Loud
    error = result.steps[0].result
    assert error == scheduler.Raised("Boom", "boom")
    assert type(error.kind) is str and type(error.message) is str
    assert len(result.steps) == 2


class _Sly(str):
    """A name that says it does not start with an underscore."""

    def startswith(self, *args: Any) -> bool:
        return False


class _Role(str, enum.Enum):
    SCRIBE = "record"


@pytest.mark.parametrize(
    "tool,shown",
    [
        (_Loud("record"), "'record', of type _Loud"),
        (_Sly("_hidden"), "'_hidden', of type _Sly"),
        (_Role.SCRIBE, "<_Role.SCRIBE: 'record'>, of type _Role"),
        (b"record", "b'record', of type bytes"),
        (None, "None, of type NoneType"),
        (7, "7, of type int"),
    ],
    ids=["subclass", "sly", "enum", "bytes", "none", "int"])
def test_tool_whose_name_is_no_string_ends_the_run_at_the_await(tool, shown):
    handled = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry=ctx.name)
        try:
            await ctx.call(tool, entry="late")
        except Exception:
            # The refusal is not the agent's to handle.
            handled.append(ctx.name)

    worlds: list[_Log] = []

    def make_world() -> _Log:
        worlds.append(_Log())
        return worlds[-1]

    coroutines = {}
    scenario = dataclasses.replace(
        _scenario(a=_kept(_writer, coroutines), z=_kept(asker, coroutines)),
        make_world=make_world)
    # A name that is more than a string can say what it likes of itself:
    # that it is equal to another name, or that it starts with no
    # underscore. No trace file holds one either. Refused when z awaits
    # the call, after its first step: the policy is not asked again, so
    # a's call, which the schedule has next, is not made.
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.ReplayPolicy(["z", "a"]))
    assert str(caught.value) == (
        f"agent 'z' cannot call a tool named {shown}: a tool's name is a "
        "string")
    assert not handled
    assert worlds[0].entries == ["z"]
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


@pytest.mark.parametrize(
    "name,shown",
    [
        (1, "1, of type int"),
        (None, "None, of type NoneType"),
        (_Loud("b"), "'b', of type _Loud"),
        (("b",), "('b',), of type tuple"),
    ],
    ids=["int", "none", "subclass", "tuple"])
def test_agent_whose_name_is_no_string_ends_the_run_before_it_starts(
    name, shown
):
    started = []
    worlds = []

    async def starter(ctx: scheduler.AgentContext) -> None:
        started.append(ctx.name)
        await ctx.call("record", entry="started")

    def make_world() -> _Log:
        worlds.append(_Log())
        return worlds[-1]

    scenario = scheduler.Scenario(
        name="log", make_world=make_world,
        agents={"a": starter, name: starter, "c": starter}, invariants=())
    # Before the names are put in order, which Python cannot do for a
    # string and a number, and before any agent is started or the world
    # is made.
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert str(caught.value) == (
        f"an agent cannot be named {shown}: an agent's name is a string")
    assert not started and not worlds


class _Listed:
    """A sampler that gives the values it was built with, in order.

    Attributes:
        asked: The agent and the request of each draw, as it was given
            them.
    """

    def __init__(self, *given: Any) -> None:
        self._given = iter(given)
        self.asked: list[tuple[str, Any]] = []

    def sample(self, agent: str, request: Any, moment: Any = None) -> Any:
        self.asked.append((agent, request))
        return next(self._given)


class _Watching:
    """A policy that takes the first waiting agent and keeps what it saw."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def choose(self, moment: scheduler.Moment) -> str:
        waiting = moment.waiting
        self.seen.append(tuple(waiting))
        return waiting[0]


async def _flipper(ctx: scheduler.AgentContext) -> None:
    """Records, draws, and records what it drew."""
    await ctx.call("record", entry=f"{ctx.name}:1")
    side = await ctx.draw("coin")
    await ctx.call("record", entry=f"{ctx.name}:{side}")


def test_agent_receives_what_the_sampler_gives():
    sampler = _Listed("heads")
    result = scheduler.run(
        _scenario(a=_flipper), scheduler.ReplayPolicy(["a", "a"]), sampler)
    assert _entries(result) == ["a:1", "a:heads"]
    assert sampler.asked == [("a", "coin")]
    assert result.draws == (scheduler.Drawn("a", "coin", "heads", 1),)


def test_draw_built_by_hand_is_a_draw():
    async def flipper(ctx: scheduler.AgentContext) -> None:
        side = await scheduler.Draw("coin")
        await ctx.call("record", entry=side)

    result = scheduler.run(
        _scenario(a=flipper), scheduler.ReplayPolicy(["a"]), _Listed("tails"))
    assert _entries(result) == ["tails"]
    assert result.draws == (scheduler.Drawn("a", "coin", "tails", 0),)


@pytest.mark.parametrize(
    "schedule", [["a", "b", "a", "b"], ["a", "b", "b", "a"]],
    ids=["a-goes-on-first", "b-goes-on-first"])
def test_draw_is_made_at_once_and_is_no_scheduling_point(schedule):
    result = scheduler.run(
        _scenario(a=_flipper, b=_flipper), scheduler.ReplayPolicy(schedule),
        _Listed("x", "y"))
    # Each agent draws as soon as its first call returns, and not when it
    # is next chosen: a's draw is made after step 1 and b's after step 2,
    # whichever of them goes on first.
    assert result.draws == (
        scheduler.Drawn("a", "coin", "x", 1),
        scheduler.Drawn("b", "coin", "y", 2),
    )
    assert result.schedule == tuple(schedule)


def test_policy_is_not_asked_at_a_draw():
    async def drawer(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("first")
        await ctx.call("record", entry=ctx.name)
        await ctx.draw("second")
        await ctx.draw("third")

    policy = _Watching()
    result = scheduler.run(
        _scenario(a=drawer, b=_writer), policy, _Listed(1, 2, 3))
    # Three calls, so three choices. Before the first of them a has drawn
    # and waits to call. After its call it draws twice and finishes with no
    # other agent between.
    assert policy.seen == [("a", "b"), ("b",), ("b",)]
    assert result.draws == (
        scheduler.Drawn("a", "first", 1, 0),
        scheduler.Drawn("a", "second", 2, 1),
        scheduler.Drawn("a", "third", 3, 1),
    )


def test_draws_before_the_first_call_are_made_in_name_order():
    async def opener(ctx: scheduler.AgentContext) -> None:
        await ctx.draw(ctx.name)
        await ctx.call("record", entry=ctx.name)

    result = scheduler.run(
        _scenario(c=opener, a=opener, b=opener), scheduler.RandomPolicy(0),
        _Listed(1, 2, 3))
    # Whatever order the scenario lists them in: a run is told from another
    # by its draws too, so the listing must not change them.
    assert result.draws == (
        scheduler.Drawn("a", "a", 1, 0),
        scheduler.Drawn("b", "b", 2, 0),
        scheduler.Drawn("c", "c", 3, 0),
    )


def test_agent_that_only_draws_makes_no_step():
    async def dreamer(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("coin")

    result = scheduler.run(
        _scenario(a=dreamer), scheduler.ReplayPolicy([]), _Listed("heads"))
    assert result == scheduler.RunResult(
        (), None, (scheduler.Drawn("a", "coin", "heads", 0),))


def test_run_that_makes_no_draw_holds_none():
    result = scheduler.run(
        _scenario(), scheduler.RandomPolicy(0), _Listed("unused"))
    assert result.draws == ()
    # And a run built by hand need not say so.
    assert scheduler.RunResult((), None) == scheduler.RunResult((), None, ())


def test_runs_that_differ_in_a_draw_are_not_equal():
    async def deaf(ctx: scheduler.AgentContext) -> None:
        # What it draws changes nothing it does.
        await ctx.draw("coin")
        await ctx.call("record", entry=ctx.name)

    def run(*given: Any) -> scheduler.RunResult:
        return scheduler.run(
            _scenario(a=deaf), scheduler.ReplayPolicy(["a"]), _Listed(*given))

    assert run("heads").steps == run("tails").steps
    assert run("heads") != run("tails")
    assert run("heads") == run("heads")


def test_draw_in_a_run_with_no_sampler_ends_the_run():
    handled = []

    async def flipper(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.draw("coin")
        except Exception:
            # The refusal is not the agent's to handle.
            handled.append(ctx.name)

    coroutines = {}
    scenario = _scenario(
        a=_kept(_writer, coroutines), z=_kept(flipper, coroutines))
    with pytest.raises(scheduler.UnsupportedAwaitError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert str(caught.value) == (
        "agent 'z' awaited a draw, and the run has no sampler")
    assert not handled
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


def test_run_with_no_sampler_runs_agents_that_do_not_draw():
    async def shy(ctx: scheduler.AgentContext) -> None:
        size = await ctx.call("size")
        if size > 9:
            await ctx.draw("coin")

    # Refused where a draw is awaited, and not for what an agent could do.
    result = scheduler.run(_scenario(a=shy), scheduler.RandomPolicy(0))
    assert len(result.steps) == 1 and result.draws == ()


def test_unsupported_await_names_what_an_agent_may_await():
    async def sleeper(ctx: scheduler.AgentContext) -> None:
        await asyncio.sleep(0)

    with pytest.raises(scheduler.UnsupportedAwaitError) as caught:
        scheduler.run(
            _scenario(a=sleeper), scheduler.RandomPolicy(0), _Listed())
    assert str(caught.value) == (
        "agent 'a' awaited NoneType; an agent may only await "
        "AgentContext.call and AgentContext.draw")


class _Meddling:
    """A sampler that writes on what it is given and keeps what it gives."""

    def __init__(self) -> None:
        self.given: list[Any] = []
        self.gave: list[list[str]] = []

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> list[str]:
        self.given.append(request)
        request.append("meddled")
        self.gave.append(["drawn"])
        return self.gave[-1]


def test_request_is_taken_when_the_agent_awaits_the_draw():
    mine = ["a", ["b"]]

    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.draw(mine)
        # No call: the agent changes its own list after the draw.
        mine[1].append("late")
        await ctx.call("record", entry=ctx.name)

    sampler = _Meddling()
    result = scheduler.run(
        _scenario(a=asker), scheduler.ReplayPolicy(["a"]), sampler)
    # The trace shows the request as the agent gave it. The sampler wrote
    # on a copy of its own, and the agent's list is the agent's alone.
    assert result.draws[0].request == ["a", ["b"]]
    assert sampler.given == [["a", ["b"], "meddled"]]
    assert mine == ["a", ["b", "late"]]
    assert sampler.given[0][1] is not mine[1]
    assert sampler.given[0][1] is not result.draws[0].request[1]


def test_drawn_value_is_shared_by_nobody():
    received = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        received.append(await ctx.draw([]))
        received[0].append("scribble")
        received.append(await ctx.draw([]))

    sampler = _Meddling()
    result = scheduler.run(
        _scenario(a=asker), scheduler.ReplayPolicy([]), sampler)
    # The sampler keeps the list it gave and writes on it afterwards.
    for gave in sampler.gave:
        gave.append("sampler's")
    assert [draw.value for draw in result.draws] == [["drawn"], ["drawn"]]
    assert received == [["drawn", "scribble"], ["drawn"]]
    assert sampler.gave == [["drawn", "sampler's"]] * 2


def test_draw_that_cannot_change_is_not_copied():
    request = (("heads", 1), ("tails", 1))
    value = ("heads", 7)
    received = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        received.append(await ctx.draw(request))

    sampler = _Listed(value)
    result = scheduler.run(
        _scenario(a=asker), scheduler.ReplayPolicy([]), sampler)
    assert sampler.asked[0][1] is request
    assert result.draws[0].request is request
    assert result.draws[0].value is value
    assert received[0] is value


def test_request_no_call_carries_ends_the_run_before_the_sampler_is_asked():
    handled = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.draw({1, 2})
        except Exception:
            # The refusal is not the agent's to handle.
            handled.append(ctx.name)

    sampler = _Listed("unused")
    coroutines = {}
    scenario = _scenario(
        a=_kept(_writer, coroutines), z=_kept(asker, coroutines))
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0), sampler)
    assert str(caught.value) == (
        "agent 'z' cannot draw: a call does not carry {1, 2}, of type set")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert not handled and not sampler.asked
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


@pytest.mark.parametrize(
    "given,shown",
    [({1, 2}, "{1, 2}, of type set"), (float("nan"), "nan, of type float")],
    ids=["set", "nan"])
def test_drawn_value_no_call_carries_ends_the_run(given, shown):
    handled = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        try:
            await ctx.draw("coin")
        except Exception:
            handled.append(ctx.name)

    sampler = _Listed(given)
    coroutines = {}
    scenario = _scenario(
        a=_kept(_writer, coroutines), z=_kept(asker, coroutines))
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0), sampler)
    assert str(caught.value) == (
        "a draw was made for agent 'z' and its value is refused: a call "
        f"does not carry {shown}")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    # The sampler was asked all the same: it may have spent something.
    assert sampler.asked == [("z", "coin")]
    assert not handled
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


def _nested(levels: int) -> Any:
    deep: Any = 0
    for _ in range(levels):
        deep = [deep]
    return deep


def test_draw_is_nested_as_deep_as_a_result_and_no_deeper():
    def run(request: Any, value: Any) -> scheduler.RunResult:
        async def asker(ctx: scheduler.AgentContext) -> None:
            await ctx.draw(request)

        return scheduler.run(
            _scenario(a=asker), scheduler.ReplayPolicy([]), _Listed(value))

    # Neither is inside a dict of arguments, so each has the 99 levels
    # that a result has.
    assert run(_nested(99), _nested(99)).draws[0] == scheduler.Drawn(
        "a", _nested(99), _nested(99), 0)
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot draw: a call does not carry a value "
                  "nested so deep"):
        run(_nested(100), 0)
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^a draw was made for agent 'a' and its value is refused: "
                  "a call does not carry a value nested so deep"):
        run(0, _nested(100))


class _Failing:
    """A sampler that raises what it was built with."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        scheduler.ReplayDivergenceError("from a sampler"),
        ValueError("from a sampler"),
        KeyboardInterrupt(),
    ],
    ids=["divergence", "error", "interrupt"])
def test_error_of_a_sampler_ends_the_run_and_closes_every_agent(error):
    handled = []

    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry=ctx.name)
        try:
            await ctx.draw("coin")
        except Exception:
            # No error of a sampler is the agent's to handle, where a
            # tool's is: a draw that failed gave the agent nothing.
            handled.append(ctx.name)

    coroutines = {}
    scenario = _scenario(
        a=_kept(_writer, coroutines), z=_kept(asker, coroutines))
    with pytest.raises(type(error)) as caught:
        scheduler.run(
            scenario, scheduler.ReplayPolicy(["z", "a"]), _Failing(error))
    assert caught.value is error
    assert not handled
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


class _Sealed(scheduler.Draw):
    """A draw whose request raises when it is read."""

    @property
    def request(self) -> Any:
        raise LookupError("no request today")


def test_error_while_a_draw_is_read_closes_every_agent():
    async def asker(ctx: scheduler.AgentContext) -> None:
        # Made without its field: a frozen draw takes no request it
        # cannot set.
        await object.__new__(_Sealed)

    coroutines = {}
    scenario = _scenario(
        a=_kept(_writer, coroutines), z=_kept(asker, coroutines))
    with pytest.raises(LookupError):
        scheduler.run(scenario, scheduler.RandomPolicy(0), _Listed("unused"))
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


def test_invariants_are_asked_before_an_agent_draws():
    seen = []

    class Noting(_Listed):
        def sample(
            self, agent: str, request: Any, moment: Any = None
        ) -> Any:
            seen.append("drew")
            return super().sample(agent, request)

    async def switcher(ctx: scheduler.AgentContext) -> None:
        await ctx.call("switch")
        await ctx.draw("coin")

    def asked(lamp: _Lamp) -> str | None:
        seen.append(f"asked at {lamp.switches}")
        return None

    scenario = dataclasses.replace(
        _lamp(0), agents={"a": switcher},
        invariants=(scheduler.Invariant("heard", asked, every_step=True),))
    result = scheduler.run(scenario, scheduler.RandomPolicy(0), Noting("x"))
    # The draw comes when the agent goes on, which is after its step was
    # judged: step 1 is the step the draw followed.
    assert seen == ["asked at 0", "asked at 1", "drew"]
    assert result.draws == (scheduler.Drawn("a", "coin", "x", 1),)


_COIN = (("heads", 1), ("tails", 1))
_NO_DISTRIBUTION = (
    "a written distribution is pairs of an outcome and a weight, each "
    "weight a positive integer")


def _flips(sampler: scheduler.Sampler, count: int) -> str:
    """Flips a coin a number of times and returns the sides' first letters."""
    return "".join(
        sampler.sample("a", _COIN)[0].upper() for _ in range(count))


def test_seeded_sampler_gives_the_draws_of_its_seed():
    # Written out, as the schedule of a seed is in the README: a seed that
    # found a failure must find it again under another Python.
    assert _flips(scheduler.SeededSampler(0), 16) == "HTTTTHTHTHTHHHTT"
    assert _flips(scheduler.SeededSampler(1), 16) == "HHTHHTTTTTHHTTTT"
    assert _flips(scheduler.SeededSampler(0), 16) == "HTTTTHTHTHTHHHTT"


def test_seeded_sampler_draws_by_weight():
    sampler = scheduler.SeededSampler(7)
    request = (("one", 1), ("two", 2), ("three", 3))
    drawn = collections.Counter(
        sampler.sample("a", request) for _ in range(6000))
    # A sixth, a third and a half of 6000 draws are 1000, 2000 and 3000,
    # with standard deviations of 29, 37 and 39. The counts of this seed
    # are each within two of them.
    assert drawn == {"one": 949, "two": 2042, "three": 3009}


@pytest.mark.parametrize(
    "request_,always",
    [
        ((("rare", 1), ("common", 10**12)), "common"),
        ((("common", 10**12), ("rare", 1)), "common"),
        ((("only", 1),), "only"),
    ],
    ids=["heavy-last", "heavy-first", "one-outcome"])
def test_weight_goes_with_its_outcome(request_, always):
    sampler = scheduler.SeededSampler(0)
    assert {sampler.sample("a", request_) for _ in range(200)} == {always}


def test_written_distribution_may_be_lists():
    # As a sampler is given one that an agent built of lists.
    sampler = scheduler.SeededSampler(0)
    assert "".join(
        sampler.sample("a", [["H", 1], ["T", 1]]) for _ in range(16)
    ) == "HTTTTHTHTHTHHHTT"


def test_seeded_sampler_draws_in_the_order_asked_whoever_asks():
    one = scheduler.SeededSampler(3)
    other = scheduler.SeededSampler(3)
    # One stream for every agent. So what an agent draws depends on the
    # draws made before it, by any agent, and with that on the schedule.
    assert [one.sample(agent, _COIN) for agent in "aabbab"] == [
        other.sample(agent, _COIN) for agent in "bababa"]


def test_seeded_sampler_does_not_draw_from_the_stream_of_its_policy():
    sides = [side for side, _ in _COIN]
    for seed in range(20):
        policy = scheduler.RandomPolicy(seed)
        sampler = scheduler.SeededSampler(seed)
        # A policy looks at who waits and at nothing else of the moment.
        chosen = [
            policy.choose(types.SimpleNamespace(waiting=sides))
            for _ in range(32)]
        drawn = [sampler.sample("a", _COIN) for _ in range(32)]
        # A search gives both the same seed. From one stream, every draw
        # would be tied to a choice of the schedule. Apart, they are
        # alike by chance once in 2**32.
        assert chosen != drawn


@pytest.mark.parametrize(
    "given",
    [
        "coin",
        5,
        None,
        (),
        [],
        {"heads": 1},
        (("heads",),),
        (("heads", 1, 2),),
        ("h1",),
        ({"heads": 1},),
        (("heads", 0),),
        (("heads", -1),),
        (("heads", True),),
        (("heads", 1.0),),
        (("heads", "1"),),
        (("heads", 1), "tails"),
        (("heads", 1), ("tails", 0)),
    ],
    ids=repr)
def test_seeded_sampler_refuses_what_is_no_written_distribution(given):
    sampler = scheduler.SeededSampler(0)
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        sampler.sample("a", given)
    assert str(caught.value) == (
        f"agent 'a' cannot draw from {given!r}: {_NO_DISTRIBUTION}")
    # And nothing was drawn for it: the next draw is the first of the seed.
    assert _flips(sampler, 16) == "HTTTTHTHTHTHHHTT"


def test_refusal_of_a_distribution_shows_no_more_than_the_start_of_it():
    given = "x" * 500
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.SeededSampler(0).sample("a", given)
    assert str(caught.value) == (
        f"agent 'a' cannot draw from '{'x' * 199}... (502 characters): "
        f"{_NO_DISTRIBUTION}")


def test_draw_from_no_written_distribution_ends_the_run():
    coroutines = {}
    scenario = _scenario(
        a=_kept(_writer, coroutines), z=_kept(_flipper, coroutines))
    # The flipper asks to draw from a word, which the sampler of the tests
    # takes and this one does not.
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'z' cannot draw from 'coin': a written"):
        scheduler.run(
            scenario, scheduler.ReplayPolicy(["z"]),
            scheduler.SeededSampler(0))
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "z": inspect.CORO_CLOSED}


def test_replay_sampler_gives_the_draws_back_in_order():
    held = [
        scheduler.Drawn("a", "coin", "heads", 0),
        scheduler.Drawn("b", "die", 6, 2),
    ]
    sampler = scheduler.ReplaySampler(held)
    # Its own: what the caller does to the list afterwards changes nothing.
    held.clear()
    assert sampler.sample("a", "coin") == "heads"
    assert sampler.sample("b", "die") == 6


def test_replay_sampler_refuses_a_draw_it_does_not_hold():
    sampler = scheduler.ReplaySampler(
        [scheduler.Drawn("a", "coin", "heads", 0)])
    sampler.sample("a", "coin")
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        sampler.sample("b", "coin")
    assert str(caught.value) == (
        "the draws end after 1 but agent 'b' asks for one more")
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.ReplaySampler([]).sample("a", "coin")
    assert str(caught.value) == (
        "the draws end after 0 but agent 'a' asks for one more")


def test_replay_sampler_refuses_a_draw_that_another_agent_made():
    sampler = scheduler.ReplaySampler([
        scheduler.Drawn("a", "coin", "heads", 0),
        scheduler.Drawn("a", "coin", "tails", 0),
    ])
    sampler.sample("a", "coin")
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        sampler.sample("b", "coin")
    assert str(caught.value) == (
        "draw 2 names 'a', but it is 'b' that draws")


def test_replay_sampler_refuses_a_draw_from_something_else():
    sampler = scheduler.ReplaySampler([
        scheduler.Drawn("a", "coin", "heads", 0),
        scheduler.Drawn("a", ("coin", 1), "tails", 0),
    ])
    sampler.sample("a", "coin")
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        sampler.sample("a", ["coin", 1])
    # The value would be given to an agent that asked for something else,
    # and the run would then be told from its record only at its end.
    assert str(caught.value) == (
        "draw 2 is from ('coin', 1), but agent 'a' asks to draw from "
        "['coin', 1]")


def test_refusal_of_a_draw_shows_no_more_than_the_start_of_a_request():
    sampler = scheduler.ReplaySampler(
        [scheduler.Drawn("a", "x" * 500, "heads", 0)])
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        sampler.sample("a", "y" * 500)
    assert str(caught.value) == (
        f"draw 1 is from '{'x' * 199}... (502 characters), but agent 'a' "
        f"asks to draw from '{'y' * 199}... (502 characters)")


class _Looking:
    """A policy and a sampler that keep what they are shown.

    As a policy it takes the first waiting agent, and as a sampler it
    gives the number of the draw.

    Attributes:
        choices: What the moment showed at each choice.
        asked: Who drew from what at each draw, and what the moment showed.
        moments: Every moment it was shown.
    """

    def __init__(self) -> None:
        self.choices: list[dict[str, Any]] = []
        self.asked: list[dict[str, Any]] = []
        self.moments: list[scheduler.Moment] = []

    def _shown(self, moment: scheduler.Moment) -> dict[str, Any]:
        self.moments.append(moment)
        return {
            "waiting": moment.waiting,
            "calls": moment.calls,
            "steps": moment.steps,
            "draws": moment.draws,
            "violation": moment.violation,
        }

    def choose(self, moment: scheduler.Moment) -> str:
        self.choices.append(self._shown(moment))
        return moment.waiting[0]

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> int:
        self.asked.append(
            {"agent": agent, "request": request, **self._shown(moment)})
        return len(self.asked)


def _record(entry: str) -> scheduler.Effect:
    return scheduler.Effect("record", {"entry": entry})


def test_policy_is_shown_who_waits_with_which_call_and_what_has_happened():
    policy = _Looking()
    result = scheduler.run(_scenario(), policy)
    assert result.schedule == ("a", "a", "b", "b")
    steps = result.steps
    assert policy.choices == [
        {"waiting": ["a", "b"],
         "calls": {"a": _record("a:1"), "b": _record("b:1")},
         "steps": (), "draws": (), "violation": None},
        {"waiting": ["a", "b"],
         "calls": {"a": _record("a:2"), "b": _record("b:1")},
         "steps": steps[:1], "draws": (), "violation": None},
        # a has finished: it is no longer among those that wait.
        {"waiting": ["b"], "calls": {"b": _record("b:1")},
         "steps": steps[:2], "draws": (), "violation": None},
        {"waiting": ["b"], "calls": {"b": _record("b:2")},
         "steps": steps[:3], "draws": (), "violation": None},
    ]
    # In name order, the calls as the agents: a file that holds them and a
    # search that goes by them must not depend on which agent awaited
    # first.
    assert list(policy.choices[0]["calls"]) == ["a", "b"]


def test_policy_is_shown_the_agents_in_name_order_whoever_waited_first():
    async def late(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry="first")
        await ctx.call("record", entry="second")

    policy = _Looking()
    # After its first call a waits again, and it is b that has waited
    # longest.
    scheduler.run(_scenario(a=late, b=_writer), policy)
    assert policy.choices[1]["waiting"] == ["a", "b"]
    assert list(policy.choices[1]["calls"]) == ["a", "b"]


def test_policy_that_changes_what_it_is_shown_changes_nothing_of_the_run():
    class Meddler:
        def choose(self, moment: scheduler.Moment) -> str:
            waiting = moment.waiting
            calls = moment.calls
            # Its own list and its own copies: none of this is the run's.
            for call in calls.values():
                call.args.setdefault("items", []).append("meddled")
                call.args["more"] = 1
            calls.clear()
            first = waiting[0]
            waiting.clear()
            return first

    async def keeper(ctx: scheduler.AgentContext) -> None:
        await ctx.call("keep", items=["mine"])
        await ctx.call("kept_size")

    worlds: list[_Shelf] = []
    result = scheduler.run(_shelf(worlds, a=keeper, b=keeper), Meddler())
    assert _shown(result) == [
        ("keep", {"items": ["mine"]}, None),
        ("kept_size", {}, 1),
        ("keep", {"items": ["mine"]}, None),
        ("kept_size", {}, 1),
    ]
    assert worlds[0].kept == ["mine"]


def test_moment_gives_what_it_shows_anew_each_time_it_is_asked():
    seen = []

    class Asker:
        def choose(self, moment: scheduler.Moment) -> str:
            seen.append((
                moment.waiting, moment.waiting, moment.calls, moment.calls))
            return moment.waiting[0]

    async def keeper(ctx: scheduler.AgentContext) -> None:
        await ctx.call("keep", items=["mine"])

    scheduler.run(_shelf(a=keeper), Asker())
    waiting, again, calls, calls_again = seen[0]
    assert waiting == again and waiting is not again
    assert calls == calls_again and calls is not calls_again
    assert calls["a"].args["items"] is not calls_again["a"].args["items"]


def test_moment_shows_the_run_as_it_stands_when_it_is_looked_at():
    policy = _Looking()
    result = scheduler.run(_scenario(), policy)
    # One moment for the whole run. It keeps nothing: looked at once the
    # run is over, it shows the run as it ended.
    assert len({id(moment) for moment in policy.moments}) == 1
    moment = policy.moments[0]
    assert moment.waiting == [] and moment.calls == {}
    assert moment.steps == result.steps
    # What was taken from it at a choice is as it was then.
    assert policy.choices[0]["steps"] == ()


def test_policy_is_shown_the_first_violation_from_the_step_that_broke_it():
    policy = _Looking()
    result = scheduler.run(_lamp(3, _ALWAYS_DARK), policy)
    broken = scheduler.Violation("always-dark", "the lamp is lit", 1)
    assert result.violation == broken
    # Lit after the first switch, dark after the second and lit again:
    # the first violation stays the one that is shown.
    assert [choice["violation"] for choice in policy.choices] == [
        None, broken, broken]


def test_policy_is_shown_a_violation_of_the_fresh_world():
    policy = _Looking()
    scheduler.run(_lamp(1, _ALWAYS_DARK, lit=True), policy)
    assert [choice["violation"] for choice in policy.choices] == [
        scheduler.Violation("always-dark", "the lamp is lit", 0)]


def test_policy_is_not_shown_what_is_asked_only_at_the_end():
    policy = _Looking()
    result = scheduler.run(_lamp(1, _DARK_AT_END), policy)
    assert result.violation is not None
    assert [choice["violation"] for choice in policy.choices] == [None]


def test_sampler_is_shown_the_run_as_it_stands_at_the_draw():
    async def drawer(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("start")
        await ctx.call("record", entry=ctx.name)
        await ctx.draw("after")
        await ctx.draw("again")

    looking = _Looking()
    result = scheduler.run(
        _scenario(a=drawer, b=_writer), scheduler.ReplayPolicy(["a", "b", "b"]),
        looking)
    first, second, third = result.draws
    assert looking.asked == [
        # Before any call. a is started first and draws before it waits,
        # and b is not started yet: nobody waits.
        {"agent": "a", "request": "start", "waiting": [], "calls": {},
         "steps": (), "draws": (), "violation": None},
        # After a's call. The step is among the steps, the draw that is
        # being made is not among the draws, and the agent that draws is
        # not among those that wait.
        {"agent": "a", "request": "after", "waiting": ["b"],
         "calls": {"b": _record("b:1")}, "steps": result.steps[:1],
         "draws": (first,), "violation": None},
        {"agent": "a", "request": "again", "waiting": ["b"],
         "calls": {"b": _record("b:1")}, "steps": result.steps[:1],
         "draws": (first, second), "violation": None},
    ]
    assert (first.value, second.value, third.value) == (1, 2, 3)


def test_sampler_is_shown_a_violation_of_the_step_its_agent_just_took():
    async def switcher(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("before")
        await ctx.call("switch")
        await ctx.draw("after")

    looking = _Looking()
    scenario = dataclasses.replace(
        _lamp(0, _ALWAYS_DARK), agents={"a": switcher})
    scheduler.run(scenario, scheduler.ReplayPolicy(["a"]), looking)
    assert [asked["violation"] for asked in looking.asked] == [
        None, scheduler.Violation("always-dark", "the lamp is lit", 1)]


def test_draw_before_the_first_call_is_made_before_the_fresh_world_is_asked():
    async def drawer(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("start")
        await ctx.call("switch")

    looking = _Looking()
    scenario = dataclasses.replace(
        _lamp(0, _ALWAYS_DARK, lit=True), agents={"a": drawer})
    result = scheduler.run(scenario, looking, looking)
    # Agents are started, and draw, before the fresh world is asked. So
    # the sampler is shown no violation where the policy, asked next, is.
    assert result.violation == scheduler.Violation(
        "always-dark", "the lamp is lit", 0)
    assert looking.asked[0]["violation"] is None
    assert looking.choices[0]["violation"] == result.violation


def _ticketed(
    worlds: list[_Log] | None = None, tools: Iterator[str] | None = None
) -> scheduler.Scenario:
    """An agent that records a ticket whose number outlives the run.

    So its call is another call in every run. With tools, the tool it
    calls is the next of them.
    """
    tickets = itertools.count()

    async def clerk(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry=ctx.name)
        tool = "record" if tools is None else next(tools)
        if tool == "record":
            await ctx.call("record", entry=f"ticket {next(tickets)}")
        else:
            await ctx.call(tool)

    def make_world() -> _Log:
        world = _Log()
        if worlds is not None:
            worlds.append(world)
        return world

    return scheduler.Scenario(
        name="ticketed", make_world=make_world,
        agents={"a": clerk, "b": _writer}, invariants=())


def test_replay_of_steps_refuses_another_call_before_it_is_made():
    worlds: list[_Log] = []
    scenario = _ticketed(worlds)
    first = scheduler.run(scenario, scheduler.ReplayPolicy("aabb"))
    assert _entries(first) == ["a", "ticket 0", "b:1", "b:2"]
    # Given the names, a replay makes every call of the schedule and only
    # then is seen to be another run.
    by_names = scheduler.run(
        scenario, scheduler.ReplayPolicy(first.schedule))
    assert by_names != first
    assert worlds[1].entries == ["a", "ticket 1", "b:1", "b:2"]
    # Given the steps, it holds each call against its step when it
    # chooses, and the call that is another is not made.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(scenario, scheduler.ReplayPolicy(first.steps))
    assert str(caught.value) == (
        "step 2 is record(entry='ticket 0'), but agent 'a' waits to call "
        "record(entry='ticket 2')")
    assert worlds[2].entries == ["a"]


def test_replay_of_steps_refuses_a_call_of_another_tool():
    scenario = _ticketed(tools=iter(["record", "size"]))
    first = scheduler.run(scenario, scheduler.ReplayPolicy("aabb"))
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(scenario, scheduler.ReplayPolicy(first.steps))
    assert str(caught.value) == (
        "step 2 is record(entry='ticket 0'), but agent 'a' waits to call "
        "size()")


def test_replay_of_steps_refuses_a_call_with_other_arguments_than_its_step():
    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("tag", label="x", into=["a", ["b"]])

    def run(*steps: scheduler.Step) -> scheduler.RunResult:
        return scheduler.run(_shelf(a=caller), scheduler.ReplayPolicy(steps))

    step = scheduler.Step(
        "a", "tag", {"label": "x", "into": ["a", ["b"]]}, None)
    assert run(step).steps == (step,)
    for args, shown in [
        ({"label": "x"}, "tag(label='x')"),
        ({"label": "x", "into": ["a", ["c"]]},
         "tag(label='x', into=['a', ['c']])"),
        ({"label": "x", "into": ["a", ["b"]], "more": 1},
         "tag(label='x', into=['a', ['b']], more=1)"),
        ({}, "tag()"),
    ]:
        with pytest.raises(scheduler.ReplayDivergenceError) as caught:
            run(dataclasses.replace(step, args=args))
        assert str(caught.value) == (
            f"step 1 is {shown}, but agent 'a' waits to call "
            "tag(label='x', into=['a', ['b']])")


def test_replay_of_steps_takes_a_call_that_is_equal_as_runs_are_compared():
    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("count", seen=(True, 2))

    # The step has 1 where the call has True, and 2.0 where it has 2:
    # equal as Python compares, which is how a run is held against its
    # replay. A trace file holds them apart, as text.
    step = scheduler.Step("a", "count", {"seen": (1, 2.0)}, 2)
    result = scheduler.run(_shelf(a=caller), scheduler.ReplayPolicy([step]))
    assert result.steps == (step,)
    assert type(result.steps[0].args["seen"][0]) is bool


def test_replay_of_steps_names_an_agent_that_is_not_waiting_as_names_do():
    first = scheduler.run(_scenario(), scheduler.ReplayPolicy("aabb"))
    steps = (first.steps[0], first.steps[1], first.steps[1], first.steps[2])
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_scenario(), scheduler.ReplayPolicy(steps))
    assert str(caught.value) == (
        "step 3 names 'a', which is not waiting; waiting: b")
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^the schedule ends after 2 steps but b can still run$"):
        scheduler.run(_scenario(), scheduler.ReplayPolicy(first.steps[:2]))


def test_replay_takes_names_and_steps_in_one_schedule():
    first = scheduler.run(_scenario(), scheduler.ReplayPolicy("abab"))
    mixed = ("a", first.steps[1], "a", first.steps[3])
    assert scheduler.run(_scenario(), scheduler.ReplayPolicy(mixed)) == first


def test_refusal_of_a_call_shows_no_more_than_the_start_of_an_argument():
    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry="y" * 500)

    step = scheduler.Step("a", "record", {"entry": "x" * 500}, None)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_scenario(a=caller), scheduler.ReplayPolicy([step]))
    assert str(caught.value) == (
        f"step 1 is record(entry='{'x' * 199}... (502 characters)), but "
        f"agent 'a' waits to call record(entry='{'y' * 199}... (502 "
        "characters))")


def _lamp_state(lamp: _Lamp) -> tuple[bool, int]:
    return lamp.lit, lamp.switches


class _Stateful(_Looking):
    """A policy and a sampler that also keep the state they are shown."""

    def __init__(self) -> None:
        super().__init__()
        self.states: list[Any] = []

    def choose(self, moment: scheduler.Moment) -> str:
        self.states.append(moment.state())
        return super().choose(moment)

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> int:
        self.states.append(moment.state())
        return super().sample(agent, request, moment)


def test_policy_is_shown_the_state_of_the_world_as_its_snapshot_gives_it():
    policy = _Stateful()
    scenario = dataclasses.replace(_lamp(3), snapshot=_lamp_state)
    scheduler.run(scenario, policy)
    # At each choice, the world as the calls so far have left it.
    assert policy.states == [(False, 0), (True, 1), (False, 2)]


def test_sampler_is_shown_the_state_of_the_world_at_its_draw():
    async def switcher(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("before")
        await ctx.call("switch")
        await ctx.draw("after")

    looking = _Stateful()
    scenario = dataclasses.replace(
        _lamp(0), agents={"a": switcher}, snapshot=_lamp_state)
    scheduler.run(scenario, scheduler.ReplayPolicy(["a"]), looking)
    assert looking.states == [(False, 0), (True, 1)]


def test_scenario_needs_no_snapshot_until_it_is_asked_for_a_state():
    scenario = _lamp(2)
    assert scenario.snapshot is None
    # A run of it is made, as ever.
    assert len(scheduler.run(scenario, _Looking()).steps) == 2
    coroutines = {}
    asked = dataclasses.replace(
        scenario, agents={"a": _kept(scenario.agents["a"], coroutines)})
    with pytest.raises(scheduler.MissingSnapshotError) as caught:
        scheduler.run(asked, _Stateful())
    assert str(caught.value) == (
        "the scenario 'lamp' has no snapshot to give the state of its "
        "world")
    assert inspect.getcoroutinestate(coroutines["a"]) == inspect.CORO_CLOSED


def test_refusal_of_a_missing_snapshot_is_one_of_the_schedulers():
    assert issubclass(
        scheduler.MissingSnapshotError, scheduler.SchedulerError)


def test_state_that_cannot_be_hashed_is_refused():
    scenario = dataclasses.replace(
        _lamp(1), snapshot=lambda lamp: [lamp.lit, lamp.switches])
    # A state is held against other states and looked up by, and a list
    # that the world goes on to change is neither.
    with pytest.raises(TypeError) as caught:
        scheduler.run(scenario, _Stateful())
    assert str(caught.value) == (
        "the snapshot of 'lamp' gave [False, 0], which cannot be hashed: "
        "unhashable type: 'list'")


def test_state_of_a_world_is_what_the_scenarios_snapshot_gives():
    scenario = dataclasses.replace(_lamp(1), snapshot=_lamp_state)
    lamp = _Lamp(True)
    lamp.switch()
    assert scheduler.state_of(scenario, lamp) == (False, 1)
    with pytest.raises(scheduler.MissingSnapshotError, match="'lamp'"):
        scheduler.state_of(_lamp(1), lamp)


@pytest.mark.parametrize(
    "entry,shown",
    [
        (["a"], "['a'], of type list"),
        ({"a": 1}, "{'a': 1}, of type dict"),
        (None, "None, of type NoneType"),
        (7, "7, of type int"),
        # Equal to the name of an agent that waits, and more than a
        # string: no agent has such a name.
        (_Loud("a"), "'a', of type _Loud"),
        (_Role.SCRIBE, "<_Role.SCRIBE: 'record'>, of type _Role"),
    ],
    ids=["list", "dict", "none", "int", "subclass", "enum"])
def test_replay_refuses_an_entry_that_is_no_name_of_an_agent(entry, shown):
    # Also one that cannot be looked up, and in words of its own: to say
    # of 'a' that it is not waiting, beside a that waits, would say
    # nothing.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_scenario(), scheduler.ReplayPolicy([entry]))
    assert str(caught.value) == (
        f"step 1 names {shown}: an agent's name is a string")


def test_replay_of_steps_refuses_another_tool_called_with_the_same_arguments():
    class Door:
        def open(self) -> None:
            pass

        def shut(self) -> None:
            pass

    async def porter(ctx: scheduler.AgentContext) -> None:
        await ctx.call("shut")

    scenario = scheduler.Scenario(
        name="door", make_world=Door, agents={"a": porter}, invariants=())
    # Neither call has an argument, so only the tool tells them apart.
    step = scheduler.Step("a", "open", {}, None)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(scenario, scheduler.ReplayPolicy([step]))
    assert str(caught.value) == (
        "step 1 is open(), but agent 'a' waits to call shut()")


class _Stride(scheduler.Step):
    """A step of another type, which can say of itself what it likes."""


def test_replay_takes_a_step_and_nothing_that_passes_for_one():
    stride = _Stride("a", "record", {"entry": "a:1"}, None)
    # Neither a name nor a step: it names nobody.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 1 names _Stride\\(agent='a', .*\\), of type "
                  "_Stride: an agent's name is a string$"):
        scheduler.run(_scenario(), scheduler.ReplayPolicy([stride]))
    # Nor is a step whose agent is more than a string the step of a run.
    loud = scheduler.Step(_Loud("a"), "record", {"entry": "a:1"}, None)
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^step 1 names 'a', of type _Loud: an agent's name is a "
                  "string$"):
        scheduler.run(_scenario(), scheduler.ReplayPolicy([loud]))


class _Who(str, enum.Enum):
    A = "a"


@pytest.mark.parametrize(
    "chosen,shown",
    [
        (_Loud("a"), "'a', of type _Loud"),
        (_Who.A, "<_Who.A: 'a'>, of type _Who"),
        (None, "None, of type NoneType"),
    ],
    ids=["subclass", "enum", "none"])
def test_policy_that_chooses_what_is_no_plain_string_ends_the_run(
    chosen, shown
):
    class Sly:
        def choose(self, moment: scheduler.Moment) -> Any:
            return chosen

    worlds: list[_Log] = []

    def make_world() -> _Log:
        worlds.append(_Log())
        return worlds[-1]

    coroutines = {}
    scenario = dataclasses.replace(
        _scenario(a=_kept(_writer, coroutines), b=_kept(_writer, coroutines)),
        make_world=make_world)
    # Equal to the name of an agent that waits, it would be looked up and
    # would then be the agent's name: in the steps, for the sampler and
    # among those that wait. No replay and no trace file takes such a run.
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, Sly())
    assert str(caught.value) == (
        f"a policy cannot choose {shown}: an agent's name is a string")
    assert worlds[0].entries == []
    assert {
        name: inspect.getcoroutinestate(coroutine)
        for name, coroutine in coroutines.items()
    } == {"a": inspect.CORO_CLOSED, "b": inspect.CORO_CLOSED}


class _CalledStr(type):
    """A metaclass whose classes say that they are called str."""

    @property
    def __name__(cls) -> str:
        return "str"


class _Masked(str, metaclass=_CalledStr):
    pass


class _MaskedBox(metaclass=_CalledStr):
    """Something an agent can await and that is no call and no draw."""

    def __await__(self) -> Any:
        value = yield self
        return value


def test_refusal_names_a_type_as_type_gives_it_whatever_the_class_says():
    assert _Masked.__name__ == "str"

    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call(_Masked("record"), entry="x")

    async def sleeper(ctx: scheduler.AgentContext) -> None:
        await _MaskedBox()

    async def packer(ctx: scheduler.AgentContext) -> None:
        await scheduler.Effect("record", _MaskedBox())

    cases = [
        (scheduler.Scenario(
            "log", _Log, {_Masked("b"): _writer}, ()),
         "an agent cannot be named 'b', of type _Masked: an agent's name "
         "is a string"),
        (_scenario(a=caller),
         "agent 'a' cannot call a tool named 'record', of type _Masked: a "
         "tool's name is a string"),
        (_scenario(a=sleeper),
         "agent 'a' awaited _MaskedBox; an agent may only await "
         "AgentContext.call and AgentContext.draw"),
        (_scenario(a=packer),
         "agent 'a' cannot call 'record': its arguments are of type "
         "_MaskedBox, not a mapping"),
    ]
    for scenario, message in cases:
        with pytest.raises(scheduler.SchedulerError) as caught:
            scheduler.run(scenario, scheduler.RandomPolicy(0))
        assert str(caught.value) == message


class _Shifty(Mapping):
    """Agents by name, with other names each time they are walked."""

    def __init__(self, agents: dict[str, scheduler.Agent]) -> None:
        self._agents = agents
        self.walks = 0

    def __getitem__(self, name: str) -> scheduler.Agent:
        return self._agents[name]

    def __iter__(self) -> Iterator[str]:
        self.walks += 1
        if self.walks == 1:
            return iter(self._agents)
        return iter([_Loud(name) for name in self._agents])

    def __len__(self) -> int:
        return len(self._agents)


def test_agents_are_walked_once_for_their_names():
    agents = _Shifty({"a": _writer, "b": _writer})
    result = scheduler.run(
        scheduler.Scenario("log", _Log, agents, ()),
        scheduler.RandomPolicy(0))
    # The names that were looked at are the names the agents run under.
    assert agents.walks == 1
    assert {type(step.agent) for step in result.steps} == {str}


@pytest.mark.parametrize(
    "args,shown",
    [
        (None, "record with None"),
        ((), "record with ()"),
        ([("entry", "a:1")], "record with [('entry', 'a:1')]"),
        ({1: "x"}, "record(1='x')"),
    ],
    ids=["none", "tuple", "pairs", "key-no-string"])
def test_replay_refuses_a_step_made_by_hand_as_a_replay_refuses(args, shown):
    step = scheduler.Step("a", "record", args, None)
    # Its arguments are no arguments of a call, and the refusal is still
    # that of a replay that left its schedule, not an error of its own.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_scenario(), scheduler.ReplayPolicy([step]))
    assert str(caught.value) == (
        f"step 1 is {shown}, but agent 'a' waits to call "
        "record(entry='a:1')")


def test_replay_refuses_a_step_whose_tool_is_no_string_as_a_replay_refuses():
    step = scheduler.Step("a", ("record",), {"entry": "a:1"}, None)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_scenario(), scheduler.ReplayPolicy([step]))
    assert str(caught.value) == (
        "step 1 is ('record',)(entry='a:1'), but agent 'a' waits to call "
        "record(entry='a:1')")


def test_refusal_of_a_state_shows_no_more_than_the_start_of_it():
    scenario = dataclasses.replace(
        _lamp(1), snapshot=lambda lamp: ["x" * 500])
    with pytest.raises(TypeError) as caught:
        scheduler.run(scenario, _Stateful())
    assert str(caught.value) == (
        f"the snapshot of 'lamp' gave ['{'x' * 198}... (504 characters), "
        "which cannot be hashed: unhashable type: 'list'")
