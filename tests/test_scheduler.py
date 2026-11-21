"""Checks the scheduler's ordering, refusals, tool errors and invariants."""

import asyncio
import dataclasses
import functools
import inspect
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
