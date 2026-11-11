"""Checks the scheduler's ordering and its refusals."""

import asyncio
import inspect
from collections.abc import Coroutine
from typing import Any

import pytest

from anyorder import scheduler


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

    with pytest.raises(scheduler.UnknownToolError, match="'missing'"):
        scheduler.run(_scenario(a=lost), scheduler.RandomPolicy(0))


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
