"""Checks the scheduler's ordering and its refusals."""

import asyncio

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
