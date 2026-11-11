"""Checks schedule search and replay on the oversell scenarios."""

import itertools

import pytest

from anyorder import explore
from anyorder import scheduler
from anyorder.scenarios import oversell

_SERIAL = ("buyer-1", "buyer-1", "buyer-1", "buyer-2")
_BOTH_READ_FIRST = (
    "buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2")


def test_search_finds_the_oversell():
    failure = explore.explore(oversell.naive(), range(20))
    assert failure is not None
    assert failure.result.violation is not None
    assert "2 orders placed against a stock of 1" in failure.result.violation


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
    assert result.violation == (
        "2 orders placed against a stock of 1: buyer-1, buyer-2")


@pytest.mark.parametrize("stock,buyers", [(1, 2), (2, 5), (3, 3)])
def test_control_never_oversells(stock, buyers):
    scenario = oversell.atomic(stock=stock, buyers=buyers)
    assert explore.explore(scenario, range(200)) is None


def test_replay_refuses_steps_left_over():
    with pytest.raises(scheduler.ReplayDivergenceError, match="ended after 4"):
        explore.replay(oversell.naive(), _SERIAL + ("buyer-2",))


def test_search_refuses_a_failure_that_does_not_replay():
    tickets = itertools.count()

    class Counter:
        """A world whose tool reads state that outlives the run."""

        def take(self) -> int:
            return next(tickets)

    async def taker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take")

    scenario = scheduler.Scenario(
        name="leaky",
        make_world=Counter,
        agents={"a": taker},
        invariants=(lambda world: "always fails",),
    )
    with pytest.raises(scheduler.ReplayDivergenceError, match="'leaky'"):
        explore.explore(scenario, range(1))
