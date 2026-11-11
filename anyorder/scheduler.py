"""Deterministic scheduler for interleaving agent tool calls.

An agent is a coroutine that reaches its world only through
AgentContext.call. Each call hands one Effect to the scheduler and suspends
the agent, so the scheduler always holds one pending effect per unfinished
agent. A policy picks which of them runs next, and its picks are the only
source of ordering: the list of chosen agent names is the schedule, and the
same scenario under the same schedule gives the same trace.

Nothing here reads the clock, starts a thread or uses an event loop.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable, Coroutine, Generator, Mapping, Sequence
from typing import Any, Protocol


class SchedulerError(Exception):
    """Base class for the scheduler's refusals."""


class ReplayDivergenceError(SchedulerError):
    """A run did not follow the schedule it was replaying."""


class UnknownToolError(SchedulerError):
    """An agent called a tool its world does not provide."""


class UnsupportedAwaitError(SchedulerError):
    """An agent awaited something other than a tool call."""


@dataclasses.dataclass(frozen=True)
class Effect:
    """One tool call an agent is waiting to make.

    Attributes:
        tool: Name of a public method on the scenario's world.
        args: Keyword arguments for the call.
    """

    tool: str
    args: Mapping[str, Any]

    def __await__(self) -> Generator[Effect, Any, Any]:
        result = yield self
        return result


class AgentContext:
    """An agent's handle on its run: its name and its world's tools."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def call(self, tool: str, **args: Any) -> Any:
        """Calls a tool once the scheduler lets this agent run.

        Args:
            tool: Name of a public method on the scenario's world.
            **args: Keyword arguments for the call.

        Returns:
            Whatever the tool returns.
        """
        return await Effect(tool, args)


Agent = Callable[[AgentContext], Coroutine[Any, Any, None]]
Invariant = Callable[[Any], str | None]


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A world, the agents acting on it and what must stay true.

    Attributes:
        name: Short identifier for reports.
        make_world: Builds a fresh world for each run. A world shared
            between runs breaks replay.
        agents: Agent functions by name.
        invariants: Checks made on the world once every agent has finished.
            Each returns a message when it is violated and None otherwise.
    """

    name: str
    make_world: Callable[[], Any]
    agents: Mapping[str, Agent]
    invariants: Sequence[Invariant]


@dataclasses.dataclass(frozen=True)
class Step:
    """One tool call that ran.

    Attributes:
        agent: The agent the policy chose.
        tool: The tool it called.
        args: The call's keyword arguments.
        result: What the tool returned.
    """

    agent: str
    tool: str
    args: Mapping[str, Any]
    result: Any


@dataclasses.dataclass(frozen=True)
class RunResult:
    """The trace of one run and what it broke, if anything.

    Attributes:
        steps: Every tool call, in the order it ran.
        violation: The message of the first invariant that failed, or None.
    """

    steps: tuple[Step, ...]
    violation: str | None

    @property
    def schedule(self) -> tuple[str, ...]:
        """The agent chosen at each step."""
        return tuple(step.agent for step in self.steps)


class Policy(Protocol):
    """Chooses which waiting agent runs next."""

    def choose(self, waiting: Sequence[str]) -> str:
        """Returns one name from waiting, which is sorted and not empty."""


class RandomPolicy:
    """Chooses uniformly from a seeded generator."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose(self, waiting: Sequence[str]) -> str:
        return self._rng.choice(waiting)


class ReplayPolicy:
    """Follows a recorded schedule and refuses to leave it."""

    def __init__(self, schedule: Sequence[str]) -> None:
        self._schedule = tuple(schedule)
        self._position = 0

    def choose(self, waiting: Sequence[str]) -> str:
        if self._position == len(self._schedule):
            raise ReplayDivergenceError(
                f"the schedule ends after {self._position} steps but "
                f"{', '.join(waiting)} can still run")
        name = self._schedule[self._position]
        if name not in waiting:
            raise ReplayDivergenceError(
                f"step {self._position + 1} names {name!r}, which is not "
                f"waiting; waiting: {', '.join(waiting)}")
        self._position += 1
        return name


_Pending = dict[str, tuple[Coroutine[Any, Any, None], Effect]]


def run(scenario: Scenario, policy: Policy) -> RunResult:
    """Runs a scenario until every agent has finished.

    Args:
        scenario: What to run.
        policy: Chooses the agent for each step, including steps where only
            one agent is waiting, so a schedule has one entry per step.

    Returns:
        The trace and the first violated invariant, if any.

    Raises:
        UnknownToolError: An agent called a tool the world lacks.
        UnsupportedAwaitError: An agent awaited something other than
            AgentContext.call.
    """
    world = scenario.make_world()
    pending: _Pending = {}
    steps: list[Step] = []
    try:
        for name, agent in scenario.agents.items():
            _advance(pending, name, agent(AgentContext(name)), None)
        while pending:
            name = policy.choose(sorted(pending))
            coroutine, effect = pending[name]
            result = _apply(world, effect)
            # Not sooner: a refused call leaves its agent here to be closed.
            del pending[name]
            steps.append(Step(name, effect.tool, dict(effect.args), result))
            _advance(pending, name, coroutine, result)
    finally:
        # An error leaves agents suspended mid-call.
        for coroutine, _ in pending.values():
            _close(coroutine)
    return RunResult(tuple(steps), _first_violation(scenario, world))


def _advance(
    pending: _Pending,
    name: str,
    coroutine: Coroutine[Any, Any, None],
    value: Any,
) -> None:
    """Resumes an agent until its next tool call or its end."""
    try:
        effect = coroutine.send(value)
    except StopIteration:
        return
    if not isinstance(effect, Effect):
        _close(coroutine)
        raise UnsupportedAwaitError(
            f"agent {name!r} awaited {type(effect).__name__}; an agent may "
            "only await AgentContext.call")
    pending[name] = (coroutine, effect)


def _close(coroutine: Coroutine[Any, Any, None]) -> None:
    """Closes an agent that an error left suspended.

    An agent can fail to close, for example by making a tool call in a
    finally block. Its own error is dropped: the one on its way out came
    first, and the agents after this one are still to be closed.
    """
    try:
        coroutine.close()
    except Exception:
        pass


def _apply(world: Any, effect: Effect) -> Any:
    """Calls the effect's tool on the world."""
    tool = None
    if not effect.tool.startswith("_"):
        tool = getattr(world, effect.tool, None)
    if not callable(tool):
        raise UnknownToolError(
            f"{type(world).__name__} has no tool {effect.tool!r}")
    return tool(**effect.args)


def _first_violation(scenario: Scenario, world: Any) -> str | None:
    for invariant in scenario.invariants:
        message = invariant(world)
        if message is not None:
            return message
    return None
