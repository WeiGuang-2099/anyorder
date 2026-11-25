"""Deterministic scheduler for interleaving agent tool calls.

An agent is a coroutine that reaches its world only through
AgentContext.call. Each call hands one Effect to the scheduler and suspends
the agent, so the scheduler always holds one pending call per unfinished
agent. A policy picks which of them runs next, and its picks are the only
source of ordering: the list of chosen agent names is the schedule, and the
same scenario under the same schedule gives the same trace.

A call is by value. An argument is taken when its agent awaits the call and
a result when its tool returns, as values.take takes a value: the tool, the
agent and the trace each get a copy of their own of whatever could change.
So the trace holds each value as it was at its call, and an agent and its
world share nothing but what a call carries. A value that no call carries
ends the run.

An agent can also draw, through AgentContext.draw: it asks for one sample,
and the run's sampler gives it. A draw is no scheduling point. It is made
as soon as the agent awaits it, and no other agent acts before this one
goes on. What was asked and what was given are values a call carries,
taken as a call's are, and the run keeps each draw beside its steps: a run
is fixed by its schedule and its draws together.

Nothing here reads the clock, starts a thread or uses an event loop.
"""

from __future__ import annotations

import bisect
import dataclasses
import functools
import itertools
import operator
import random
from collections.abc import Callable, Coroutine, Generator, Mapping, Sequence
from typing import Any, Protocol

from anyorder import values


class SchedulerError(Exception):
    """Base class for the scheduler's refusals."""


class ReplayDivergenceError(SchedulerError):
    """A run did not follow the schedule it was replaying."""


class UnknownToolError(SchedulerError):
    """An agent called a tool its world does not provide."""


class UnsupportedAwaitError(SchedulerError):
    """An agent awaited what its run does not give.

    That is anything but a tool call and a draw, and a draw in a run that
    has no sampler.
    """


class UnsupportedValueError(SchedulerError):
    """A value cannot be used where it was given.

    An argument or a result of a call, or what a draw asked for or was
    given, is not a value a call carries. Or a sampler was asked to draw
    from what it does not draw from. Or the name of an agent or of a tool
    is not a string and nothing more.
    """


@dataclasses.dataclass(frozen=True)
class Effect:
    """One tool call an agent is waiting to make.

    Attributes:
        tool: Name of a public method on the scenario's world: a str and
            nothing more.
        args: Keyword arguments for the call.
    """

    tool: str
    args: Mapping[str, Any]

    def __await__(self) -> Generator[Effect, Any, Any]:
        result = yield self
        return result


@dataclasses.dataclass(frozen=True)
class Draw:
    """One sample an agent is waiting for.

    Attributes:
        request: What to draw from. The scheduler does not read it: it is
            whatever the run's sampler takes, such as a distribution that
            a scripted agent writes out.
    """

    request: Any

    def __await__(self) -> Generator[Draw, Any, Any]:
        value = yield self
        return value


class AgentContext:
    """An agent's handle on its run: its name, its world's tools and draws."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def call(self, tool: str, **args: Any) -> Any:
        """Calls a tool once the scheduler lets this agent run.

        Args:
            tool: Name of a public method on the scenario's world.
            **args: Keyword arguments for the call. They are taken at this
                await: what the agent does to them afterwards is no part
                of the call.

        Returns:
            What the tool returns, or a Raised if the tool raised. A
            list or a dict in it is the agent's own copy.
        """
        return await Effect(tool, args)

    async def draw(self, request: Any) -> Any:
        """Draws one sample from the run's sampler, at once.

        No other agent acts between this await and what the agent does
        with the sample: only a tool call lets the scheduler choose.

        Args:
            request: What to draw from, as the run's sampler reads it. It
                is taken at this await, as an argument of a call is.

        Returns:
            What the sampler gave. A list or a dict in it is the agent's
            own copy.
        """
        return await Draw(request)


Agent = Callable[[AgentContext], Coroutine[Any, Any, None]]


@dataclasses.dataclass(frozen=True)
class Invariant:
    """Something that must stay true of the world, and when.

    Attributes:
        name: Says which invariant a violation broke. The message cannot:
            it changes with the schedule.
        check: Returns a message when the world violates the invariant and
            None otherwise.
        every_step: True if it must hold on the fresh world and after every
            step, False if only once every agent has finished. It has no
            default, so that each invariant says which.
    """

    name: str
    check: Callable[[Any], str | None]
    every_step: bool = dataclasses.field(kw_only=True)


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A world, the agents acting on it and what must stay true.

    Attributes:
        name: Short identifier for reports.
        make_world: Builds a fresh world for each run. A world shared
            between runs breaks replay.
        agents: Agent functions by name. A name is a str and nothing more.
            A run starts the agents in name order, whatever order they
            are listed in.
        invariants: What must stay true of the world. Each says when it
            must hold.
    """

    name: str
    make_world: Callable[[], Any]
    agents: Mapping[str, Agent]
    invariants: Sequence[Invariant]


# Defined with the values a call carries, of which it is one.
Raised = values.Raised


@dataclasses.dataclass(frozen=True)
class Step:
    """One tool call that ran.

    Attributes:
        agent: The agent the policy chose.
        tool: The tool it called.
        args: The call's keyword arguments, as they were when the agent
            awaited the call.
        result: What the tool returned, or a Raised if it raised, as it
            was when the call returned.
    """

    agent: str
    tool: str
    args: Mapping[str, Any]
    result: Any


@dataclasses.dataclass(frozen=True)
class Drawn:
    """One draw that was made.

    Attributes:
        agent: The agent that drew.
        request: What it asked to draw from, as it was when the agent
            awaited the draw.
        value: What the sampler gave, as it was when it gave it.
        step: How many steps had run when the draw was made: 0 for a draw
            that an agent made before the first call of the run. After a
            step only the agent that took it goes on, so a draw made then
            is that agent's.
    """

    agent: str
    request: Any
    value: Any
    step: int


@dataclasses.dataclass(frozen=True)
class Violation:
    """An invariant that did not hold, and where a run first saw that.

    Attributes:
        invariant: The invariant's name.
        message: What its check returned.
        step: How many steps had run when the check failed: 0 for the fresh
            world. For an invariant that must hold only at the end it is
            the length of the run, and does not point at a cause.
    """

    invariant: str
    message: str
    step: int


@dataclasses.dataclass(frozen=True)
class RunResult:
    """The trace of one run and what it broke, if anything.

    Attributes:
        steps: Every tool call, in the order it ran. A run goes on to its
            end after a violation.
        violation: The first violation the run saw, or None.
        draws: Every draw, in the order it was made. The schedule and the
            draws together fix the run: neither does alone once an agent
            draws.
    """

    steps: tuple[Step, ...]
    violation: Violation | None
    draws: tuple[Drawn, ...] = ()

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


class Sampler(Protocol):
    """Gives the samples that agents draw."""

    def sample(self, agent: str, request: Any) -> Any:
        """Returns one sample.

        Args:
            agent: The agent that draws.
            request: What it asked to draw from. A list or a dict in it is
                a copy for the sampler.

        Returns:
            The sample, a value a call carries.
        """


class SeededSampler:
    """Draws from written distributions with a seeded generator.

    A written distribution is a tuple or a list of pairs, each an outcome
    and its weight. A weight is a positive integer, so that every chance
    is an exact fraction: ((True, 1), (False, 3)) gives True one time in
    four.

    The draws come from one stream, in the order they are asked for. So
    what an agent draws depends on how many draws were made before, by
    any agent: under another schedule the same seed can give an agent
    another sample where it has seen and done the same.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng: random.Random | None = None

    def sample(self, agent: str, request: Any) -> Any:
        """Returns one outcome of a written distribution.

        Raises:
            UnsupportedValueError: The request is no written distribution.
                Nothing is drawn for it.
        """
        bounds = list(itertools.accumulate(_weights(agent, request)))
        if self._rng is None:
            # Only now: most runs make no draw. And seeded with a text: a
            # policy of the same seed chooses from the stream of the
            # number, and a draw must not be tied to a choice.
            self._rng = random.Random(f"anyorder draws {self._seed}")
        point = self._rng.randrange(bounds[-1])
        return request[bisect.bisect_right(bounds, point)][0]


class ReplaySampler:
    """Gives recorded draws back in order and refuses to leave them."""

    def __init__(self, draws: Sequence[Drawn]) -> None:
        self._draws = tuple(draws)
        self._position = 0

    def sample(self, agent: str, request: Any) -> Any:
        if self._position == len(self._draws):
            raise ReplayDivergenceError(
                f"the draws end after {self._position} but agent {agent!r} "
                "asks for one more")
        held = self._draws[self._position]
        if held.agent != agent:
            raise ReplayDivergenceError(
                f"draw {self._position + 1} names {held.agent!r}, but it is "
                f"{agent!r} that draws")
        if held.request != request:
            # Here, and not when the run is over: the agent would go on
            # with a sample of something it did not ask for.
            raise ReplayDivergenceError(
                f"draw {self._position + 1} is from "
                f"{values.shown(held.request)}, but agent {agent!r} asks to "
                f"draw from {values.shown(request)}")
        self._position += 1
        return held.value


def _weights(agent: str, request: Any) -> list[int]:
    """Returns the weights of a written distribution, in its order.

    Raises:
        UnsupportedValueError: The request is no written distribution.
    """
    if not _is_written(request):
        raise UnsupportedValueError(
            f"agent {agent!r} cannot draw from {values.shown(request)}: a "
            "written distribution is pairs of an outcome and a weight, each "
            "weight a positive integer")
    return [weight for _, weight in request]


def _is_written(request: Any) -> bool:
    """Says whether a request is a written distribution.

    A weight is an integer and nothing that passes for one: True is no
    weight.
    """
    if not (type(request) is tuple or type(request) is list) or not request:
        return False
    for pair in request:
        if not (type(pair) is tuple or type(pair) is list) or len(pair) != 2:
            return False
        if type(pair[1]) is not int or pair[1] < 1:
            return False
    return True


# By agent: its coroutine, the tool and the arguments of the call it waits
# to make, and whether none of the arguments can change. The arguments are
# the scheduler's own since the await.
_Pending = dict[
    str, tuple[Coroutine[Any, Any, None], str, dict[str, Any], bool]]


def run(
    scenario: Scenario, policy: Policy, sampler: Sampler | None = None
) -> RunResult:
    """Runs a scenario until every agent has finished.

    Args:
        scenario: What to run.
        policy: Chooses the agent for each step, including steps where only
            one agent is waiting, so a schedule has one entry per step.
        sampler: Gives the agents their draws. A scenario whose agents do
            not draw needs none.

    Returns:
        The trace, the draws and the first violation, if any. Invariants
        that must hold after every step are asked on the fresh world and
        after each step, the others once every agent has finished. Nothing
        is asked after the first violation.

    Raises:
        UnknownToolError: An agent called a tool the world lacks.
        UnsupportedAwaitError: An agent awaited something other than
            AgentContext.call and AgentContext.draw, or it drew and the
            run has no sampler.
        UnsupportedValueError: An argument or a result is not a value a
            call carries. An argument is refused when its agent awaits
            the call, before the tool runs and before the policy is
            asked again. A result is refused once its tool has run. A
            draw is refused alike: what it asks for before the sampler
            is asked, and what the sampler gave once it gave it. Or a
            name is not a string and nothing more: an agent's before
            anything is started, a tool's when its agent awaits the
            call.
    """
    names = _names(scenario)
    world = scenario.make_world()
    pending: _Pending = {}
    steps: list[Step] = []
    draws: list[Drawn] = []
    draw = functools.partial(_drawn, sampler, draws, steps)
    try:
        for name in names:
            agent = scenario.agents[name]
            _advance(pending, name, agent(AgentContext(name)), None, draw)
        violation = _first_violation(scenario, world, 0, every_step=True)
        while pending:
            name = policy.choose(sorted(pending))
            coroutine, tool, args, fixed = pending[name]
            # A copy of its own for the tool: what it does to a list it is
            # given must not show in the trace.
            returned = _apply(
                world, tool, args if fixed else values.take(args))
            result = _taken_result(name, tool, returned)
            # And one for the agent, which may change what it is given. A
            # value that was handed back as it is cannot change.
            handed = (
                returned if result is returned else values.take(result))
            steps.append(Step(name, tool, args, result))
            if violation is None:
                violation = _first_violation(
                    scenario, world, len(steps), every_step=True)
            # Not sooner: an error raised above leaves the agent here to be
            # closed.
            del pending[name]
            _advance(pending, name, coroutine, handed, draw)
    finally:
        # An error leaves agents suspended mid-call.
        for coroutine, *_ in pending.values():
            _close(coroutine)
    if violation is None:
        violation = _first_violation(
            scenario, world, len(steps), every_step=False)
    return RunResult(tuple(steps), violation, tuple(draws))


def _names(scenario: Scenario) -> list[str]:
    """Returns the names of a scenario's agents in the order they are started.

    In name order, as a policy is shown the waiting agents: what an agent
    does before its first call must not depend on the order a scenario
    happens to list its agents in.

    Raises:
        UnsupportedValueError: A name is not a string and nothing more.
    """
    for name in scenario.agents:
        if type(name) is not str:
            raise UnsupportedValueError(
                f"an agent cannot be named {values.shown(name)}, of type "
                f"{type(name).__name__}: an agent's name is a string")
    return sorted(scenario.agents)


def _advance(
    pending: _Pending,
    name: str,
    coroutine: Coroutine[Any, Any, None],
    value: Any,
    draw: Callable[[str, Any], Any],
) -> None:
    """Resumes an agent until its next tool call or its end.

    A draw on the way is made at once, and the agent goes on with what it
    drew: no policy is asked between.

    Args:
        pending: Where the agent's next call is put.
        name: The agent.
        coroutine: The agent as it runs.
        value: What its last await gives it.
        draw: Makes a draw for an agent, given what it asked to draw from.
    """
    while True:
        try:
            awaited = coroutine.send(value)
        except StopIteration:
            return
        if isinstance(awaited, Effect):
            break
        if not isinstance(awaited, Draw):
            _close(coroutine)
            raise UnsupportedAwaitError(
                f"agent {name!r} awaited {type(awaited).__name__}; an agent "
                "may only await AgentContext.call and AgentContext.draw")
        try:
            value = draw(name, awaited.request)
        except BaseException:
            # As for a call below: nothing else closes this agent.
            _close(coroutine)
            raise
    try:
        tool = awaited.tool
        if type(tool) is not str:
            # More than a string can say what it likes of itself: that it
            # is equal to another name, or that it starts with no
            # underscore.
            raise UnsupportedValueError(
                f"agent {name!r} cannot call a tool named "
                f"{values.shown(tool)}, of type {type(tool).__name__}: a "
                "tool's name is a string")
        args, fixed = _taken_args(name, tool, awaited.args)
    except BaseException:
        # Whatever was raised, an interrupt too: the agent is not among
        # the pending ones that run closes.
        _close(coroutine)
        raise
    pending[name] = (coroutine, tool, args, fixed)


def _drawn(
    sampler: Sampler | None,
    draws: list[Drawn],
    steps: Sequence[Step],
    name: str,
    given: Any,
) -> Any:
    """Makes one draw for an agent and adds it to the draws of its run.

    Args:
        sampler: The run's sampler, if it has one.
        draws: The draws of the run so far.
        steps: The steps of the run so far.
        name: The agent that draws.
        given: What it asked to draw from.

    Returns:
        The agent's own copy of what the sampler gave.

    Raises:
        UnsupportedAwaitError: The run has no sampler.
        UnsupportedValueError: What the agent asked for or what the
            sampler gave is not a value a call carries. The sampler has
            not been asked in the first case and has in the second.
    """
    if sampler is None:
        raise UnsupportedAwaitError(
            f"agent {name!r} awaited a draw, and the run has no sampler")
    try:
        request = values.take(given)
    except values.NotTakenError as error:
        raise UnsupportedValueError(
            f"agent {name!r} cannot draw: {error}") from None
    # A copy of its own for the sampler, as a tool gets one of its
    # arguments. A value that was handed back as it is cannot change.
    returned = sampler.sample(
        name, given if request is given else values.take(request))
    try:
        value = values.take(returned)
    except values.NotTakenError as error:
        raise UnsupportedValueError(
            f"a draw was made for agent {name!r} and its value is refused: "
            f"{error}") from None
    draws.append(Drawn(name, request, value, len(steps)))
    # And one for the agent, which may change what it is given.
    return returned if value is returned else values.take(value)


def _taken_args(
    name: str, tool: str, given: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Returns a call's arguments as they are when its agent awaits it.

    Now, and not when the call runs: until then the agent's own lists stay
    in reach of whoever else holds them.

    Returns:
        The arguments in a dict of the scheduler's own, and whether none of
        them can change. Then the tool may be given these very values.

    Raises:
        UnsupportedValueError: An argument is not a value a call carries,
            or the arguments are no mapping.
    """
    if type(given) is not dict:
        # An agent can await an Effect it built itself, with any mapping.
        # What is looked at below must be a dict and nothing that can say
        # otherwise. And a list of pairs is no mapping, though dict would
        # make one of it.
        if not isinstance(given, Mapping):
            raise UnsupportedValueError(
                f"agent {name!r} cannot call {tool!r}: its arguments are of "
                f"type {type(given).__name__}, not a mapping")
        given = dict(given)
    if not given:
        # Most reads: nothing to take.
        return {}, True
    try:
        args = values.take(given)
    except values.NotTakenError as error:
        raise UnsupportedValueError(
            f"agent {name!r} cannot call {tool!r}: {error}") from None
    # A value that cannot change is handed back as it is.
    fixed = all(map(operator.is_, args.values(), given.values()))
    return args, fixed


def _taken_result(name: str, tool: str, result: Any) -> Any:
    """Returns a result as it is when its tool returns.

    Raises:
        UnsupportedValueError: The result is not a value a call carries.
            Its tool has run all the same.
    """
    try:
        return values.take(result)
    except values.NotTakenError as error:
        raise UnsupportedValueError(
            f"{tool!r} ran for agent {name!r} and its result is refused: "
            f"{error}") from None


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


def _apply(world: Any, name: str, args: Mapping[str, Any]) -> Any:
    """Calls a tool on the world.

    Returns:
        What the tool returned, or a Raised if it raised.
    """
    tool = None
    if not name.startswith("_"):
        tool = getattr(world, name, None)
    if not callable(tool):
        raise UnknownToolError(
            f"{type(world).__name__} has no tool {name!r}")
    try:
        return tool(**args)
    except SchedulerError:
        # A refusal is never the agent's to handle, least of all a replay
        # that left its schedule.
        raise
    except Exception as error:
        return Raised(_plain(type(error).__name__), _plain(str(error)))


def _plain(text: str) -> str:
    """Returns a text as a str and nothing more.

    An error can give its name or its text as an instance of a subclass of
    str, and no call carries one.
    """
    return text if type(text) is str else str.__str__(text)


def _first_violation(
    scenario: Scenario, world: Any, step: int, *, every_step: bool
) -> Violation | None:
    """Asks the invariants of one kind, in the order listed.

    Args:
        scenario: Whose invariants to ask.
        world: The world as it is now.
        step: How many steps have run.
        every_step: Which kind to ask.

    Returns:
        The first one that does not hold, or None.
    """
    for invariant in scenario.invariants:
        if invariant.every_step != every_step:
            continue
        message = invariant.check(world)
        if message is not None:
            return Violation(invariant.name, message, step)
    return None
