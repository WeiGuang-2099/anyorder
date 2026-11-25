"""Where a run stands, as a value that a search can look up by.

An agent's local history is what the agent has done and been given so far:
its own calls with their arguments and results and its own draws with what
they asked for and were given, in the order they came. An agent reaches
nothing but through its calls and its draws, so an agent that has one local
history in two runs goes on alike in both.

A configuration is the state of the world, as the scenario's snapshot gives
it, and the local history of the agents that wait to make a call and,
while a sampler is asked, of the one that draws.
Two runs that stand at one configuration go on alike under the same
choices and the same draws: the world answers alike, and each agent does
what its history has it do. So a search that has gone on from a
configuration once need not go on from it again. An agent that has
finished is left out: it has no part in what comes. So is one that is not
started yet, at a draw before the first call: it has done nothing, in
every run.

A value in a history is held by its key, as values.key gives it, so that 1
is not taken for True.

A configuration does not say whether an invariant has been broken on the
way. A world can pass through a state that breaks one and come back, and a
run that did has failed. Whoever looks runs up by their configuration asks
the moment for the violation as well.

A table sampler gives one draw for each agent and local history. Under it
an agent draws the same wherever it has seen and done the same, so a draw
that was made once serves every schedule that comes to that place: all the
schedules of a scenario need no more draws than there are places.

Outside the scheduler: all of this is read off a Moment, or for the places
of a run off the run.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Hashable, Sequence
from typing import Any

from anyorder import scheduler
from anyorder import values

# One event of a local history: a call or a draw, with the keys of its
# values.
Event = tuple[Hashable, ...]


@dataclasses.dataclass(frozen=True)
class Configuration:
    """The state of a world and what each waiting agent has seen and done.

    It can be hashed. Two runs that stand at equal configurations go on
    alike.

    Attributes:
        state: The state of the world, as the scenario's snapshot gives it.
        histories: Each agent that waits to make a call, and at a draw the
            one that draws, with its local history, in name order.
    """

    state: Hashable
    histories: tuple[tuple[str, tuple[Event, ...]], ...]


def local_history(
    moment: scheduler.Moment, agent: str
) -> tuple[Event, ...]:
    """Returns what an agent has done and been given so far, in order.

    Args:
        moment: A run as it stands.
        agent: The name of one of its agents.

    Returns:
        A ("call", tool, arguments, result) for each call the agent made
        and a ("draw", request, value) for each draw, with the keys of the
        values. At a draw, without the draw that is being made.
    """
    histories, _ = _read(moment.steps, moment.draws, places=False)
    return tuple(histories.get(agent, ()))


def at(moment: scheduler.Moment) -> Configuration:
    """Returns the configuration a run stands at.

    At a choice it holds the agents that wait. Asked by a sampler, it also
    holds the agent that draws, which waits for no call yet and is the one
    that goes on. What that agent asks to draw from is not in it: an agent
    asks by what it has seen and done.

    Raises:
        MissingSnapshotError: The scenario has no snapshot.
        TypeError: The snapshot gave a value that cannot be hashed.
    """
    histories, _ = _read(moment.steps, moment.draws, places=False)
    names = moment.waiting
    if moment.drawing is not None:
        names = sorted([*names, moment.drawing])
    return Configuration(moment.state(), tuple(
        (name, tuple(histories.get(name, ()))) for name in names))


def places(
    run: scheduler.RunResult,
) -> list[tuple[Hashable, scheduler.Drawn]]:
    """Returns each draw of a run with the place it was made at.

    A place is the agent that drew, its local history before the draw and
    the key of what it asked to draw from. It is where a table sampler
    holds a draw, and two draws at one place are draws of one agent that
    had seen and done the same.
    """
    _, placed = _read(run.steps, run.draws, places=True)
    return placed


class TableSampler:
    """Draws from written distributions, one draw for each place in a table.

    A place is an agent, its local history and what it asks to draw from.
    The draw at a place is made once, with a generator seeded with the
    table's seed and the place, and is held: whoever asks at that place
    again is given the same, under every schedule and in every run, and
    whichever places were asked for before. So under one table the agents
    of a scenario are a program with no chance left in it, and every
    schedule of it can be run. One table is one replicate of the draws of
    a scenario.

    Attributes:
        held: The draws made so far, by place. Their number is the number
            of draws that all the runs so far needed between them.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self.held: dict[Hashable, Any] = {}

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        """Returns the draw of the place the agent stands at.

        Raises:
            UnsupportedValueError: The request is no written distribution.
        """
        place = (
            agent, local_history(moment, agent), values.key(request))
        if place not in self.held:
            # Written with ascii: how Python shows a character outside
            # it depends on the version.
            rng = random.Random(
                f"anyorder table {self._seed} {ascii(place)}")
            self.held[place] = scheduler.pick(agent, request, rng)
        return self.held[place]


def _read(
    steps: Sequence[scheduler.Step],
    draws: Sequence[scheduler.Drawn],
    *,
    places: bool,
) -> tuple[
    dict[str, list[Event]], list[tuple[Hashable, scheduler.Drawn]]
]:
    """Reads the steps and the draws of a run as the agents came to them.

    A run keeps its steps in one row and its draws in another, each draw
    with the number of steps before it. An agent's history has both in the
    order they came to the agent, so the two rows are read as one.

    Args:
        steps: The steps of a run, or of a run so far.
        draws: Its draws.
        places: Whether the places of the draws are wanted. A place holds
            a copy of its agent's history up to the draw, so the places
            of a run cost the square of its draws, where the histories
            alone cost their number.

    Returns:
        The local history of every agent that has one, and each draw with
        the place it was made at, or no draw if the places are not
        wanted.
    """
    histories: dict[str, list[Event]] = {}
    placed: list[tuple[Hashable, scheduler.Drawn]] = []

    def add(draw: scheduler.Drawn) -> None:
        history = histories.setdefault(draw.agent, [])
        request = values.key(draw.request)
        if places:
            placed.append(((draw.agent, tuple(history), request), draw))
        history.append(("draw", request, values.key(draw.value)))

    position = 0
    for before, step in enumerate(steps):
        while position < len(draws) and draws[position].step <= before:
            add(draws[position])
            position += 1
        histories.setdefault(step.agent, []).append((
            "call", step.tool, values.key(step.args),
            values.key(step.result)))
    for draw in draws[position:]:
        add(draw)
    return histories, placed
