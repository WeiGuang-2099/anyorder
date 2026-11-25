"""Where a run stands, as a value that a search can look up by.

An agent's local history is what the agent has done and been given so far:
its own calls with their arguments and results and its own draws with what
they asked for and were given, in the order they came. An agent reaches
nothing but through its calls and its draws, so two agents of one scenario
with one local history go on alike.

A configuration is the state of the world, as the scenario's snapshot gives
it, and the local history of every agent that still waits to make a call.
Two runs that stand at one configuration go on alike under the same
choices and the same draws: the world answers alike, and each agent does
what its history has it do. So a search that has gone on from a
configuration once need not go on from it again. An agent that has
finished is left out: it has no part in what comes.

A value in a history is held by its key, as values.key gives it, so that 1
is not taken for True.

A configuration does not say whether an invariant has been broken on the
way. A world can pass through a state that breaks one and come back, and a
run that did has failed. Whoever looks runs up by their configuration asks
the moment for the violation as well.

Outside the scheduler: all of this is read off a Moment.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable

from anyorder import scheduler
from anyorder import values

# One event of a local history: a call or a draw, with the keys of its
# values.
Event = tuple[Hashable, ...]


@dataclasses.dataclass(frozen=True)
class Configuration:
    """The state of a world and what each waiting agent has seen and done.

    It can be hashed, and is equal for two runs that go on alike.

    Attributes:
        state: The state of the world, as the scenario's snapshot gives it.
        histories: Each agent that waits to make a call with its local
            history, in name order.
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
    return tuple(_histories(moment).get(agent, ()))


def at(moment: scheduler.Moment) -> Configuration:
    """Returns the configuration a run stands at.

    Raises:
        MissingSnapshotError: The scenario has no snapshot.
        TypeError: The snapshot gave a value that cannot be hashed.
    """
    histories = _histories(moment)
    return Configuration(moment.state(), tuple(
        (name, tuple(histories.get(name, ()))) for name in moment.waiting))


def _histories(moment: scheduler.Moment) -> dict[str, list[Event]]:
    """Returns the local history of every agent that has one.

    A run keeps its steps in one row and its draws in another, each draw
    with the number of steps before it. An agent's history has both in the
    order they came to the agent, so the two rows are read as one.
    """
    histories: dict[str, list[Event]] = {}
    draws = moment.draws
    position = 0
    for before, step in enumerate(moment.steps):
        while position < len(draws) and draws[position].step <= before:
            _add_draw(histories, draws[position])
            position += 1
        histories.setdefault(step.agent, []).append((
            "call", step.tool, values.key(step.args),
            values.key(step.result)))
    for draw in draws[position:]:
        _add_draw(histories, draw)
    return histories


def _add_draw(
    histories: dict[str, list[Event]], draw: scheduler.Drawn
) -> None:
    histories.setdefault(draw.agent, []).append(
        ("draw", values.key(draw.request), values.key(draw.value)))
