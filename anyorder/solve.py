"""Works out exactly how often a small scenario fails under a scheduler.

A scenario whose agents draw from written distributions fails with a chance
that depends on who chooses the next agent. This module gives that chance
as an exact fraction for the schedulers a result is held against:

* uniform: a uniform choice among the waiting agents at every step, which
  is what a seeded search does.
* serial: the agents one after another, in a given order.
* round_robin: one call each in turn, in name order.
* adaptive: the scheduler that does the most harm of all that see
  everything that has happened and every waiting call, and no draw that is
  still to come. With it comes one failing run.
* clairvoyant: the scheduler that also knows every draw to come. Its
  value is the chance, over the tables of draws, that some schedule
  fails.

A failure is a broken invariant of the scenario, as it is for every search:
a run fails if it breaks what must hold after every step, also when the
world is whole again later, or what must hold at the end.

An agent cannot be copied in the middle of a run, so a point of a run is
reached by running again: one run for each choice and for each outcome of
a draw, its start replayed from the steps and the draws of the run before.
A scenario that is not the same twice is therefore refused here as an
exhaustive search refuses it, where the replay sees it: an agent that
waits to make another call, asks to draw from something else, or is
answered otherwise. Every run goes to its end and every choice is tried,
also once a schedule has failed, so whatever ends a run ends the search.

Two runs that stand at one configuration go on alike, so what follows a
configuration is worked out once and looked up after that. A run is
looked up by the state of its world, written as values.key writes a
value, by the local history of each agent that still acts, and by
whether an invariant has broken on the way: a run that broke one has
failed whatever follows. This rests on the scenario's snapshot. A
snapshot that leaves out something a later call or check goes by makes
two runs one that are not, and the value is then not the scenario's,
with no error. Solving again with lookup=False, which needs no snapshot
and makes every run, is the check.

Nothing here is sampled, and the work grows with the number of schedules
times the number of outcomes of the draws, or of configurations where
runs are looked up: this is for scenarios small enough to be the ground
truth of a search.

Outside the scheduler: all of this is done through a policy and a sampler.
"""

from __future__ import annotations

import dataclasses
import fractions
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

from anyorder import configuration
from anyorder import explore
from anyorder import scheduler
from anyorder import values

Fraction = fractions.Fraction

# How many choices and draws a run is followed for, unless the caller says.
LIMIT = 200


class TooLongError(scheduler.SchedulerError):
    """A run went past the number of choices and draws it is followed for.

    A value is exact only if every run ends, so a run that is cut has
    none.
    """


@dataclasses.dataclass(frozen=True)
class _Edge:
    """Where a run stopped for want of a decision.

    Attributes:
        steps: The steps of the run so far.
        draws: Its draws so far.
        waiting: The agents that wait, in name order, where the run stopped
            at a choice.
        agent: The agent that draws, where it stopped at a draw.
        request: What that agent asked to draw from.
        key: What the run is looked up by where it stopped, or None if
            runs are not looked up.
    """

    steps: tuple[scheduler.Step, ...]
    draws: tuple[scheduler.Drawn, ...]
    waiting: tuple[str, ...] = ()
    agent: str | None = None
    request: Any = None
    key: Hashable | None = None


class _Stop(Exception):
    """Ends a run at an edge. Never leaves this module."""

    def __init__(self, edge: _Edge) -> None:
        super().__init__()
        self.edge = edge


class _Hole(Exception):
    """Ends a run at a place its table has no draw for.

    Attributes:
        place: The place: the agent, its local history and the key of
            what it asks to draw from, as a table sampler holds one.
        agent: The agent that draws there.
        request: What it asks to draw from.
    """

    def __init__(self, place: Hashable, agent: str, request: Any) -> None:
        super().__init__()
        self.place = place
        self.agent = agent
        self.request = request


# One draw for each place: a table of draws, as far as it has been needed.
_Table = dict[Hashable, Any]


# A decision: the agent that is chosen, or the outcome that is drawn.
_Decision = tuple[str, Any]
_CHOOSE = "choose"
_DRAW = "draw"


class _Guide:
    """A policy and a sampler that take a run to the edge after its start.

    They follow the steps and the draws of the start as a replay does,
    make one more decision if they are given one, and stop the run where
    it asks for the next. Given a table, a draw past the start is no
    decision: it is the draw the table holds at that place, and the run
    stops only at a place the table lacks.
    """

    def __init__(
        self,
        scenario: scheduler.Scenario,
        steps: Sequence[scheduler.Step],
        draws: Sequence[scheduler.Drawn],
        then: _Decision | None,
        limit: int,
        table: _Table | None,
        lookup: bool,
    ) -> None:
        self._scenario = scenario
        self._steps = steps
        self._draws = draws
        self._policy = scheduler.ReplayPolicy(steps)
        self._sampler = scheduler.ReplaySampler(draws)
        self.then = then
        self._limit = limit
        self._table = table
        self._lookup = lookup
        self._asked = 0
        self._chosen = 0
        self._drawn = 0

    def choose(self, moment: scheduler.Moment) -> str:
        self._count()
        if self._chosen < len(self._steps):
            self._chosen += 1
            return self._policy.choose(moment)
        if self._drawn < len(self._draws):
            raise self._left("it waits for a choice where it drew before")
        return self._decide(_CHOOSE, moment)

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        self._count()
        if self._drawn < len(self._draws):
            held = self._draws[self._drawn]
            if held.step != self._chosen:
                raise self._left(
                    f"draw {self._drawn + 1} was made after {self._chosen} "
                    f"steps, not after {held.step}")
            self._drawn += 1
            return self._sampler.sample(agent, request, moment)
        if self._chosen < len(self._steps):
            raise self._left("it draws where it made a call before")
        # Here, so that a request the package refuses is refused with the
        # agents closed, as a sampler refuses it.
        scheduler.weights(agent, request)
        if self._table is None:
            return self._decide(_DRAW, moment, agent, request)
        place = (
            agent, configuration.local_history(moment, agent),
            values.key(request))
        if place not in self._table:
            raise _Hole(place, agent, request)
        return self._table[place]

    def _count(self) -> None:
        """Counts one more choice or draw that the run asks for."""
        self._asked += 1
        if self._asked > self._limit:
            raise TooLongError(
                f"a run of {self._scenario.name!r} went past {self._limit} "
                "choices and draws, and the solver follows no run further: "
                "every run must end")

    def _decide(
        self,
        kind: str,
        moment: scheduler.Moment,
        agent: str | None = None,
        request: Any = None,
    ) -> Any:
        """Makes the one decision after the start, or stops the run."""
        if self.then is None:
            raise _Stop(_Edge(
                moment.steps, moment.draws,
                waiting=() if kind == _DRAW else tuple(moment.waiting),
                agent=agent, request=request, key=self._key(moment)))
        asked, decision = self.then
        if asked != kind:
            raise self._left(
                "it draws where it waited for a choice before"
                if kind == _DRAW else
                "it waits for a choice where it drew before")
        self.then = None
        if kind == _CHOOSE and decision not in moment.waiting:
            # Not left to the scheduler, which looks the name up and says
            # nothing of a start.
            raise self._left(
                f"agent {decision!r} does not wait where it waited before")
        return decision

    def _key(self, moment: scheduler.Moment) -> Hashable | None:
        """Returns what the run is looked up by where it stands.

        Raises:
            MissingSnapshotError: The scenario has no snapshot.
            UnsupportedValueError: The snapshot gave a state that
                values.key does not write.
        """
        if not self._lookup:
            return None
        where = configuration.at(moment)
        try:
            # Written as a trace file would write it: a state is held as
            # Python compares, which takes True, 1 and 1.0 for equal, and
            # an agent that is handed one of them can go on otherwise.
            state = values.key(where.state)
        except values.NotTakenError as error:
            raise scheduler.UnsupportedValueError(
                f"the solver cannot look the runs of {self._scenario.name!r} "
                f"up by the state its snapshot gives: {error}. Give a state "
                "of values a call carries, or solve with lookup=False"
            ) from None
        return state, where.histories, moment.violation is not None

    def _left(self, how: str) -> scheduler.ReplayDivergenceError:
        return scheduler.ReplayDivergenceError(
            f"a run of {self._scenario.name!r} left the start it was given "
            f"to follow: {how}")


# Which of the waiting agents are tried at a choice, and what the values of
# going on with each make together.
_Among = Callable[[_Edge], Sequence[str]]
_Merge = Callable[[Sequence[Fraction]], Fraction]


class _Search:
    """Tries every decision of a scenario's runs, one run for each.

    Attributes:
        failure: The first failing run met so far, if any.
    """

    def __init__(
        self, scenario: scheduler.Scenario, limit: int, lookup: bool
    ) -> None:
        self._scenario = scenario
        self._limit = limit
        self._lookup = lookup
        self._known: dict[Hashable, Fraction] = {}
        self.failure: scheduler.RunResult | None = None

    def value(
        self,
        among: _Among,
        merge: _Merge,
        steps: Sequence[scheduler.Step] = (),
        draws: Sequence[scheduler.Drawn] = (),
        then: _Decision | None = None,
    ) -> Fraction:
        """Returns the chance that a run fails from a point on.

        Args:
            among: Gives the agents to try at a choice.
            merge: Makes one value of theirs.
            steps: The steps of the run up to the point.
            draws: Its draws up to the point.
            then: The decision to make at the point, if it is not the
                start of a run.
        """
        reached = self._advance(steps, draws, then)
        if isinstance(reached, scheduler.RunResult):
            if reached.violation is None:
                return Fraction(0)
            if self.failure is None:
                self.failure = reached
            return Fraction(1)
        if reached.key in self._known:
            return self._known[reached.key]
        if reached.agent is None:
            found = merge([
                self.value(
                    among, merge, reached.steps, reached.draws,
                    (_CHOOSE, name))
                for name in among(reached)])
        else:
            found = Fraction(0)
            for outcome, chance in _chances(reached.agent, reached.request):
                found += chance * self.value(
                    among, merge, reached.steps, reached.draws,
                    (_DRAW, outcome))
        if reached.key is not None:
            self._known[reached.key] = found
        return found

    def some_schedule_fails(
        self,
        table: _Table,
        known: dict[Hashable, bool],
        steps: Sequence[scheduler.Step] = (),
        draws: Sequence[scheduler.Drawn] = (),
        then: _Decision | None = None,
    ) -> bool:
        """Says whether a schedule fails from a point on, under a table.

        Under a table the agents are a program with no chance left in
        it: every draw is given, and only the choices are tried. Each of
        them is, also once a schedule has failed: a run that would end
        this search for any other reason must end it whichever schedule
        is tried first.

        Args:
            table: The draws, by place.
            known: What has been worked out under this table so far, by
                what a run is looked up by. It is added to.
            steps: The steps of the run up to the point.
            draws: Its draws up to the point.
            then: The choice to make at the point, if it is not the start
                of a run.

        Raises:
            _Hole: A run came to a place the table has no draw for.
        """
        reached = self._advance(steps, draws, then, table)
        if isinstance(reached, scheduler.RunResult):
            return reached.violation is not None
        if reached.key in known:
            return known[reached.key]
        found = False
        for name in reached.waiting:
            found = self.some_schedule_fails(
                table, known, reached.steps, reached.draws,
                (_CHOOSE, name)) or found
        if reached.key is not None:
            known[reached.key] = found
        return found

    def _advance(
        self,
        steps: Sequence[scheduler.Step],
        draws: Sequence[scheduler.Drawn],
        then: _Decision | None,
        table: _Table | None = None,
    ) -> _Edge | scheduler.RunResult:
        """Runs a start and one decision, to the next edge or to the end.

        Raises:
            ReplayDivergenceError: The run did not make its start again.
            _Hole: The run is under a table and came to a place the table
                has no draw for.
        """
        scenario = self._scenario
        guide = _Guide(
            scenario, steps, draws, then, self._limit, table, self._lookup)
        try:
            reached: _Edge | scheduler.RunResult = scheduler.run(
                scenario, guide, guide)
        except _Stop as stop:
            reached = stop.edge
        else:
            if guide.then is not None:
                raise scheduler.ReplayDivergenceError(
                    f"a run of {scenario.name!r} ended at the start it was "
                    "given to follow, where it went on before")
        # A replay holds each call and each request against the start. The
        # results are held here, once the run is back.
        if (tuple(reached.steps[:len(steps)]) != tuple(steps)
                or tuple(reached.draws[:len(draws)]) != tuple(draws)):
            raise scheduler.ReplayDivergenceError(
                f"a run of {scenario.name!r} replayed its start to a "
                "different trace")
        return reached


def _chances(agent: str, request: Any) -> list[tuple[Any, Fraction]]:
    """Returns each outcome of a written distribution with its chance."""
    weights = scheduler.weights(agent, request)
    total = sum(weights)
    return [
        (pair[0], Fraction(weight, total))
        for pair, weight in zip(request, weights)]


def _mean(numbers: Sequence[Fraction]) -> Fraction:
    return sum(numbers, Fraction(0)) / len(numbers)


def _only(numbers: Sequence[Fraction]) -> Fraction:
    (number,) = numbers
    return number


def uniform(
    scenario: scheduler.Scenario,
    *,
    limit: int = LIMIT,
    lookup: bool = True,
) -> Fraction:
    """Returns the chance that a run fails under uniform random scheduling.

    At every step each waiting agent is chosen with the same chance, as
    scheduler.RandomPolicy chooses, and every draw comes by its weights.

    Args:
        scenario: What to solve. Every draw of its agents must be from a
            written distribution, and every run must end.
        limit: How many choices and draws a run is followed for.
        lookup: Whether runs are looked up by where they stand. The value
            is then exact only if the scenario's snapshot says everything
            of the world that a later call or check goes by. Without it
            every run is made, and no snapshot is needed.

    Raises:
        TooLongError: A run went past the limit.
        ReplayDivergenceError: A run did not make its start again, which
            means the scenario is not deterministic.
        UnsupportedValueError: An agent asked to draw from something that
            is no written distribution. Or runs are looked up and the
            snapshot gave a state that values.key does not write.
        MissingSnapshotError: Runs are looked up and the scenario has no
            snapshot.
        TypeError: Runs are looked up and the snapshot gave a value that
            cannot be hashed.
    """
    search = _Search(scenario, limit, lookup)
    return search.value(lambda edge: edge.waiting, _mean)


def serial(
    scenario: scheduler.Scenario,
    order: Iterable[str],
    *,
    limit: int = LIMIT,
) -> Fraction:
    """Returns the chance that a run fails with the agents one after another.

    Each agent makes all its calls before the next makes one. The draws
    come by their weights: this is the failure rate of one serial
    scheduler over fresh draws, and not the oracle's serial orders, which
    are held against one run.

    Args:
        scenario: What to solve, as uniform takes it.
        order: The names of the agents, each once, first to last.
        limit: How many choices and draws a run is followed for.

    Raises:
        ValueError: The order does not name each agent once, holds
            something that is not a string and nothing more, or is a set,
            which has no first and no last.
        TypeError: The order cannot be walked.
        TooLongError: A run went past the limit.
        ReplayDivergenceError: A run did not make its start again.
        UnsupportedValueError: An agent asked to draw from something that
            is no written distribution.
    """
    names = sorted(scenario.agents)
    if isinstance(order, (set, frozenset)):
        # It would be walked as this process happens to hash its names.
        raise ValueError(
            f"a serial order of {scenario.name!r} is walked first to last, "
            "and a set has no first: give a list or a tuple")
    # Walked once, and looked at before it is put in order: an order need
    # not be a list, and Python cannot sort a number beside a name.
    turns = tuple(order)
    if (any(type(name) is not str for name in turns)
            or sorted(turns) != names):
        if not names:
            raise ValueError(
                f"a serial order of {scenario.name!r} names nobody: the "
                "scenario has no agents")
        raise ValueError(
            f"a serial order of {scenario.name!r} names each of "
            f"{', '.join(names)} once")

    def first(edge: _Edge) -> Sequence[str]:
        return [next(name for name in turns if name in edge.waiting)]

    return _Search(scenario, limit, lookup=False).value(first, _only)


def round_robin(
    scenario: scheduler.Scenario, *, limit: int = LIMIT
) -> Fraction:
    """Returns the chance that a run fails with one call each in turn.

    The turn goes round in name order, from the first name, and passes
    over an agent that does not wait. So the value goes by the names: the
    same agents under other names take their turns in another order.

    Args:
        scenario: What to solve, as uniform takes it.
        limit: How many choices and draws a run is followed for.

    Raises:
        TooLongError: A run went past the limit.
        ReplayDivergenceError: A run did not make its start again.
        UnsupportedValueError: An agent asked to draw from something that
            is no written distribution.
    """
    names = sorted(scenario.agents)

    def next_in_turn(edge: _Edge) -> Sequence[str]:
        after = names.index(edge.steps[-1].agent) + 1 if edge.steps else 0
        turn = names[after:] + names[:after]
        return [next(name for name in turn if name in edge.waiting)]

    search = _Search(scenario, limit, lookup=False)
    return search.value(next_in_turn, _only)


@dataclasses.dataclass(frozen=True)
class Adaptive:
    """What the strongest scheduler does to a scenario.

    Attributes:
        value: The largest chance of a failing run that a scheduler
            reaches.
        failure: One failing run, or None if the value is 0. It is the
            first that the search met, with the waiting agents tried in
            name order and the outcomes of a draw as they are written, and
            it is no more than one way to fail: the strongest scheduler
            need not take it.
    """

    value: Fraction
    failure: scheduler.RunResult | None


def adaptive(
    scenario: scheduler.Scenario,
    *,
    limit: int = LIMIT,
    lookup: bool = True,
) -> Adaptive:
    """Returns the largest chance of a failing run under any scheduler.

    The scheduler chooses each next agent knowing all that has happened
    and every waiting call, and no draw that is still to come: at a
    choice it takes the agent with the largest chance from there on, and
    a draw comes by its weights. No scheduler that goes by the run so far
    does more harm, and one that also knew the draws to come could.

    A failing run is replayed from its steps and its draws before it is
    returned, so a reported failure is one that can be reproduced.

    Args:
        scenario: What to solve, as uniform takes it.
        limit: How many choices and draws a run is followed for.
        lookup: Whether runs are looked up by where they stand, as for
            uniform.

    Raises:
        TooLongError: A run went past the limit.
        ReplayDivergenceError: A run did not make its start again, or the
            failing run did not replay to the same trace.
        UnsupportedValueError: An agent asked to draw from something that
            is no written distribution. Or runs are looked up and the
            snapshot gave a state that values.key does not write.
        MissingSnapshotError: Runs are looked up and the scenario has no
            snapshot.
        TypeError: Runs are looked up and the snapshot gave a value that
            cannot be hashed.
    """
    search = _Search(scenario, limit, lookup)
    value = search.value(lambda edge: edge.waiting, max)
    failure = search.failure
    if failure is not None and explore.replay(
            scenario, failure.steps, failure.draws) != failure:
        raise scheduler.ReplayDivergenceError(
            f"a failing run of {scenario.name!r} replayed to a different "
            "trace")
    return Adaptive(value, failure)


def clairvoyant(
    scenario: scheduler.Scenario,
    *,
    limit: int = LIMIT,
    lookup: bool = True,
) -> Fraction:
    """Returns the chance that some schedule fails, over the tables of draws.

    A table holds one draw for each place: an agent, what it has seen and
    done, and what it asks to draw from. Under a table the scenario is a
    program with no chance left in it, and a scheduler that knows the
    table breaks it if any schedule does. No scheduler does more harm, and
    one that must choose before a draw is made cannot count on as much.

    A table is filled in as the search comes to its places: at a place it
    lacks, the tables that give each outcome there are taken apart, each
    with the outcome's chance. Under a table every schedule is tried, also
    once one has failed, so whatever ends a run under one schedule ends
    this search as it ends the uniform and the adaptive one, once the
    tables before that one have been gone through. So the work can grow
    with the product of the outcomes over the places, and nothing is
    promised of its cost.

    Args:
        scenario: What to solve, as uniform takes it.
        limit: How many choices and draws a run is followed for.
        lookup: Whether runs are looked up by where they stand, as for
            uniform. What is looked up is kept for one table.

    Raises:
        TooLongError: A run went past the limit.
        ReplayDivergenceError: A run did not make its start again.
        UnsupportedValueError: An agent asked to draw from something that
            is no written distribution. Or runs are looked up and the
            snapshot gave a state that values.key does not write.
        MissingSnapshotError: Runs are looked up and the scenario has no
            snapshot.
        TypeError: Runs are looked up and the snapshot gave a value that
            cannot be hashed.
    """
    return _over_tables(_Search(scenario, limit, lookup), {})


def _over_tables(search: _Search, table: _Table) -> Fraction:
    """Returns the chance of a failing schedule over the tables that hold
    the draws of one table, wherever it has any."""
    try:
        return Fraction(search.some_schedule_fails(table, {}))
    except _Hole as hole:
        total = Fraction(0)
        for outcome, chance in _chances(hole.agent, hole.request):
            total += chance * _over_tables(
                search, {**table, hole.place: outcome})
        return total
