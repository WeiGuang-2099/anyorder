"""Checks small things that are easy to change with no test noticing.

One test each. They were found by changing the code a line at a time and
looking at what still passed: which type a refusal names, how much of a
value it shows, which of two checks comes first, what a search holds a draw
against, what a history and a configuration hold of an agent, and who a
moment says draws.
"""

import dataclasses
import itertools
import random
import types
from collections.abc import Iterator
from typing import Any

import pytest

from anyorder import configuration
from anyorder import explore
from anyorder import oracle
from anyorder import scheduler
from anyorder import shrink
from anyorder import values


class _Log:
    """A world that keeps what agents record."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def record(self, entry: str) -> None:
        self.entries.append(entry)


async def _writer(ctx: scheduler.AgentContext) -> None:
    await ctx.call("record", entry=f"{ctx.name}:1")
    await ctx.call("record", entry=f"{ctx.name}:2")


async def _once(ctx: scheduler.AgentContext) -> None:
    await ctx.call("record", entry=ctx.name)


def _log(**agents: scheduler.Agent) -> scheduler.Scenario:
    return scheduler.Scenario(
        name="log", make_world=_Log, agents=agents, invariants=())


class _Holding:
    """A policy and a sampler that keeps the moment it was shown.

    Attributes:
        moment: The moment of the run, which goes on showing the run.
        asked: How many draws it was asked for.
    """

    def __init__(self) -> None:
        self.moment: scheduler.Moment | None = None
        self.asked = 0

    def choose(self, moment: scheduler.Moment) -> str:
        self.moment = moment
        return moment.waiting[0]

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> int:
        self.asked += 1
        return 1


class _Loud(str):
    """A string and more."""


class _Masked:
    """Says of itself that it is a str."""

    @property
    def __class__(self):
        return str

    def __repr__(self) -> str:
        return "masked"


def test_refusal_of_an_agents_name_shows_no_more_than_its_start():
    name = tuple(range(100))
    scenario = scheduler.Scenario(
        name="log", make_world=_Log, agents={name: _writer}, invariants=())
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    shown = repr(name)
    assert len(shown) > 200
    assert str(caught.value) == (
        f"an agent cannot be named {shown[:200]}... ({len(shown)} "
        "characters), of type tuple: an agent's name is a string")


def test_refusal_of_a_tools_name_shows_no_more_than_its_start():
    tool = b"x" * 300

    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call(tool)

    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(_log(a=caller), scheduler.RandomPolicy(0))
    shown = repr(tool)
    assert str(caught.value) == (
        f"agent 'a' cannot call a tool named {shown[:200]}... ({len(shown)} "
        "characters), of type bytes: a tool's name is a string")


def test_refusal_of_an_agents_name_names_its_type_as_type_gives_it():
    scenario = scheduler.Scenario(
        name="log", make_world=_Log, agents={_Masked(): _writer},
        invariants=())
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(scenario, scheduler.RandomPolicy(0))
    assert str(caught.value) == (
        "an agent cannot be named masked, of type _Masked: an agent's name "
        "is a string")


def test_refusal_of_a_tools_name_names_its_type_as_type_gives_it():
    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call(_Masked())

    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(_log(a=caller), scheduler.RandomPolicy(0))
    assert str(caught.value) == (
        "agent 'a' cannot call a tool named masked, of type _Masked: a "
        "tool's name is a string")


def test_moment_takes_no_attribute_of_anybodys_own():
    holding = _Holding()
    scheduler.run(_log(a=_once), holding)
    moment = holding.moment
    assert not hasattr(moment, "__dict__")
    with pytest.raises(AttributeError):
        moment.note = 1


class _Odd:
    """Cannot be hashed, and is shown otherwise than it is written."""

    __hash__ = None

    def __repr__(self) -> str:
        return "Odd()"

    def __str__(self) -> str:
        return "odd"


def test_refusal_of_a_state_that_cannot_be_hashed_stands_alone():
    scenario = dataclasses.replace(_log(), snapshot=lambda log: log.entries)
    with pytest.raises(TypeError) as caught:
        scheduler.state_of(scenario, _Log())
    # Raised from None: the error of hash is in the text and is not shown
    # a second time as what the refusal came of.
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_refusal_of_a_state_shows_it_as_python_shows_a_value():
    scenario = dataclasses.replace(_log(), snapshot=lambda log: _Odd())
    with pytest.raises(TypeError) as caught:
        scheduler.state_of(scenario, _Log())
    assert str(caught.value) == (
        "the snapshot of 'log' gave Odd(), which cannot be hashed: "
        "unhashable type: '_Odd'")


def test_state_is_the_value_the_snapshot_gave_when_it_was_asked_once():
    asked = []

    def snapshot(log: _Log) -> tuple[int]:
        asked.append(log)
        return (len(asked),)

    world = _Log()
    scenario = dataclasses.replace(_log(), snapshot=snapshot)
    assert scheduler.state_of(scenario, world) == (1,)
    assert asked == [world]


class _Sore:
    """A state whose hash raises an error of its own."""

    def __hash__(self) -> int:
        raise ValueError("sore")


def test_only_a_state_that_cannot_be_hashed_is_refused_in_other_words():
    scenario = dataclasses.replace(_log(), snapshot=lambda log: _Sore())
    with pytest.raises(ValueError, match="^sore$"):
        scheduler.state_of(scenario, _Log())


def test_replay_names_those_that_wait_in_name_order():
    # After its first call a waits again, and b has waited longest.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(
            _log(a=_writer, b=_writer), scheduler.ReplayPolicy(["a", "c"]))
    assert str(caught.value) == (
        "step 2 names 'c', which is not waiting; waiting: a, b")


def test_replay_holds_a_tool_against_its_step_by_its_name():
    tool = "".join(["rec", "ord"])
    step = scheduler.Step("a", tool, {"entry": "a"}, None)
    # Equal to the name the agent calls and another object, as a name
    # read from a file is.
    assert tool == "record" and id(tool) != id("record")
    result = scheduler.run(_log(a=_once), scheduler.ReplayPolicy([step]))
    assert result.steps == (step,)


def test_pick_takes_one_number_also_where_there_is_one_outcome():
    rng = random.Random(5)
    other = random.Random(5)
    assert scheduler.pick("a", (("only", 3),), rng) == "only"
    other.randrange(3)
    assert rng.getstate() == other.getstate()


def test_pick_takes_nothing_from_the_generator_for_what_it_refuses():
    rng = random.Random(5)
    before = rng.getstate()
    with pytest.raises(scheduler.UnsupportedValueError):
        scheduler.pick("a", "coin", rng)
    assert rng.getstate() == before


def test_tools_name_is_refused_before_its_arguments_are_looked_at():
    async def caller(ctx: scheduler.AgentContext) -> None:
        await ctx.call(_Loud("record"), entry={1, 2})

    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot call a tool named 'record', of type "
                  "_Loud: a tool's name is a string$"):
        scheduler.run(_log(a=caller), scheduler.RandomPolicy(0))


async def _drawer(ctx: scheduler.AgentContext) -> None:
    await ctx.draw("coin")
    await ctx.call("record", entry=ctx.name)


class _Longer:
    """Gives 300 times x, then 300 times y, and so on."""

    def __init__(self) -> None:
        self._letters = iter("xyz")

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> str:
        return next(self._letters) * 300


def test_refusal_of_another_draw_shows_the_start_of_each():
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        list(explore.exhaust(_log(a=_drawer, b=_once), _Longer()))
    assert str(caught.value) == (
        f"agent 'a' of 'log' drew '{'x' * 199}... (302 characters) in one "
        f"run and '{'y' * 199}... (302 characters) in another, where it had "
        "seen and done the same: an exhaustive search needs a sampler that "
        "gives one draw at one place")


class _SecondDiffers:
    """Gives the same for an agent's first draw and a new number after."""

    def __init__(self) -> None:
        self._numbers = itertools.count()

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        return "same" if not moment.draws else next(self._numbers)


def test_search_holds_every_draw_of_a_run_against_the_first_at_its_place():
    async def twice(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("one")
        await ctx.draw("two")
        await ctx.call("record", entry=ctx.name)

    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^agent 'a' of 'log' drew 0 in one run and 1 in another"):
        list(explore.exhaust(_log(a=twice, b=_once), _SecondDiffers()))


def test_search_does_not_yield_the_run_that_drew_otherwise():
    runs = []
    with pytest.raises(scheduler.ReplayDivergenceError):
        for run in explore.exhaust(_log(a=_drawer, b=_once), _Longer()):
            runs.append(run)
    assert [run.schedule for run in runs] == [("a", "b")]


class _Board:
    """A board that takes notes and marks, and can be closed."""

    def __init__(self) -> None:
        self.closed = False
        self.notes: list[str] = []
        self.marks: list[str] = []

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def note(self, who: str) -> None:
        self.notes.append(who)

    def mark(self, who: str) -> None:
        self.marks.append(who)


async def _noter(ctx: scheduler.AgentContext) -> None:
    """Notes its name, and marks it too on a board it read as open."""
    closed = await ctx.call("is_closed")
    await ctx.call("note", who=ctx.name)
    if not closed:
        await ctx.call("mark", who=ctx.name)


async def _closer(ctx: scheduler.AgentContext) -> None:
    await ctx.call("close")


def test_shrinker_goes_on_with_the_first_by_name_once_its_order_ends():
    scenario = scheduler.Scenario(
        name="board", make_world=_Board,
        agents={"a": _noter, "b": _noter, "c": _closer},
        invariants=(scheduler.Invariant(
            "one-note",
            lambda board: "two notes" if len(board.notes) > 1 else None,
            every_step=True),))
    run = explore.replay(scenario, ["c", "b", "a", "b", "a"])
    assert run.violation is not None
    shrunk = shrink.shrink(scenario, run)
    # With c put off to the end, a and b read an open board, and each has
    # one call more to make than the order holds for it. Those two calls
    # come once the order has ended, a before b by name, though b has
    # waited longer.
    assert shrunk.result.schedule == (
        "b", "b", "a", "a", "c", "a", "b")


class _Nothing:
    """Says that it is equal to None."""

    def __eq__(self, other: object) -> bool:
        return other is None

    def __hash__(self) -> int:
        return hash(None)


def test_key_refuses_what_says_it_is_equal_to_none():
    with pytest.raises(values.NotTakenError):
        values.key(_Nothing())


class _AsStr(type):
    """Answers as str does when a class is compared."""

    def __eq__(cls, other: object) -> bool:
        return other is str

    def __hash__(cls) -> int:
        return hash(str)


class _Passing(metaclass=_AsStr):
    """A value of a class that answers as str does."""

    def __repr__(self) -> str:
        return "passing"


def test_key_refuses_a_class_that_answers_as_str_does():
    with pytest.raises(values.NotTakenError):
        values.key(_Passing())


class _Given:
    """A sampler that gives what it was built with, in order."""

    def __init__(self, *given: Any) -> None:
        self._given = iter(given)

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        return next(self._given)


def test_two_draws_before_a_call_come_before_it_in_the_history():
    async def drawer(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("one")
        await ctx.draw("two")
        await ctx.call("record", entry="a")
        await ctx.draw("three")

    run = scheduler.run(
        _log(a=drawer), scheduler.ReplayPolicy(["a"]), _Given("x", "y", "z"))
    call = ("call", "record", ("dict", (("entry", "a"),)), None)
    assert [place for place, _ in configuration.places(run)] == [
        ("a", (), "one"),
        ("a", (("draw", "one", "x"),), "two"),
        ("a", (("draw", "one", "x"), ("draw", "two", "y"), call), "three"),
    ]


class _Teller:
    """Tells whoever asks what it was built with."""

    def __init__(self, told: Any) -> None:
        self._told = told

    def tell(self) -> Any:
        return self._told


def _told_place(told: Any) -> Any:
    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("tell")
        await ctx.draw("coin")

    scenario = scheduler.Scenario(
        name="teller", make_world=lambda: _Teller(told), agents={"a": asker},
        invariants=())
    run = scheduler.run(scenario, scheduler.ReplayPolicy(["a"]), _Given(0))
    ((place, _),) = configuration.places(run)
    return place


def test_history_holds_a_result_by_its_key():
    # The calls are the same call, and only what they returned differs.
    assert len({_told_place(1), _told_place(True), _told_place(1.0)}) == 3
    assert _told_place(["x"]) == (
        "a", (("call", "tell", ("dict", ()), ("list", ("x",))),), "coin")
    assert hash(_told_place(["x"])) == hash(_told_place(["x"]))


class _Standing:
    """A policy and a sampler that keep the configuration at each choice."""

    def __init__(self, schedule: str, *given: Any) -> None:
        self._schedule = iter(schedule)
        self._given = iter(given)
        self.stood: list[configuration.Configuration] = []

    def choose(self, moment: scheduler.Moment) -> str:
        self.stood.append(configuration.at(moment))
        return next(self._schedule)

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        return next(self._given)


def test_configuration_holds_what_an_agent_that_waits_has_drawn():
    def stood(given: Any) -> list[configuration.Configuration]:
        standing = _Standing("a", given)
        scenario = dataclasses.replace(
            _log(a=_drawer), snapshot=lambda log: tuple(log.entries))
        scheduler.run(scenario, standing, standing)
        return standing.stood

    assert stood(True) == [
        configuration.Configuration(
            (), (("a", (("draw", "coin", ("bool", True)),)),))]
    # An agent that drew otherwise can go on otherwise.
    assert stood(True) != stood(False)


class _Named(type):
    """A metaclass whose classes say that they are int."""

    @property
    def __name__(cls) -> str:
        return "int"


class _Liar(metaclass=_Named):
    """A value whose class gives itself another name."""

    def __repr__(self) -> str:
        return "liar"


def test_key_shows_no_more_than_the_start_of_what_it_refuses():
    value = set(range(100))
    shown = repr(value)
    with pytest.raises(values.NotTakenError) as caught:
        values.key(value)
    assert str(caught.value) == (
        f"a call does not carry {shown[:200]}... ({len(shown)} characters), "
        "of type set")


def test_key_names_the_type_of_what_it_refuses_as_type_gives_it():
    with pytest.raises(values.NotTakenError) as caught:
        values.key(_Liar())
    assert str(caught.value) == "a call does not carry liar, of type _Liar"


class _Counting:
    """Gives a new number at every draw."""

    def __init__(self) -> None:
        self._numbers = itertools.count()

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> int:
        return next(self._numbers)


def test_search_holds_draws_at_one_place_after_other_numbers_of_steps():
    async def late(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry="a")
        await ctx.draw("coin")

    # a has made its one call when it draws, after one step in the first
    # run and after two in the second: one place.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^agent 'a' of 'log' drew 0 in one run and 1 in another"):
        list(explore.exhaust(_log(a=late, b=_once), _Counting()))


def test_oracle_refuses_a_scenario_with_no_snapshot_and_no_agent():
    with pytest.raises(
            scheduler.MissingSnapshotError,
            match="^the scenario 'log' has no snapshot, so its runs leave "
                  "no outcome to compare$"):
        oracle.serial_outcomes(_log())


class _ByAgent:
    """A table of the tests' own: an agent draws its own name, always."""

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> str:
        return agent


def test_two_agents_with_one_history_stand_at_two_places_in_a_search():
    # a and b have done nothing when they draw from the same thing, and
    # each has a draw of its own: one draw at each place.
    runs = list(explore.exhaust(_log(a=_drawer, b=_drawer), _ByAgent()))
    assert [run.schedule for run in runs] == [("a", "b"), ("b", "a")]


class _SecondAgentDiffers:
    """Gives a the same in every run and b a new number at every draw."""

    def __init__(self) -> None:
        self._numbers = itertools.count()

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        return "same" if agent == "a" else next(self._numbers)


def test_search_holds_the_draws_of_every_agent_that_draws():
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^agent 'b' of 'log' drew 0 in one run and 1 in another"):
        list(explore.exhaust(
            _log(a=_drawer, b=_drawer), _SecondAgentDiffers()))


def test_configuration_holds_all_that_an_agent_that_waits_has_done():
    async def thrice(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry="one")
        await ctx.call("record", entry="two")
        await ctx.call("record", entry="three")

    standing = _Standing("aaa")
    scenario = dataclasses.replace(
        _log(a=thrice), snapshot=lambda log: len(log.entries))
    scheduler.run(scenario, standing)

    def call(entry: str) -> tuple:
        return ("call", "record", ("dict", (("entry", entry),)), None)

    assert standing.stood[2] == configuration.Configuration(
        2, (("a", (call("one"), call("two"))),))


def test_search_names_the_draw_of_a_run_that_does_not_replay_either():
    made = []

    class Fickle:
        """Answers otherwise from the fourth world on."""

        def __init__(self) -> None:
            made.append(self)
            self._late = len(made) >= 4

        def record(self, entry: str) -> bool:
            return self._late

    scenario = scheduler.Scenario(
        name="log", make_world=Fickle, agents={"a": _drawer, "b": _once},
        invariants=())
    # The first run and its replay have worlds 1 and 2. The second run
    # draws otherwise in world 3, and its replay would have world 4 and
    # give another trace. The draw is named, and the replay is not made.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^agent 'a' of 'log' drew 0 in one run and 1 in another"):
        list(explore.exhaust(scenario, _Counting()))
    assert len(made) == 3


class _Unshown:
    """A value whose repr raises."""

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def test_replay_refuses_an_entry_of_a_class_that_answers_as_str_does():
    assert type(_Passing()) == str and not type(_Passing()) != str
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_log(a=_writer), scheduler.ReplayPolicy([_Passing()]))
    assert str(caught.value) == (
        "step 1 names passing, of type _Passing: an agent's name is a "
        "string")


def test_replay_shows_no_more_than_the_start_of_an_entry_it_refuses():
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(
            _log(a=_writer), scheduler.ReplayPolicy([("x" * 500,)]))
    assert str(caught.value) == (
        f"step 1 names ('{'x' * 198}... (505 characters), of type tuple: "
        "an agent's name is a string")
    # And an entry that cannot be shown is still refused as an entry.
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_log(a=_writer), scheduler.ReplayPolicy([_Unshown()]))
    assert str(caught.value) == (
        "step 1 names a value that cannot be shown, of type _Unshown: an "
        "agent's name is a string")


def test_replay_names_the_type_of_an_entry_as_type_gives_it():
    assert _Liar.__name__ == "int"
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_log(a=_writer), scheduler.ReplayPolicy([_Liar()]))
    assert str(caught.value) == (
        "step 1 names liar, of type _Liar: an agent's name is a string")


class _Chooses:
    """A policy that chooses what it was built with."""

    def __init__(self, chosen: Any) -> None:
        self._chosen = chosen

    def choose(self, moment: scheduler.Moment) -> Any:
        return self._chosen


def test_policy_cannot_choose_a_value_of_a_class_that_answers_as_str_does():
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(_log(a=_writer), _Chooses(_Passing()))
    assert str(caught.value) == (
        "a policy cannot choose passing, of type _Passing: an agent's name "
        "is a string")


def test_refusal_of_a_policys_choice_shows_no_more_than_its_start():
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(_log(a=_writer), _Chooses(("x" * 500,)))
    assert str(caught.value) == (
        f"a policy cannot choose ('{'x' * 198}... (505 characters), of "
        "type tuple: an agent's name is a string")
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(_log(a=_writer), _Chooses(_Unshown()))
    assert str(caught.value) == (
        "a policy cannot choose a value that cannot be shown, of type "
        "_Unshown: an agent's name is a string")


def test_refusal_of_a_policys_choice_names_its_type_as_type_gives_it():
    with pytest.raises(scheduler.UnsupportedValueError) as caught:
        scheduler.run(_log(a=_writer), _Chooses(_Liar()))
    assert str(caught.value) == (
        "a policy cannot choose liar, of type _Liar: an agent's name is a "
        "string")


class _Mute(str):
    """A string that cannot be put into a message as it is."""

    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __format__(self, spec: str) -> str:
        raise RuntimeError("no format")


def _refusal(step: scheduler.Step) -> str:
    """Returns the words a replay refuses a step of agent a with."""
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        scheduler.run(_log(a=_writer), scheduler.ReplayPolicy([step]))
    return str(caught.value)


def test_tool_of_a_step_that_is_more_than_a_string_is_shown_as_a_value():
    # Equal to the tool the agent waits to call, so only the arguments
    # part the two, and the refusal must not be another error.
    step = scheduler.Step("a", _Mute("record"), {"entry": "other"}, None)
    assert _refusal(step) == (
        "step 1 is 'record'(entry='other'), but agent 'a' waits to call "
        "record(entry='a:1')")
    # Alike where the arguments are no mapping.
    assert _refusal(scheduler.Step("a", _Mute("record"), None, None)) == (
        "step 1 is 'record' with None, but agent 'a' waits to call "
        "record(entry='a:1')")


def test_arguments_of_a_step_in_a_mapping_that_is_no_dict_are_shown_alike():
    step = scheduler.Step(
        "a", "record", types.MappingProxyType({"entry": "other"}), None)
    assert _refusal(step) == (
        "step 1 is record(entry='other'), but agent 'a' waits to call "
        "record(entry='a:1')")


def test_arguments_of_a_step_that_are_no_mapping_are_shown_by_their_start():
    step = scheduler.Step("a", "record", ["x" * 500], None)
    assert _refusal(step) == (
        f"step 1 is record with ['{'x' * 198}... (504 characters), but "
        "agent 'a' waits to call record(entry='a:1')")
    assert _refusal(scheduler.Step("a", "record", _Unshown(), None)) == (
        "step 1 is record with a value that cannot be shown, but agent 'a' "
        "waits to call record(entry='a:1')")


def test_key_of_a_step_that_is_more_than_a_string_is_shown_as_a_value():
    step = scheduler.Step("a", "record", {_Mute("entry"): "other"}, None)
    assert _refusal(step) == (
        "step 1 is record('entry'='other'), but agent 'a' waits to call "
        "record(entry='a:1')")


def test_nobody_draws_once_what_an_agent_asked_to_draw_from_was_refused():
    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry=ctx.name)
        await ctx.draw({1, 2})

    holding = _Holding()
    with pytest.raises(scheduler.UnsupportedValueError):
        scheduler.run(_log(a=asker), holding, holding)
    # The sampler was never asked, so nobody drew.
    assert holding.asked == 0
    assert holding.moment.drawing is None


def test_nobody_draws_in_a_run_that_has_no_sampler():
    async def asker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("record", entry=ctx.name)
        await ctx.draw("coin")

    holding = _Holding()
    with pytest.raises(scheduler.UnsupportedAwaitError):
        scheduler.run(_log(a=asker), holding)
    assert holding.moment.drawing is None


def test_run_that_drew_and_made_no_step_is_not_judged_either():
    scenario = dataclasses.replace(
        _log(a=_once), snapshot=lambda log: tuple(log.entries))
    serial = oracle.serial_outcomes(scenario)
    # Put together by hand: no run of this scenario draws.
    drew = scheduler.RunResult((), None, (scheduler.Drawn("a", "coin", 7, 0),))
    with pytest.raises(
            ValueError,
            match="^the run made draws, and a run that drew is not judged "
                  "yet$"):
        oracle.judge(serial, drew)


async def _lost(ctx: scheduler.AgentContext) -> None:
    await ctx.call("missing")


def _alone(world: Any, agent: scheduler.Agent) -> scheduler.RunResult:
    """Runs one agent, named a, on a world of the given class."""
    scenario = scheduler.Scenario("alone", world, {"a": agent}, ())
    return scheduler.run(scenario, scheduler.RandomPolicy(0))


def test_world_that_lacks_a_tool_is_named_by_the_short_name_of_its_class():
    class Pad:
        """A world with no tool, of a class made inside a function."""

    assert Pad.__qualname__ != "Pad"
    with pytest.raises(scheduler.UnknownToolError) as caught:
        _alone(Pad, _lost)
    assert str(caught.value) == "Pad has no tool 'missing'"


class _Posing:
    """A world that says it is of another class."""

    @property
    def __class__(self) -> type:
        return dict


def test_world_is_named_by_its_type_whatever_it_says_its_class_is():
    assert _Posing().__class__ is dict
    with pytest.raises(scheduler.UnknownToolError) as caught:
        _alone(_Posing, _lost)
    assert str(caught.value) == "_Posing has no tool 'missing'"


class _Stubborn(str):
    """A string with a say of its own in how it is written out."""

    def __str__(self) -> str:
        return self

    def __format__(self, spec: str) -> str:
        return "LOUD"


class _Sham(Exception):
    """An error that says it is of another class."""

    @property
    def __class__(self) -> type:
        return KeyError


def _raised_by(error: Exception) -> scheduler.RunResult:
    """Runs an agent that calls, twice, a tool that raises the error."""
    class Mine:
        """A world whose one tool raises."""

        def step(self) -> None:
            raise error

    async def walker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("step")
        await ctx.call("step")

    return _alone(Mine, walker)


def test_tool_error_is_of_the_kind_of_its_type_whatever_it_says_it_is():
    assert _Sham("no").__class__ is KeyError
    result = _raised_by(_Sham("no"))
    assert [step.result for step in result.steps] == [
        scheduler.Raised("_Sham", "no"), scheduler.Raised("_Sham", "no")]


def test_text_of_a_tool_error_is_plain_whatever_the_error_says_it_is():
    class Boom(Exception):
        """An error whose text is more than a str."""

        def __str__(self) -> str:
            return _Stubborn("boom")

    # str and format of such a text give what it says of itself.
    assert type(str(_Stubborn("boom"))) is _Stubborn
    assert format(_Stubborn("boom")) == "LOUD"
    result = _raised_by(Boom())
    error = result.steps[0].result
    assert error == scheduler.Raised("Boom", "boom")
    assert type(error.message) is str
    assert len(result.steps) == 2


def test_replay_keeps_the_schedule_as_it_was_when_it_was_given():
    order = ["a", "b"]
    policy = scheduler.ReplayPolicy(order)
    order.reverse()
    result = scheduler.run(_log(a=_once, b=_once), policy)
    assert result.schedule == ("a", "b")


class _Asked:
    """A sampler that keeps who asked it, in order."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> str:
        self.asked.append(agent)
        return "heads"


def test_agents_are_started_in_the_order_python_puts_their_names_in():
    async def early(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("coin")

    sampler = _Asked()
    scheduler.run(
        _log(a=early, B=early, c=early), scheduler.RandomPolicy(0), sampler)
    # A capital comes before every small letter, as sorted has it and as
    # a moment gives those that wait.
    assert sampler.asked == ["B", "a", "c"]


async def _calls_then_draws(ctx: scheduler.AgentContext) -> None:
    await ctx.call("record", entry=ctx.name)
    await ctx.draw("coin")


class _Interrupted:
    """A sampler that is interrupted when it is asked."""

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        raise KeyboardInterrupt()


def test_nobody_draws_once_a_sampler_was_interrupted():
    holding = _Holding()
    with pytest.raises(KeyboardInterrupt):
        scheduler.run(_log(a=_calls_then_draws), holding, _Interrupted())
    assert holding.moment.drawing is None


class _GivesASet:
    """A sampler that gives what no call carries."""

    def sample(
        self, agent: str, request: Any, moment: scheduler.Moment
    ) -> Any:
        return {1, 2}


def test_nobody_draws_once_what_a_sampler_gave_was_refused():
    holding = _Holding()
    with pytest.raises(scheduler.UnsupportedValueError):
        scheduler.run(_log(a=_calls_then_draws), holding, _GivesASet())
    assert holding.moment.drawing is None


def test_search_holds_the_first_draw_of_a_run_as_well_as_its_last():
    async def twice(ctx: scheduler.AgentContext) -> None:
        await ctx.draw("coin")
        await ctx.draw("coin")
        await ctx.call("record", entry=ctx.name)

    # Two schedules, and a draws twice before either makes a call. In the
    # second run its first draw is another. Its second draw, the last of
    # the run, is then at a place no run has stood at and tells nothing.
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="^agent 'a' of 'log' drew 'x' in one run and 'z' in "
                  "another"):
        list(explore.exhaust(
            _log(a=twice, b=_once), _Given("x", "y", "z", "y")))


class _NoRepr:
    """A value whose repr raises an error of another kind."""

    def __repr__(self) -> str:
        raise TypeError("not today")


def test_value_is_shown_whatever_kind_of_error_its_repr_raises():
    assert values.shown(_NoRepr()) == "a value that cannot be shown"
    with pytest.raises(
            values.NotTakenError,
            match="^a call does not carry a value that cannot be shown, of "
                  "type _NoRepr$"):
        values.take(_NoRepr())


def test_refusal_shows_a_letter_outside_ascii_as_python_shows_it():
    # A letter that every Python prints as it is: a refusal shows a value
    # with repr and not with ascii.
    letter = chr(233)
    assert values.shown(letter) == "'" + letter + "'"


class _Twisty(str):
    """A string that gives another text whichever way it is copied."""

    def __iter__(self) -> Iterator[str]:
        return iter("OTHER")

    def __getitem__(self, index: Any) -> str:
        return "OTHER"

    def __add__(self, other: str) -> str:
        return "OTHER"

    def encode(self, *args: Any, **kwargs: Any) -> bytes:
        return b"OTHER"


class _SaysStr(str):
    """A string that says its class is str and is written out as LOUD."""

    @property
    def __class__(self) -> type:
        return str

    def __format__(self, spec: str) -> str:
        return "LOUD"


class _AnswersAsStr(type):
    """Answers as str does, and as itself, when a class is compared."""

    def __eq__(cls, other: object) -> bool:
        return other is str or other is cls

    def __hash__(cls) -> int:
        return hash(str)


class _LikeStr(str, metaclass=_AnswersAsStr):
    """A string of a class that answers as str, written out as LOUD."""

    def __format__(self, spec: str) -> str:
        return "LOUD"


_MORE_THAN_STR = [_Twisty, _SaysStr, _LikeStr]


@pytest.mark.parametrize("kind", _MORE_THAN_STR)
def test_refusal_names_a_class_by_the_text_of_its_name_and_no_other_way(
    kind
):
    # The name a class holds can be a string with ways of its own. The
    # refusal has the text of the name, taken as str itself takes it.
    bare = type(kind("Bare"), (), {})
    assert type(bare.__name__) is kind
    name = values.name_of(bare)
    assert type(name) is str
    assert name == "Bare"
    with pytest.raises(scheduler.UnknownToolError) as caught:
        _alone(bare, _lost)
    assert str(caught.value) == "Bare has no tool 'missing'"


@pytest.mark.parametrize("kind", _MORE_THAN_STR)
def test_text_of_a_tool_error_is_its_own_text_and_taken_no_other_way(kind):
    class Boom(Exception):
        """An error whose text is a string with ways of its own."""

        def __str__(self) -> str:
            return kind("boom")

    result = _raised_by(Boom())
    error = result.steps[0].result
    assert error == scheduler.Raised("Boom", "boom")
    assert type(error.message) is str
    # The agent was given the error as a result and went on.
    assert len(result.steps) == 2


@pytest.mark.parametrize(
    "text", [" no", "no ", "\tno\n", " \n ", "\n"],
    ids=["blank-before", "blank-after", "tab-and-line-end", "blanks-only",
         "line-end-only"])
def test_text_of_a_tool_error_is_kept_whole(text):
    result = _raised_by(ValueError(text))
    raised = scheduler.Raised("ValueError", text)
    assert [step.result for step in result.steps] == [raised, raised]
