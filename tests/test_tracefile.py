"""Checks trace files: what is written, read, refused and replayed."""

import collections
import dataclasses
import enum
import itertools
import json
import sys
from collections.abc import Sequence
from typing import Any

import pytest

from anyorder import explore
from anyorder import scheduler
from anyorder import tracefile
from anyorder.scenarios import oversell

# The shrunk failure the README prints, and its file written out by hand.
# Line 1 opens the file. Lines 2 to 7 are the README's six steps, each with
# its agent, its call and what the call returned. Line 8 holds the
# violation: the second order is the sixth call, so the step is 6, and the
# message names the orders as they were placed, buyer-1 first.
_SHRUNK = (
    "buyer-2", "buyer-1", "buyer-1", "buyer-1", "buyer-2", "buyer-2")
_LINES = (
    '{"kind": "anyorder-trace", "version": 2, "scenario": "oversell-naive"}',
    '{"kind": "step", "agent": "buyer-2", "tool": "read_stock", '
    '"args": {}, "result": 1}',
    '{"kind": "step", "agent": "buyer-1", "tool": "read_stock", '
    '"args": {}, "result": 1}',
    '{"kind": "step", "agent": "buyer-1", "tool": "write_stock", '
    '"args": {"value": 0}, "result": null}',
    '{"kind": "step", "agent": "buyer-1", "tool": "place_order", '
    '"args": {"buyer": "buyer-1"}, "result": null}',
    '{"kind": "step", "agent": "buyer-2", "tool": "write_stock", '
    '"args": {"value": 0}, "result": null}',
    '{"kind": "step", "agent": "buyer-2", "tool": "place_order", '
    '"args": {"buyer": "buyer-2"}, "result": null}',
    '{"kind": "end", "violation": {"invariant": "no-oversell", '
    '"message": "2 orders placed against a stock of 1: buyer-1, buyer-2", '
    '"step": 6}}',
)
_NO_VIOLATION = '{"kind": "end", "violation": null}'
# One buyer after the other: buyer-1 takes the unit in three calls, which
# are lines 3 to 5 above, and buyer-2 reads 0 and stops. Nothing breaks.
_SERIAL = ("buyer-1", "buyer-1", "buyer-1", "buyer-2")
_SERIAL_LINES = _LINES[:1] + _LINES[2:5] + (
    '{"kind": "step", "agent": "buyer-2", "tool": "read_stock", '
    '"args": {}, "result": 0}',
    _NO_VIOLATION,
)
# With a third buyer the shrunk failure goes on for one step: buyer-3
# reads 0 and stops. The violation stays at step 6, which is no longer the
# last.
_THIRD_BUYER_LINES = _LINES[:-1] + (
    '{"kind": "step", "agent": "buyer-3", "tool": "read_stock", '
    '"args": {}, "result": 0}',
    _LINES[-1],
)

# Each value with the text the module's docstring gives for it. No two
# texts are alike: True is not 1 and 1 is not 1.0, a tuple is not a list,
# and a dict with a "$" key is not a tuple or an error.
_VALUES = [
    (None, "null"),
    (True, "true"),
    (False, "false"),
    (0, "0"),
    (-7, "-7"),
    (2**70, "1180591620717411303424"),
    (0.5, "0.5"),
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (1.0, "1.0"),
    ("", '""'),
    ("caf\u00e9\n", '"caf\\u00e9\\n"'),
    ([], "[]"),
    ([1, [2, None]], "[1, [2, null]]"),
    ((), '{"$tuple": []}'),
    ((1, (2,), [3]), '{"$tuple": [1, {"$tuple": [2]}, [3]]}'),
    ({}, "{}"),
    ({"a": {"b": ()}}, '{"a": {"b": {"$tuple": []}}}'),
    ({"$tuple": [1]}, '{"$$tuple": [1]}'),
    ({"$$x": 1, "y": 2}, '{"$$$x": 1, "y": 2}'),
    (scheduler.Raised("ValueError", "only 1 left"),
     '{"$raised": {"kind": "ValueError", "message": "only 1 left"}}'),
    (scheduler.Raised((1,), None),
     '{"$raised": {"kind": {"$tuple": [1]}, "message": null}}'),
]


class _Echo:
    """A world whose one tool gives back what it is given."""

    def say(self, value: Any) -> Any:
        return value


def _echo(value: Any) -> scheduler.Scenario:
    """One agent that says one value: an argument and then a result."""
    async def speaker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("say", value=value)

    return scheduler.Scenario(
        name="echo",
        make_world=_Echo,
        agents={"a": speaker},
        invariants=(),
    )


def _said(value: Any) -> scheduler.RunResult:
    return explore.replay(_echo(value), ("a",))


def _built(value: Any) -> scheduler.RunResult:
    """A run of the echo scenario with a value, put together by hand.

    The scheduler makes no run with a value that no call carries, and the
    writer must still refuse to write one.
    """
    return scheduler.RunResult(
        (scheduler.Step("a", "say", {"value": value}, value),), None)


def _text(lines: Sequence[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def _echo_file(written: str) -> str:
    """The file of the echo scenario's run, with its value as written."""
    return _text((
        '{"kind": "anyorder-trace", "version": 2, "scenario": "echo"}',
        '{"kind": "step", "agent": "a", "tool": "say", '
        f'"args": {{"value": {written}}}, "result": {written}}}',
        _NO_VIOLATION,
    ))


def _edited(number: int, old: str, new: str) -> str:
    """The README failure's file with one line changed, counted from 1."""
    lines = list(_LINES)
    assert old in lines[number - 1]
    lines[number - 1] = lines[number - 1].replace(old, new)
    return _text(lines)


def _ledger() -> scheduler.Scenario:
    """Two writers add to a list that the world returns from every call."""
    class Ledger:
        def __init__(self) -> None:
            self.entries: list[str] = []

        def add(self, entry: str) -> list[str]:
            self.entries.append(entry)
            return self.entries

    async def writer(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add", entry=ctx.name)

    return scheduler.Scenario(
        name="ledger",
        make_world=Ledger,
        agents={"a": writer, "b": writer},
        invariants=(),
    )


def _ledger_file(first: str) -> str:
    """The file of a then b on the ledger, with the first result as given."""
    return _text((
        '{"kind": "anyorder-trace", "version": 2, "scenario": "ledger"}',
        '{"kind": "step", "agent": "a", "tool": "add", '
        f'"args": {{"entry": "a"}}, "result": {first}}}',
        '{"kind": "step", "agent": "b", "tool": "add", '
        '"args": {"entry": "b"}, "result": ["a", "b"]}',
        _NO_VIOLATION,
    ))


def _door() -> scheduler.Scenario:
    """One agent pushes a door that raises."""
    class Door:
        def push(self) -> None:
            raise RuntimeError("stuck")

    async def pusher(ctx: scheduler.AgentContext) -> None:
        await ctx.call("push")

    return scheduler.Scenario(
        name="door", make_world=Door, agents={"a": pusher}, invariants=())


_NOT_A_TRACE_FILE = [
    ("", "the text is empty"),
    ("\n", "line 1 is blank"),
    ("null\n", "line 1 is not a JSON object"),
    ("[1]\n", "line 1 is not a JSON object"),
    ("\ufeff" + _text(_LINES),
     "line 1 cannot be read: Unexpected UTF-8 BOM"),
    (_edited(1, "anyorder-trace", "trace"),
     'line 1 is of kind "trace", not "anyorder-trace"'),
    (_edited(1, '"kind": "anyorder-trace", ', ""),
     'line 1 is of kind null, not "anyorder-trace"'),
    (_edited(1, '"version": 2', '"version": 3'),
     "line 1 gives version 3, and this reader knows versions 1 and 2$"),
    (_edited(1, '"version": 2', '"version": 0'),
     "line 1 gives version 0, and this reader knows versions 1 and 2$"),
    (_edited(1, '"version": 2', '"version": "2"'),
     "line 1 gives version \"2\", and"),
    # True is equal to 1, which is a version, and is not one.
    (_edited(1, '"version": 2', '"version": true'),
     "line 1 gives version true, and"),
    (_edited(1, '"version": 2', '"version": 2.0'),
     "line 1 gives version 2.0, and"),
    (_edited(1, '"version": 2', '"version": 1.0'),
     "line 1 gives version 1.0, and"),
    (_edited(1, '"version": 2, ', ""), "line 1 gives no version"),
    # The version is looked at before the keys: a later version may put
    # other keys on this line.
    (_edited(1, '"version": 2', '"version": 3, "draws": 3'),
     "line 1 gives version 3, and"),
    (_edited(1, "}", ', "seed": 4}'),
     "line 1 holds the keys kind, scenario, seed, version, not kind, "
     "scenario, version"),
    (_edited(1, ', "scenario": "oversell-naive"', ""),
     "line 1 holds the keys kind, version, not kind, scenario, version"),
    (_edited(1, '"oversell-naive"', "7"),
     "line 1 holds 7 as its scenario, which is not a string"),
    (_text(_LINES[:1]), "the file has no end line"),
    (_text(_LINES[:-1]), 'line 7 is of kind "step", not "end"'),
    (_text(_LINES + _LINES[-1:]), 'line 8 is of kind "end", not "step"'),
    (_text(_LINES + _LINES[1:2]), 'line 8 is of kind "end", not "step"'),
    (_text(_LINES[:2] + ("",) + _LINES[2:]), "line 3 is blank"),
    (_text(_LINES[:2] + (" \t",) + _LINES[2:]), "line 3 is blank"),
    (_text(_LINES) + "\n", "line 9 is blank"),
    # json counts lines too, and its count is not the file's.
    (_text(_LINES[:2] + ("# a note",) + _LINES[2:]),
     r"line 3 cannot be read: Expecting value \(column 1\)$"),
    (_edited(3, '"kind": "step"', '"kind": "draw"'),
     'line 3 is of kind "draw", not "step"'),
    (_edited(2, '"result": 1', '"result": 1, "hash": "ab"'),
     "line 2 holds the keys agent, args, hash, kind, result, tool, not "
     "agent, args, kind, result, tool"),
    (_edited(2, ', "result": 1', ""),
     "line 2 holds the keys agent, args, kind, tool, not "),
    (_edited(2, '"buyer-2"', "2"),
     "line 2 holds 2 as its agent, which is not a string"),
    (_edited(2, '"read_stock"', "null"),
     "line 2 holds null as its tool, which is not a string"),
    (_edited(2, '"args": {}', '"args": []'),
     r"line 2 holds \[\] as its args, which is not an object"),
    (_edited(2, '"result": 1', '"result": NaN'),
     "line 2 cannot be read: NaN is not a finite number"),
    (_edited(2, '"result": 1', '"result": -Infinity'),
     "line 2 cannot be read: -Infinity is not a finite number"),
    (_edited(2, '"result": 1', '"result": 1e999'),
     "line 2 cannot be read: 1e999 is not a finite number"),
    # The line is one level and the lists are 100 more.
    (_edited(2, '"result": 1', f'"result": {"[" * 100}{"]" * 100}'),
     "line 2 is nested deeper than 100 levels"),
    # Every line has the bound, also those that hold no value of a run.
    (_edited(1, '"oversell-naive"', f'{"[" * 100}{"]" * 100}'),
     "line 1 is nested deeper than 100 levels"),
    (_text(_LINES[:-1] + (
        f'{{"kind": "end", "violation": {"[" * 100}{"]" * 100}}}',)),
     "line 8 is nested deeper than 100 levels"),
    (_edited(1, '"version": 2', '"version": 3, "version": 2'),
     "line 1 cannot be read: an object gives the key 'version' twice"),
    (_edited(2, '"result": 1', '"result": 9, "result": 1'),
     "line 2 cannot be read: an object gives the key 'result' twice"),
    (_edited(4, '{"value": 0}', '{"value": 5, "value": 0}'),
     "line 4 cannot be read: an object gives the key 'value' twice"),
    (_text(_LINES[:-1] + ('{"kind": "end", "violation": 6}',)),
     "line 8 holds a violation that is neither null nor an object"),
    (_text(_LINES[:-1] + ('{"kind": "end", "violation": false}',)),
     "line 8 holds a violation that is neither null nor an object"),
    (_text(_LINES[:-1] + ('{"kind": "end", "violation": null, "draws": 1}',)),
     "line 8 holds the keys draws, kind, violation, not kind, violation"),
    (_text(_LINES[:-1] + ('{"kind": "end"}',)),
     "line 8 holds the keys kind, not kind, violation"),
    (_edited(8, ', "step": 6', ""),
     "line 8 holds a violation that is neither null nor an object with "
     "the keys invariant, message, step"),
    (_edited(8, '"step": 6', '"step": 6, "seed": 4'),
     "line 8 holds a violation that is neither null nor an object"),
    (_edited(8, '"step": 6', '"step": "6"'),
     "line 8 holds \"6\" as its step, which is not an integer"),
    (_edited(8, '"step": 6', '"step": true'),
     "line 8 holds true as its step, which is not an integer"),
    (_edited(8, '"step": 6', '"step": 6.0'),
     "line 8 holds 6.0 as its step, which is not an integer"),
    (_edited(8, '"no-oversell"', "null"),
     "line 8 holds null as its invariant, which is not a string"),
    (_edited(8, '"2 orders placed', '["2 orders placed').replace(
        'buyer-2", "step"', 'buyer-2"], "step"'),
     "as its message, which is not a string"),
]

_ANOTHER_TRACE = [
    # Both buyers read 1, so a 2 on line 2 is the first step's result.
    (_edited(2, '"result": 1', '"result": 2'),
     'step 1 is {"agent": "buyer-2", "tool": "read_stock", "args": {}, '
     '"result": 2} in the file and {"agent": "buyer-2", "tool": '
     '"read_stock", "args": {}, "result": 1} in the run'),
    (_edited(2, '"result": 1', '"result": true'), "step 1 is "),
    (_edited(2, '"result": 1', '"result": 1.0'), "step 1 is "),
    (_edited(2, '"result": 1', '"result": "1"'), "step 1 is "),
    (_edited(4, '"value": 0', '"value": false'), "step 3 is "),
    (_edited(4, '"value": 0', '"value": 5'), "step 3 is "),
    (_edited(4, '{"value": 0}', "{}"), "step 3 is "),
    (_edited(4, '{"value": 0}', '{"value": 0, "more": 1}'), "step 3 is "),
    (_edited(5, '{"buyer": "buyer-1"}', '{"buyer": "buyer-2"}'),
     "step 4 is "),
    # No value is written with a tag this version lacks.
    (_edited(2, '"result": 1', '"result": {"$set": [1]}'), "step 1 is "),
    (_edited(2, "read_stock", "reserve"), "step 1 is "),
    (_edited(8, "buyer-1, buyer-2", "buyer-2, buyer-1"),
     "the violation is "),
    (_edited(8, '"step": 6', '"step": 5'), "the violation is "),
    (_edited(8, "no-oversell", "no-undersell"), "the violation is "),
    (_text(_LINES[:-1] + (_NO_VIOLATION,)),
     "the violation is null in the file and "),
    # The serial run breaks nothing, and the file says it does.
    (_text(_SERIAL_LINES[:-1] + (_LINES[-1],)),
     "in the file and null in the run"),
    # As deep as a line may be: read, and then not what the call returned.
    (_edited(2, '"result": 1', f'"result": {"[" * 99}{"]" * 99}'),
     "step 1 is "),
    # Lines 4 and 5 are buyer-1's write and its order. Swapped, the agents
    # come in the same order and only the calls differ.
    (_text(_LINES[:3] + (_LINES[4], _LINES[3]) + _LINES[5:]), "step 3 is "),
    # Without its line 7, buyer-2 still has its order to place.
    (_text(_LINES[:6] + _LINES[7:]),
     "the schedule ends after 5 steps but buyer-2 can still run"),
    # With one more read, the run is over before the schedule is.
    (_text(_LINES[:-1] + (_LINES[1], _LINES[-1])),
     "the run ended after 6 steps but the schedule has 7"),
    # buyer-1 reads 1 and still has its write and its order.
    (_text((_LINES[0], _LINES[2], _LINES[-1])),
     "the schedule ends after 1 steps but buyer-1, buyer-2 can still run"),
    # No agent of the scenario has this name.
    (_edited(2, "buyer-2", "buyer-9"),
     "step 1 names 'buyer-9', which is not waiting"),
]


def test_file_of_the_shrunk_failure_replays_to_it():
    scenario = oversell.naive()
    saved = tracefile.loads(_text(_LINES))
    assert saved.scenario == "oversell-naive"
    assert saved.schedule == _SHRUNK
    run = tracefile.replay(scenario, saved)
    assert run == explore.replay(scenario, _SHRUNK)
    assert run.violation == scheduler.Violation(
        "no-oversell",
        "2 orders placed against a stock of 1: buyer-1, buyer-2",
        6)


def _version_1(text: str) -> str:
    """A text with the number of version 1 in place of 2."""
    assert text.count('"version": 2') == 1
    return text.replace('"version": 2', '"version": 1')


@pytest.mark.parametrize(
    "buyers,lines",
    [(2, _LINES), (2, _SERIAL_LINES), (3, _THIRD_BUYER_LINES)],
    ids=["failure", "serial", "third-buyer"])
def test_text_of_version_1_is_read_and_replays_as_version_2_does(
    buyers, lines
):
    # No value of these runs is changed after its call, so the text holds
    # the values of the calls, which is what version 2 asks, under
    # either number.
    scenario = oversell.naive(buyers=buyers)
    new = tracefile.loads(_text(lines))
    old = tracefile.loads(_version_1(_text(lines)))
    assert old == new
    assert tracefile.replay(scenario, old) == explore.replay(
        scenario, old.schedule)


def test_file_of_a_run_that_passed_replays_to_it():
    saved = tracefile.loads(_text(_SERIAL_LINES))
    assert saved.schedule == _SERIAL
    assert saved.violation is None
    run = tracefile.replay(oversell.naive(), saved)
    assert run == explore.replay(oversell.naive(), _SERIAL)
    assert run.violation is None


def test_file_holds_the_step_of_a_violation_that_is_not_the_last():
    scenario = oversell.naive(buyers=3)
    saved = tracefile.loads(_text(_THIRD_BUYER_LINES))
    run = tracefile.replay(scenario, saved)
    assert len(run.steps) == 7
    assert run.violation is not None and run.violation.step == 6
    late = _text(_THIRD_BUYER_LINES).replace('"step": 6', '"step": 7')
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="the violation is "):
        tracefile.replay(scenario, tracefile.loads(late))


def test_file_holds_what_it_reads_as_the_file_has_it():
    saved = tracefile.loads(_echo_file('{"$tuple": [1, 2]}'))
    # Nothing is turned back into a tuple: what is read is not a run.
    assert saved == tracefile.Saved(
        scenario="echo",
        steps=({
            "agent": "a",
            "tool": "say",
            "args": {"value": {"$tuple": [1, 2]}},
            "result": {"$tuple": [1, 2]},
        },),
        violation=None)


def test_file_reads_the_same_with_other_spacing_key_order_and_line_ends():
    compact = "\r\n".join(
        json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))
        for line in _LINES)
    assert compact.count("\n") == 7
    assert tracefile.loads(compact) == tracefile.loads(_text(_LINES))
    assert tracefile.loads(compact + "\r\n") == tracefile.loads(_text(_LINES))
    # The keys of each line are in another order now, and it still replays.
    step = compact.splitlines()[1]
    assert json.loads(step) == json.loads(_LINES[1])
    assert list(json.loads(step)) != list(json.loads(_LINES[1]))
    tracefile.replay(oversell.naive(), tracefile.loads(compact))


@pytest.mark.parametrize(
    "text,message",
    _NOT_A_TRACE_FILE,
    ids=[message for _, message in _NOT_A_TRACE_FILE])
def test_text_that_is_not_a_trace_file_is_refused(text, message):
    with pytest.raises(tracefile.FormatError, match=message):
        tracefile.loads(text)


@pytest.mark.parametrize("kept", range(len(_LINES)))
def test_file_cut_short_is_refused(kept):
    # The violation is on the last line, so a file without it is not a
    # failure with fewer steps. It is no trace file at all.
    with pytest.raises(tracefile.FormatError):
        tracefile.loads(_text(_LINES[:kept]))


def test_file_cut_inside_a_line_is_refused():
    with pytest.raises(
            tracefile.FormatError,
            match=r"line 8 cannot be read: Unterminated string starting "
                  r"at \(column 70\)$"):
        tracefile.loads(_text(_LINES)[:-20])


def test_line_this_python_cannot_read_is_refused(monkeypatch):
    # How deep a line json reads depends on the Python and on the stack it
    # is given, so the test stands in for a line that is too deep.
    def too_deep(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError("too deep")

    monkeypatch.setattr(json, "loads", too_deep)
    with pytest.raises(
            tracefile.FormatError, match="line 1 cannot be read: too deep"):
        tracefile.loads(_text(_LINES))


def test_line_ends_only_at_a_line_feed():
    # JSON takes U+2028 raw inside a string, and str.splitlines would
    # break the line there.
    value = "a" + chr(0x2028) + chr(0x85) + "b"
    saved = tracefile.loads(
        _echo_file(json.dumps(value, ensure_ascii=False)))
    assert tracefile.replay(_echo(value), saved) == _said(value)


def test_line_of_a_kind_this_version_lacks_is_refused_and_not_skipped():
    # A reader that skipped a line it does not know could replay another
    # run than the file holds and call it the same.
    draw = '{"kind": "draw", "agent": "buyer-1", "value": true}'
    with pytest.raises(
            tracefile.FormatError,
            match='line 4 is of kind "draw", not "step"'):
        tracefile.loads(_text(_LINES[:3] + (draw,) + _LINES[3:]))


@pytest.mark.parametrize(
    "text,message",
    _ANOTHER_TRACE,
    ids=[message[:30] for _, message in _ANOTHER_TRACE])
def test_file_that_replays_to_another_trace_is_refused(text, message):
    saved = tracefile.loads(text)
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        tracefile.replay(oversell.naive(), saved)
    assert message in str(caught.value)


def test_file_with_the_two_reads_swapped_is_the_file_of_another_run():
    # Nothing ties a file to the run it was written from. Lines 2 and 3
    # are the two reads of 1. Swapped, the file holds the schedule where
    # buyer-1 reads first, and that run makes the same calls with the same
    # results and places the orders in the same order. The claim is true,
    # so it replays.
    saved = tracefile.loads(
        _text(_LINES[:1] + (_LINES[2], _LINES[1]) + _LINES[3:]))
    assert saved.schedule == (
        "buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2")
    run = tracefile.replay(oversell.naive(), saved)
    assert run.schedule == saved.schedule
    assert run.violation is not None


def test_file_is_refused_by_the_same_scenario_with_other_numbers():
    # The file names its scenario and not the stock or the buyers. It is
    # the replay that tells: with a stock of 2 the first read gives 2, and
    # a third buyer is left waiting when the schedule ends.
    saved = tracefile.loads(_text(_LINES))
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.replay(oversell.naive(stock=2), saved)
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="buyer-3 can still run"):
        tracefile.replay(oversell.naive(buyers=3), saved)


def test_file_of_a_scenario_with_another_name_is_refused_before_it_runs():
    worlds: list[oversell.Inventory] = []

    def make_world() -> oversell.Inventory:
        worlds.append(oversell.Inventory(1))
        return worlds[-1]

    control = oversell.atomic()
    scenario = scheduler.Scenario(
        name=control.name,
        make_world=make_world,
        agents=control.agents,
        invariants=control.invariants,
    )
    with pytest.raises(ValueError) as caught:
        tracefile.replay(scenario, tracefile.loads(_text(_LINES)))
    # Not a divergence: nothing was replayed.
    assert type(caught.value) is ValueError
    assert str(caught.value) == (
        "the file holds a run of 'oversell-naive', not of 'oversell-atomic'")
    assert not worlds


@pytest.mark.parametrize(
    "value,written", _VALUES, ids=[written for _, written in _VALUES])
def test_value_replays_from_the_text_the_rules_give(value, written):
    saved = tracefile.loads(_echo_file(written))
    assert tracefile.replay(_echo(value), saved) == _said(value)


@pytest.mark.parametrize(
    "ran,held",
    [
        (1, "true"), (True, "1"), (1, "1.0"), (1.0, "1"), (0, "false"),
        (False, "0"), (0.0, "-0.0"), (-0.0, "0.0"), (0, "0.0"), (0, "-0.0"),
    ])
def test_file_with_a_value_python_takes_for_equal_is_refused(ran, held):
    # The run of the file's schedule is equal to the run the file tells
    # of, as Python compares, and it is not the same trace.
    assert json.loads(held) == ran
    saved = tracefile.loads(_echo_file(held))
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.replay(_echo(ran), saved)


@pytest.mark.parametrize(
    "ran,held",
    [
        (1.0, "1.00"), (1.0, "1e0"), (1.0, "10E-1"), (100.0, "1e2"),
        (0, "-0"), ("A", '"' + chr(92) + 'u0041"'),
    ])
def test_file_with_a_value_spelled_another_way_replays(ran, held):
    # A number is read as Python reads it and a string as JSON reads it:
    # the spelling is not part of what a file holds. The kind is: the
    # test above refuses 1 for 1.0.
    saved = tracefile.loads(_echo_file(held))
    assert tracefile.replay(_echo(ran), saved) == _said(ran)


def test_file_with_the_keys_of_a_value_in_another_order_replays():
    value = {"a": 1, "b": {"c": 2, "d": 3}}
    saved = tracefile.loads(_echo_file('{"b": {"d": 3, "c": 2}, "a": 1}'))
    assert tracefile.replay(_echo(value), saved) == _said(value)


@pytest.mark.parametrize(
    "ran,held",
    [
        # JSON has one kind of array, and the file keeps the two apart.
        ([1, 2], '{"$tuple": [1, 2]}'),
        ((1, 2), "[1, 2]"),
        # A dict with the key of a tag is not what the tag stands for.
        ({"$tuple": [1]}, '{"$tuple": [1]}'),
        ((1,), '{"$$tuple": [1]}'),
        ({"$raised": {"kind": "K", "message": "m"}},
         '{"$raised": {"kind": "K", "message": "m"}}'),
        (scheduler.Raised("K", "m"), '{"kind": "K", "message": "m"}'),
        ("1", "1"),
        (None, '"None"'),
    ],
    ids=repr)
def test_file_with_a_value_of_another_kind_is_refused(ran, held):
    saved = tracefile.loads(_echo_file(held))
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.replay(_echo(ran), saved)


@pytest.mark.parametrize("lists,fits", [(98, True), (99, False)])
def test_file_is_read_as_deep_as_a_line_may_be(lists, fits):
    # The line is one level, its args are another, and 98 lists make 100.
    value: list[Any] = []
    for _ in range(lists - 1):
        value = [value]
    text = _echo_file("[" * lists + "]" * lists)
    if fits:
        saved = tracefile.loads(text)
        assert tracefile.replay(_echo(value), saved) == _said(value)
    else:
        with pytest.raises(
                tracefile.FormatError,
                match="line 2 is nested deeper than 100 levels"):
            tracefile.loads(text)


def test_run_with_a_surrogate_pair_is_not_the_run_of_a_file():
    # json writes a pair of surrogates as it writes the one character the
    # pair stands for. A file that held both would replay against either.
    one = chr(0x1F600)
    two = chr(0xD83D) + chr(0xDE00)
    assert json.dumps(one) == json.dumps(two) and one != two
    saved = tracefile.loads(_echo_file(json.dumps(one)))
    assert tracefile.replay(_echo(one), saved) == _said(one)
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="step 1: a trace file does not hold .*, a string with a "
                  "surrogate"):
        tracefile.replay(_echo(two), saved)


@pytest.mark.parametrize("part", ["invariant", "message"])
def test_violation_with_a_surrogate_pair_is_not_the_violation_of_a_file(part):
    # Not in a step: nothing puts a step's number before the refusal, and
    # it is still a divergence.
    one = chr(0x1F600)
    two = chr(0xD83D) + chr(0xDE00)

    def broken(text: str) -> scheduler.Scenario:
        held = {"invariant": "holds", "message": "broken", part: text}
        return scheduler.Scenario(
            name="echo",
            make_world=_Echo,
            agents={},
            invariants=(
                scheduler.Invariant(
                    held["invariant"], lambda world: held["message"],
                    every_step=False),
            ),
        )

    end = (
        '{"kind": "end", "violation": {"invariant": "holds", '
        '"message": "broken", "step": 0}}')
    old = "holds" if part == "invariant" else "broken"
    saved = tracefile.loads(_text((
        '{"kind": "anyorder-trace", "version": 2, "scenario": "echo"}',
        end.replace(f'"{old}"', json.dumps(one)),
    )))
    assert saved.violation is not None and saved.violation[part] == one
    assert tracefile.replay(broken(one), saved).violation is not None
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        tracefile.replay(broken(two), saved)
    assert str(caught.value) == (
        "a trace file does not hold '\\ud83d\\ude00', a string with a "
        "surrogate")


@pytest.mark.parametrize(
    "halves",
    [
        chr(0xD83D) + chr(0xDE00),
        chr(0xD83D) + chr(92) + "ude00",
        chr(92) + "ud83d" + chr(0xDE00),
    ],
    ids=["both-raw", "high-raw", "low-raw"])
def test_text_with_a_surrogate_that_is_not_escaped_is_refused(halves):
    # A text decoded strictly never holds one. Decoded with leniency it
    # can, and json does not join a raw half with the other half, so the
    # line would hold two code points that are written as the one
    # character is.
    text = _echo_file(f'"{halves}"')
    with pytest.raises(
            tracefile.FormatError,
            match="line 2 holds a surrogate that is not escaped"):
        tracefile.loads(text)


def test_scenario_with_a_value_no_call_carries_does_not_replay_a_file():
    # The file says the call was made with a list. The scenario makes it
    # with a set, which no call carries, so its run ends at the call. That
    # is a refusal of the scenario and not a divergence: there is no run
    # to hold against the file.
    saved = tracefile.loads(_echo_file("[1, 2]"))
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot call 'say': a call does not carry "
                  "{1, 2}, of type set$"):
        tracefile.replay(_echo({1, 2}), saved)


@dataclasses.dataclass
class _Link:
    """One link of a chain, as a tool might return one. No file holds it."""

    rest: Any


def _chain(links: int) -> Any:
    value: Any = None
    for _ in range(links):
        value = _Link(value)
    return value


def test_scenario_with_a_value_python_cannot_show_does_not_replay_a_file():
    # The refusal shows the value, and to show a chain of 5000 links
    # Python calls deeper than it can. The run is refused all the same,
    # and with the error of a value that no call carries.
    value = _chain(5000)
    with pytest.raises(RecursionError):
        repr(value)
    saved = tracefile.loads(_echo_file("1"))
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="a call does not carry a value that cannot be shown, of "
                  "type _Link$"):
        tracefile.replay(_echo(value), saved)


def test_tool_error_replays_from_its_file():
    text = _text((
        '{"kind": "anyorder-trace", "version": 2, "scenario": "door"}',
        '{"kind": "step", "agent": "a", "tool": "push", "args": {}, '
        '"result": {"$raised": {"kind": "RuntimeError", '
        '"message": "stuck"}}}',
        _NO_VIOLATION,
    ))
    run = tracefile.replay(_door(), tracefile.loads(text))
    assert run.steps[0].result == scheduler.Raised("RuntimeError", "stuck")
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.replay(
            _door(), tracefile.loads(text.replace("stuck", "open")))


@pytest.mark.parametrize("version", [1, 2])
def test_file_holds_a_shared_list_as_it_was_at_each_call(version):
    # The world returns its own list, and the second call changes it. The
    # trace holds each result as it was when its call returned, so the
    # file that replays is the one with what the first call returned, and
    # the file with the final content is refused. Under version 1 too,
    # whose writer wrote the final content for this run: its text is
    # read, and it is not the run a scheduler makes now.
    at_call, at_end = (
        _ledger_file(first) if version == 2
        else _version_1(_ledger_file(first))
        for first in ('["a"]', '["a", "b"]'))
    run = tracefile.replay(_ledger(), tracefile.loads(at_call))
    assert [step.result for step in run.steps] == [["a"], ["a", "b"]]
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.replay(_ledger(), tracefile.loads(at_end))


class _Colour(enum.IntEnum):
    RED = 1


_Point = collections.namedtuple("_Point", "x y")


class _Name(str):
    pass


class _Stack(list):
    pass


class _Ratio(float):
    pass


class _Failure(scheduler.Raised):
    pass


def _nested(wrap: Any, times: int) -> Any:
    value: Any = 0
    for _ in range(times):
        value = wrap(value)
    return value


def test_shrunk_failure_is_written_as_the_file_above():
    scenario = oversell.naive()
    failure = explore.replay(scenario, _SHRUNK)
    assert tracefile.dumps(scenario, failure) == _text(_LINES)


def test_run_that_passed_is_written_with_no_violation():
    scenario = oversell.naive()
    serial = explore.replay(scenario, _SERIAL)
    assert tracefile.dumps(scenario, serial) == _text(_SERIAL_LINES)


def test_violation_is_written_with_its_step_when_that_is_not_the_last():
    # The check of a replay writes the run as the writer does, so a wrong
    # step would agree with itself. Only the text can tell.
    scenario = oversell.naive(buyers=3)
    run = explore.replay(scenario, _SHRUNK + ("buyer-3",))
    assert tracefile.dumps(scenario, run) == _text(_THIRD_BUYER_LINES)


@pytest.mark.parametrize(
    "value,written", _VALUES, ids=[written for _, written in _VALUES])
def test_value_is_written_as_the_rules_give(value, written):
    assert tracefile.dumps(_echo(value), _said(value)) == _echo_file(written)


def test_file_is_ascii_with_one_json_object_on_each_line():
    value = "caf\u00e9\u2028\r\n"
    text = tracefile.dumps(_echo(value), _said(value))
    # A line break inside a value is written as an escape, so a line of the
    # file is a line whatever a reader takes for a line break.
    assert text.isascii()
    assert text.endswith("\n")
    assert "\r" not in text
    assert len(text.splitlines()) == text.count("\n") == 3
    assert all(type(json.loads(line)) is dict for line in text.splitlines())


@pytest.mark.parametrize(
    "scenario,schedules",
    [
        (oversell.naive(), 16),
        (oversell.atomic(buyers=3), 6),
        (oversell.naive(buyers=3), 702),
    ],
    ids=["naive", "control", "three-buyers"])
def test_every_run_reads_back_and_replays_as_it_was_written(
    scenario, schedules
):
    # The counts are those test_explore derives by hand. Runs that passed
    # are written too.
    runs = list(explore.exhaust(scenario))
    assert len(runs) == schedules
    for run in runs:
        saved = tracefile.loads(tracefile.dumps(scenario, run))
        assert saved.scenario == scenario.name
        assert saved.schedule == run.schedule
        assert tracefile.replay(scenario, saved) == run


_NOT_HELD = [
    ({1, 2}, "set"),
    (frozenset(), "frozenset"),
    (b"x", "bytes"),
    (object(), "object"),
    (float("nan"), "float"),
    (float("inf"), "float"),
    (_Colour.RED, "_Colour"),
    (_Point(1, 2), "_Point"),
    (_Name("a"), "_Name"),
    (_Stack([1]), "_Stack"),
    (collections.OrderedDict(), "OrderedDict"),
    ([{"a": ({2},)}], "set"),
    ({1: "a"}, "dict"),
    ({_Name("a"): 1}, "dict"),
    (_Ratio(0.5), "_Ratio"),
    (_Failure("K", "m"), "_Failure"),
]


@pytest.mark.parametrize("value,kind", _NOT_HELD)
def test_value_the_file_cannot_hold_is_not_written(value, kind):
    # Written as its nearest JSON value it would be written as another
    # value is, and the file would replay against a run that is not its.
    with pytest.raises(
            TypeError,
            match=f"step 1: a trace file does not hold .*, of type {kind}$"):
        tracefile.dumps(_echo(value), _built(value))


@pytest.mark.parametrize("value,kind", _NOT_HELD)
def test_value_the_file_cannot_hold_does_not_run(value, kind):
    # No call carries any of these either, so the run ends at the call,
    # long before there is anything to write.
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot call 'say': a call does not carry "
                  f".*, of type {kind}$"):
        _said(value)


@pytest.mark.parametrize(
    "value",
    [
        chr(0xD83D) + chr(0xDE00),
        chr(0xD83D),
        "a" + chr(0xDFFF),
        [chr(0xDC00)],
        {chr(0xD800): 1},
    ],
    ids=ascii)
def test_string_with_a_surrogate_is_not_written(value):
    # The first is a pair, which json writes as it writes U+1F600. The
    # others would read back as they are, and the rule is kept simple.
    with pytest.raises(
            TypeError,
            match="step 1: a trace file does not hold .*, a string with a "
                  "surrogate$"):
        tracefile.dumps(_echo(value), _said(value))


def test_value_python_cannot_show_is_not_written():
    value = _chain(5000)
    with pytest.raises(
            TypeError,
            match="step 1: a trace file does not hold a value too deep to "
                  "show, of type _Link$"):
        tracefile.dumps(_echo(value), _built(value))
    # A message is shown the same way, and it is not in a step.
    scenario = scheduler.Scenario(
        name="echo",
        make_world=_Echo,
        agents={},
        invariants=(
            scheduler.Invariant(
                "holds", lambda world: value, every_step=False),
        ),
    )
    with pytest.raises(
            TypeError,
            match="^a trace file does not hold a value too deep to show, "
                  "of type _Link, as the violation's message$"):
        tracefile.dumps(scenario, explore.replay(scenario, ()))


def test_value_that_holds_itself_is_not_written():
    value: list[Any] = []
    value.append(value)
    with pytest.raises(TypeError, match="step 1: .* nested so deep"):
        tracefile.dumps(_echo(value), _built(value))
    with pytest.raises(
            scheduler.UnsupportedValueError, match="nested so deep"):
        _said(value)


@pytest.mark.parametrize(
    "wrap,most",
    [
        (lambda value: [value], 98),
        (lambda value: {"k": value}, 98),
        (lambda value: (value,), 49),
        (lambda value: scheduler.Raised("K", value), 49),
    ],
    ids=["list", "dict", "tuple", "error"])
def test_value_is_written_as_deep_as_a_line_may_be_and_no_deeper(wrap, most):
    # A line may be nested 100 deep. The line is one level and its args
    # are another, which leaves 98 to an argument: 98 lists or dicts, or
    # 49 tuples or errors, which take two levels each.
    fits = _nested(wrap, most)
    saved = tracefile.loads(tracefile.dumps(_echo(fits), _said(fits)))
    assert tracefile.replay(_echo(fits), saved) == _said(fits)
    deeper = _nested(wrap, most + 1)
    # The scheduler has the bound as well, so a run that is made holds
    # nothing deeper. Put together by hand, one is not written.
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^agent 'a' cannot call 'say': .* nested so deep"):
        _said(deeper)
    with pytest.raises(
            TypeError,
            match="step 1: a trace file does not hold a value nested so "
                  "deep: a line has at most 100 levels"):
        tracefile.dumps(_echo(deeper), _built(deeper))


def test_result_is_written_one_level_deeper_than_an_argument():
    # A result is not inside args, so the line leaves it 99 levels.
    def made(value: Any) -> scheduler.Scenario:
        class Maker:
            def make(self) -> Any:
                return value

        async def caller(ctx: scheduler.AgentContext) -> None:
            await ctx.call("make")

        return scheduler.Scenario(
            name="maker", make_world=Maker, agents={"a": caller},
            invariants=())

    fits = made(_nested(lambda value: [value], 99))
    run = explore.replay(fits, ("a",))
    saved = tracefile.loads(tracefile.dumps(fits, run))
    assert tracefile.replay(fits, saved) == run
    value = _nested(lambda value: [value], 100)
    deeper = made(value)
    with pytest.raises(
            scheduler.UnsupportedValueError,
            match="^'make' ran for agent 'a' and its result is refused: "
                  ".* nested so deep"):
        explore.replay(deeper, ("a",))
    by_hand = scheduler.RunResult(
        (scheduler.Step("a", "make", {}, value),), None)
    with pytest.raises(
            TypeError,
            match="step 1: a trace file does not hold a value nested so "
                  "deep: a line has at most 100 levels of lists and "
                  "objects$"):
        tracefile.dumps(deeper, by_hand)


@pytest.mark.parametrize(
    "empty,most", [([], 97), ({}, 97), ((), 96)],
    ids=["list", "dict", "tuple"])
def test_empty_value_at_the_bottom_takes_its_levels(empty, most):
    # An argument has 98 levels. An empty list or dict takes one and an
    # empty tuple two, as for the reader, so 97 or 96 lists fit around it.
    def around(lists: int) -> Any:
        value: Any = empty
        for _ in range(lists):
            value = [value]
        return value

    fits = around(most)
    saved = tracefile.loads(tracefile.dumps(_echo(fits), _said(fits)))
    assert tracefile.replay(_echo(fits), saved) == _said(fits)
    deeper = around(most + 1)
    with pytest.raises(
            scheduler.UnsupportedValueError, match="nested so deep"):
        _said(deeper)
    with pytest.raises(TypeError, match="step 1: .* nested so deep"):
        tracefile.dumps(_echo(deeper), _built(deeper))


def test_value_nested_far_too_deep_is_not_written():
    # Deeper than Python calls: the answer is the same and still names
    # the step.
    deep = _nested(lambda value: [value], 5000)
    with pytest.raises(TypeError, match="step 1: .* nested so deep"):
        tracefile.dumps(_echo(deep), _built(deep))
    with pytest.raises(
            scheduler.UnsupportedValueError, match="nested so deep"):
        _said(deep)


@pytest.mark.skipif(
    not getattr(sys, "get_int_max_str_digits", lambda: 0)(),
    reason="this Python writes an integer of any length")
def test_integer_too_long_for_json_is_not_written():
    value = 10 ** sys.get_int_max_str_digits()
    with pytest.raises(TypeError, match="step 1: "):
        tracefile.dumps(_echo(value), _said(value))


@pytest.mark.skipif(
    not hasattr(sys, "set_int_max_str_digits"),
    reason="this Python writes an integer of any length")
def test_integer_json_writes_and_cannot_read_is_not_written():
    # With a limit this high Python checks the length of an integer only
    # roughly when it writes one and exactly when it reads one, so json
    # writes 10 ** 20000 and then refuses to read it.
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(20000)
    try:
        value = 10 ** 20000
        with pytest.raises(TypeError, match="step 1: Exceeds the limit"):
            tracefile.dumps(_echo(value), _said(value))
    finally:
        sys.set_int_max_str_digits(limit)


def test_run_whose_message_is_not_a_string_is_not_written():
    # A check that answers False where it means None gives a violation
    # whose message is False. The reader refuses that line, so the writer
    # must not write it.
    scenario = scheduler.Scenario(
        name="echo",
        make_world=_Echo,
        agents={},
        invariants=(
            scheduler.Invariant(
                "holds", lambda world: False, every_step=False),
        ),
    )
    run = explore.replay(scenario, ())
    assert run.violation == scheduler.Violation("holds", False, 0)
    with pytest.raises(
            TypeError,
            match="does not hold False, of type bool, as the violation's "
                  "message"):
        tracefile.dumps(scenario, run)


def test_run_whose_names_are_not_strings_is_not_written():
    async def speaker(ctx: scheduler.AgentContext) -> None:
        await ctx.call(_Name("say"), value=1)

    def scenario(name: Any, agent: Any) -> scheduler.Scenario:
        return scheduler.Scenario(
            name=name, make_world=_Echo, agents={agent: speaker},
            invariants=())

    numbered = scenario("echo", 1)
    with pytest.raises(
            TypeError,
            match="step 1: a trace file does not hold 1, of type int, as "
                  "the agent's name"):
        tracefile.dumps(numbered, explore.replay(numbered, (1,)))
    named = scenario("echo", "a")
    with pytest.raises(
            TypeError,
            match="step 1: a trace file does not hold 'say', of type "
                  "_Name, as the tool's name"):
        tracefile.dumps(named, explore.replay(named, ("a",)))
    unnamed = scenario(None, "a")
    with pytest.raises(
            TypeError,
            match="does not hold None, of type NoneType, as the "
                  "scenario's name"):
        tracefile.dumps(unnamed, _said(1))
    split = scenario("echo", chr(0xD83D) + chr(0xDE00))
    with pytest.raises(TypeError, match="step 1: .* with a surrogate"):
        tracefile.dumps(split, explore.replay(split, tuple(split.agents)))


def test_run_whose_invariant_has_no_string_for_a_name_is_not_written():
    scenario = scheduler.Scenario(
        name="echo",
        make_world=_Echo,
        agents={},
        invariants=(
            scheduler.Invariant(7, lambda world: "broken", every_step=False),
        ),
    )
    run = explore.replay(scenario, ())
    assert run.violation == scheduler.Violation(7, "broken", 0)
    with pytest.raises(
            TypeError,
            match="does not hold 7, of type int, as the invariant's name"):
        tracefile.dumps(scenario, run)


def test_run_that_does_not_replay_is_not_written():
    tickets = itertools.count()

    class Counter:
        def take(self) -> int:
            return next(tickets)

    async def taker(ctx: scheduler.AgentContext) -> None:
        await ctx.call("take")

    scenario = scheduler.Scenario(
        name="leaky", make_world=Counter, agents={"a": taker}, invariants=())
    run = scheduler.run(scenario, scheduler.RandomPolicy(0))
    # The run took ticket 0, and the replay of its text takes ticket 1.
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.dumps(scenario, run)


def test_run_whose_list_outlives_it_replays_otherwise_and_is_not_written():
    shared: list[str] = []

    class Pile:
        """Adds to a list that outlives the run, and returns it."""

        def add(self) -> list[str]:
            shared.append("x")
            return shared

    async def adder(ctx: scheduler.AgentContext) -> None:
        await ctx.call("add")

    scenario = scheduler.Scenario(
        name="pile", make_world=Pile, agents={"a": adder}, invariants=())
    run = scheduler.run(scenario, scheduler.RandomPolicy(0))
    # The run holds the list as its call returned it, with one entry, and
    # a replay holds it with two. So a replay already tells that state
    # outlives a run. The text is written from the first run, and its own
    # replay is the third.
    assert explore.replay(scenario, run.schedule) != run
    with pytest.raises(scheduler.ReplayDivergenceError) as caught:
        tracefile.dumps(scenario, run)
    assert str(caught.value) == (
        'step 1 is {"agent": "a", "tool": "add", "args": {}, '
        '"result": ["x"]} in the file and {"agent": "a", '
        '"tool": "add", "args": {}, "result": ["x", "x", "x"]} in the run')


def test_run_of_another_scenario_is_not_written():
    run = explore.replay(oversell.naive(), _SHRUNK)
    with pytest.raises(
            scheduler.ReplayDivergenceError, match="buyer-3 can still run"):
        tracefile.dumps(oversell.naive(buyers=3), run)
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        tracefile.dumps(oversell.naive(stock=2), run)
