"""Writes a run as the text of a trace file, reads one and replays it.

A trace file is JSON lines, each an object with a kind:

* The first line is of kind "anyorder-trace". It holds the version of the
  format and the name of the scenario.
* One line of kind "step" follows for each tool call, in the order they
  ran. Its agent, tool and args are the schedule entry, and its result is
  what the call returned.
* The last line is of kind "end" and holds the first violation, or null.
  A file without it was cut short.

A value is written as JSON where JSON keeps it apart from every other
value: None, a boolean, an integer, a finite float, a string, a list and a
dict whose keys are strings. A tuple is {"$tuple": [...]}, a tool's error
is {"$raised": {"kind": ..., "message": ...}}, and a dict key that starts
with "$" gets one more. Nothing else is written, so no two values are
written alike. A string that holds a surrogate is not written either: json
writes two surrogates that make a pair as it writes the one character they
stand for. And no line is nested deeper than 100 levels of lists and
objects, so that what a file holds does not depend on how deep the Python
that reads it can go.

A file is a claim about a run, and replay checks the claim: it runs the
schedule the file holds, writes the run it gets and refuses the file unless
every step and the violation are written as the file holds them. What is
compared is what the lines hold as JSON, with an integer kept apart from a
float. Spacing, the order of an object's keys and the spelling of a number
or a string are not part of that: 1.00 holds the float 1.0, and not the
integer 1, and a number with more digits than a float keeps holds the
float nearest to it. Nothing is turned back into a value of a run, so the
only way from a file to a run is to replay it.

The values are those of the calls, as the scheduler takes them: an argument
as its agent gave it and a result as its tool returned it.

Nothing here opens a file. The caller reads and writes the text.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Collection, Mapping
from typing import Any

from anyorder import explore
from anyorder import scheduler
from anyorder import values

VERSION = 1

# The scheduler takes no value nested deeper than a line lets a result be,
# and the line is one level itself.
_MAX_DEPTH = values.ROOM + 1
_OPENING = "anyorder-trace"
_STEP = {"agent": str, "tool": str, "args": dict, "result": object}
_VIOLATION = {"invariant": str, "message": str, "step": int}
_JSON_NAMES = {str: "a string", int: "an integer", dict: "an object"}


class FormatError(ValueError):
    """A text is not a trace file of the version this reader knows."""


@dataclasses.dataclass(frozen=True)
class Saved:
    """What a trace file holds, as the file holds it. Nothing has run.

    The values are JSON's and not a run's: a tuple is held as
    {"$tuple": [...]}. The way to a run is replay.

    Attributes:
        scenario: Name of the scenario the run is said to be a run of.
        steps: The agent, tool, args and result of each step, in order.
        violation: The invariant, message and step of the first violation,
            or None.
    """

    scenario: str
    steps: tuple[Mapping[str, Any], ...]
    violation: Mapping[str, Any] | None

    @property
    def schedule(self) -> tuple[str, ...]:
        """The agent chosen at each step."""
        return tuple(step["agent"] for step in self.steps)


def dumps(scenario: scheduler.Scenario, result: scheduler.RunResult) -> str:
    """Returns the text of a trace file for a run.

    The text is read back and replayed once before it is returned, so no
    text leaves here that its own reader refuses or that did not replay
    here. A scenario that keeps state between runs can still replay
    differently later.

    Args:
        scenario: What the run is a run of.
        result: The run to write. It need not have broken an invariant.

    Raises:
        TypeError: The run is not written: an argument or a result is of
            a type a file does not hold, is nested too deep or holds an
            integer too long for this Python to write, a name or a
            message is not a string, or a string in the run holds a
            surrogate. Nothing is written as the nearest thing a file
            holds.
        ReplayDivergenceError: The text did not replay to the same trace.
    """
    steps, violation = _written(result)
    records = [
        {
            "kind": _OPENING,
            "version": VERSION,
            "scenario": _string("the scenario's name", scenario.name),
        },
        *({"kind": "step", **step} for step in steps),
        {"kind": "end", "violation": violation},
    ]
    text = "".join(f"{json.dumps(record)}\n" for record in records)
    replay(scenario, loads(text))
    return text


def loads(text: str) -> Saved:
    """Reads the text of a trace file.

    Args:
        text: The whole file.

    Returns:
        What the file holds. Nothing has checked it against a scenario.

    Raises:
        FormatError: The text is empty, is of another version, lacks its
            last line, or holds a blank line, a line or a key that this
            version does not define, a key twice or a line nested too
            deep. Nothing unknown is skipped: a reader that skipped a line
            could replay another run and call it the same.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        del lines[-1]
    if not lines:
        raise FormatError("the text is empty")
    # First, and alone: what the other lines may hold depends on it.
    scenario = _opening(_parse(1, lines[0]))
    records = [
        _parse(number, line) for number, line in enumerate(lines[1:], start=2)]
    if not records:
        raise FormatError("the file has no end line")
    steps = tuple(
        _step(number, record)
        for number, record in enumerate(records[:-1], start=2))
    return Saved(scenario, steps, _end(len(lines), records[-1]))


def replay(
    scenario: scheduler.Scenario, saved: Saved
) -> scheduler.RunResult:
    """Runs the schedule a trace file holds and checks the run against it.

    The run is checked once it is over, so a scenario that is not the
    file's can make every call of the schedule before it is refused.

    Args:
        scenario: What to run. A file names its scenario and not what the
            scenario was built with, so the caller brings it.
        saved: What the file holds.

    Returns:
        The run. Written as a trace file, it holds what the file holds.

    Raises:
        ValueError: The file holds a run of a scenario with another name.
            Nothing has run.
        ReplayDivergenceError: The run left the file's schedule, or a step
            or the violation is not written as the file holds it.
        UnsupportedValueError: The scenario makes a call with a value no
            call carries, under this schedule. That ends its run, so
            there is none to hold against the file.
    """
    if saved.scenario != scenario.name:
        raise ValueError(
            f"the file holds a run of {saved.scenario!r}, not of "
            f"{scenario.name!r}")
    result = explore.replay(scenario, saved.schedule)
    try:
        steps, violation = _written(result)
    except TypeError as error:
        # A run that no file holds is not the run this file holds.
        raise scheduler.ReplayDivergenceError(str(error)) from None
    # As many steps on both sides: explore.replay refuses any other run.
    for number, (held, ran) in enumerate(zip(saved.steps, steps), start=1):
        _compare(f"step {number}", held, ran)
    _compare("the violation", saved.violation, violation)
    return result


def _compare(what: str, held: Any, ran: Any) -> None:
    """Refuses a run that is not written as the file holds it.

    As text, with the keys of each object in one order: Python takes True,
    1 and 1.0 for equal, and a file does not.
    """
    if _in_one_order(held) != _in_one_order(ran):
        raise scheduler.ReplayDivergenceError(
            f"{what} is {json.dumps(held)} in the file and "
            f"{json.dumps(ran)} in the run")


def _in_one_order(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _written(
    result: scheduler.RunResult,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Returns a run's steps and its violation as a file holds them.

    Raises:
        TypeError: The run is not written: an argument or a result is of
            a type a file does not hold, is nested too deep or holds an
            integer too long for this Python to write, a name or a
            message is not a string, or a string in the run holds a
            surrogate.
    """
    steps = []
    for number, step in enumerate(result.steps, start=1):
        try:
            steps.append(_written_step(step))
        except (TypeError, ValueError) as error:
            raise TypeError(f"step {number}: {error}") from None
    violation = result.violation
    if violation is None:
        return steps, None
    return steps, {
        "invariant": _string("the invariant's name", violation.invariant),
        "message": _string("the violation's message", violation.message),
        "step": violation.step,
    }


def _written_step(step: scheduler.Step) -> dict[str, Any]:
    held = {
        "agent": _string("the agent's name", step.agent),
        "tool": _string("the tool's name", step.tool),
        # The line is one level itself.
        "args": _encode(dict(step.args), _MAX_DEPTH - 1),
        "result": _encode(step.result, _MAX_DEPTH - 1),
    }
    # json refuses an integer that is too long for it, and it can write
    # one that it then refuses to read. Asked here, the error names the
    # step.
    json.loads(json.dumps(held))
    return held


def _string(what: str, value: Any) -> str:
    """Refuses a name or a message that a file does not hold."""
    if type(value) is not str:
        raise TypeError(
            f"a trace file does not hold {_shown(value)}, of type "
            f"{type(value).__name__}, as {what}")
    return _no_surrogate(value)


def _shown(value: Any) -> str:
    """Returns a value as a refusal names it."""
    try:
        return repr(value)
    except RecursionError:
        # Python cannot show it, and still no file holds it.
        return "a value too deep to show"


def _no_surrogate(value: str) -> str:
    """Refuses a string that holds a surrogate.

    json writes two surrogates that make a pair as it writes the one
    character they stand for, so the two strings would be written alike.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise TypeError(
            f"a trace file does not hold {value!r}, a string with a "
            "surrogate") from None
    return value


def _encode(value: Any, room: int) -> Any:
    """Returns the JSON value a trace file holds for a value.

    Types are matched exactly. An instance of a subclass can compare equal
    to one of them and still be more, and nothing here keeps the rest.

    Args:
        value: An argument or a result, or a part of one.
        room: How many levels of lists and objects the value may take
            before its line is nested too deep.

    Raises:
        TypeError: The file does not hold the value.
    """
    kind = type(value)
    if kind in (type(None), bool, int):
        return value
    if kind is str:
        return _no_surrogate(value)
    if kind is float and math.isfinite(value):
        return value
    if kind is list:
        room = _room_under(room, 1)
        return [_encode(item, room) for item in value]
    if kind is tuple:
        room = _room_under(room, 2)
        return {"$tuple": [_encode(item, room) for item in value]}
    if kind is scheduler.Raised:
        room = _room_under(room, 2)
        return {"$raised": {
            "kind": _encode(value.kind, room),
            "message": _encode(value.message, room),
        }}
    if kind is dict and all(type(key) is str for key in value):
        room = _room_under(room, 1)
        # One more "$", so that no dict is taken for a tuple or an error.
        return {
            _no_surrogate(f"${key}" if key.startswith("$") else key):
                _encode(item, room)
            for key, item in value.items()}
    raise TypeError(
        f"a trace file does not hold {_shown(value)}, of type "
        f"{kind.__name__}")


def _room_under(room: int, levels: int) -> int:
    """Returns the room left to the parts of a value that takes levels."""
    if room < levels:
        raise TypeError(
            "a trace file does not hold a value nested so deep: a line has "
            f"at most {_MAX_DEPTH} levels of lists and objects")
    return room - levels


def _parse(number: int, line: str) -> dict[str, Any]:
    """Reads one line as a JSON object."""
    if not line.strip():
        raise FormatError(f"line {number} is blank")
    try:
        # json does not join a raw half of a pair with its other half, so
        # the line would hold two code points written as one character is.
        line.encode("utf-8")
    except UnicodeEncodeError:
        raise FormatError(
            f"line {number} holds a surrogate that is not escaped") from None
    try:
        record = json.loads(
            line, parse_constant=_finite, parse_float=_finite,
            object_pairs_hook=_once)
    except json.JSONDecodeError as error:
        # Not the error as text: it names a line, and that is always 1.
        raise FormatError(
            f"line {number} cannot be read: {error.msg} (column "
            f"{error.colno})") from None
    except (ValueError, RecursionError) as error:
        raise FormatError(f"line {number} cannot be read: {error}") from None
    if type(record) is not dict:
        raise FormatError(f"line {number} is not a JSON object")
    if _depth(record) > _MAX_DEPTH:
        raise FormatError(
            f"line {number} is nested deeper than {_MAX_DEPTH} levels")
    return record


def _depth(value: Any) -> int:
    """Counts the levels of lists and objects in a JSON value.

    Level by level and not by calling itself: the count must not depend
    on how deep this Python can call.
    """
    depth = 0
    holders = [value]
    while True:
        holders = [
            holder for holder in holders if type(holder) in (list, dict)]
        if not holders:
            return depth
        depth += 1
        holders = [
            part
            for holder in holders
            for part in (
                holder.values() if type(holder) is dict else holder)]


def _finite(text: str) -> float:
    """Reads a number, and refuses those Python reads and JSON lacks."""
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{text} is not a finite number")
    return number


def _once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuses an object that gives a key twice. json would keep the last."""
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"an object gives the key {key!r} twice")
        record[key] = value
    return record


def _opening(record: dict[str, Any]) -> str:
    """Checks the first line and returns the scenario's name."""
    _expect_kind(1, record, _OPENING)
    if "version" not in record:
        raise FormatError("line 1 gives no version")
    version = record["version"]
    # Before the keys: another version may put other keys on this line.
    # By type too: True and 1.0 are equal to 1.
    if type(version) is not int or version != VERSION:
        raise FormatError(
            f"line 1 gives version {json.dumps(version)}, and this reader "
            f"knows version {VERSION}")
    _expect_keys(1, record, ("version", "scenario"))
    return _typed(1, record, {"scenario": str})["scenario"]


def _step(number: int, record: dict[str, Any]) -> dict[str, Any]:
    _expect_kind(number, record, "step")
    _expect_keys(number, record, _STEP)
    return _typed(number, record, _STEP)


def _end(number: int, record: dict[str, Any]) -> dict[str, Any] | None:
    _expect_kind(number, record, "end")
    _expect_keys(number, record, ("violation",))
    violation = record["violation"]
    if violation is None:
        return None
    if type(violation) is not dict or sorted(violation) != sorted(_VIOLATION):
        raise FormatError(
            f"line {number} holds a violation that is neither null nor an "
            f"object with the keys {', '.join(sorted(_VIOLATION))}")
    return _typed(number, violation, _VIOLATION)


def _expect_kind(number: int, record: Mapping[str, Any], kind: str) -> None:
    if record.get("kind") != kind:
        raise FormatError(
            f"line {number} is of kind {json.dumps(record.get('kind'))}, "
            f"not {json.dumps(kind)}")


def _expect_keys(
    number: int, record: Mapping[str, Any], keys: Collection[str]
) -> None:
    expected = sorted(("kind", *keys))
    if sorted(record) != expected:
        raise FormatError(
            f"line {number} holds the keys {', '.join(sorted(record))}, "
            f"not {', '.join(expected)}")


def _typed(
    number: int, record: Mapping[str, Any], kinds: Mapping[str, type]
) -> dict[str, Any]:
    """Returns the fields of a line, each of exactly its type."""
    for key, kind in kinds.items():
        if kind is not object and type(record[key]) is not kind:
            raise FormatError(
                f"line {number} holds {json.dumps(record[key])} as its "
                f"{key}, which is not {_JSON_NAMES[kind]}")
    return {key: record[key] for key in kinds}
