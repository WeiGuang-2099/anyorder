"""Writes a run as the text of a trace file, reads one and replays it.

A trace file is JSON lines, each an object with a kind:

* The first line is of kind "anyorder-trace". It holds the version of the
  format and the name of the scenario.
* One line of kind "step" follows for each tool call, in the order they
  ran. Its agent, tool and args are the schedule entry, and its result is
  what the call returned.
* One line of kind "draw" stands for each draw, where it was made: after
  the line of the step it followed, or before the first step line. It
  holds the agent that drew, the request it asked to draw from and the
  value it was given. A file says when a draw was made by where its line
  stands, and in no other way.
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
schedule the file holds, gives each draw the file holds back to the agent
that made it, writes the run it gets and refuses the file unless every
step, every draw and the violation are written as the file holds them.
What is compared is what the lines hold as JSON, with an integer kept apart
from a float. Spacing, the order of an object's keys and the spelling of a
number or a string are not part of that: 1.00 holds the float 1.0, and not
the integer 1, and a number with more digits than a float keeps holds the
float nearest to it. A draw is turned back into values of a run, because
its agent must be given what it drew. Nothing else is, so the only way from
a file to a run is to replay it. A draw is given back as the file holds it:
nothing here knows what a request stands for, so a replay does not ask
whether a sampler could have given that value.

The values are those of the calls and the draws, as the scheduler takes
them: an argument as its agent gave it when it awaited the call, and a
result as its tool returned it. That is what version 2 says, and the writer
writes version 2. Version 1 said less. Its values were those the trace held
when the run ended, and a trace then kept a value by reference, so one that
was changed after its call was written with its final content. And it has
no draw line: a text of version 1 that holds one is refused. A text of
version 1 is still read. It replays where its run did not rest on a value
being shared: none was changed after its call, and neither an agent nor a
tool went by a list or a dict that the other still held.

Nothing here opens a file. The caller reads and writes the text.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from anyorder import explore
from anyorder import scheduler
from anyorder import values

VERSION = 2

# The versions this reader knows, each with the kinds of line it has
# between its first and its last. Version 1 has no draw, and says less of
# a value.
_KINDS = {1: ("step",), VERSION: ("step", "draw")}
_VERSIONS = tuple(_KINDS)

# The scheduler takes no value nested deeper than a line lets a result be,
# and the line is one level itself.
_MAX_DEPTH = values.ROOM + 1
_OPENING = "anyorder-trace"
_STEP = {"agent": str, "tool": str, "args": dict, "result": object}
_DRAW = {"agent": str, "request": object, "value": object}
_VIOLATION = {"invariant": str, "message": str, "step": int}
_JSON_NAMES = {str: "a string", int: "an integer", dict: "an object"}


class FormatError(ValueError):
    """A text is not a trace file of a version this reader knows."""


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
        version: The version the file says it is of.
        draws: The agent, request and value of each draw, in order. Each
            also has a step, which is on no line: it is the number of step
            lines the file has before the draw's.
    """

    scenario: str
    steps: tuple[Mapping[str, Any], ...]
    violation: Mapping[str, Any] | None
    version: int = VERSION
    draws: tuple[Mapping[str, Any], ...] = ()

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
        TypeError: The run is not written: an argument or a result, or
            what a draw asked for or was given, is of a type a file does
            not hold, is nested too deep or holds an integer too long for
            this Python to write, a name or a message is not a string, a
            string in the run holds a surrogate, or a draw is out of
            place among the others. Nothing is written as the nearest
            thing a file holds.
        ReplayDivergenceError: The text did not replay to the same trace.
    """
    steps, draws, violation = _written(result)
    records = [
        {
            "kind": _OPENING,
            "version": VERSION,
            "scenario": _string("the scenario's name", scenario.name),
        },
        *_lines(steps, draws),
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
        FormatError: The text is empty, is of a version this reader does
            not know, lacks its last line, or holds a blank line, a line
            or a key that its version does not define, a key twice or a
            line nested too deep. Nothing unknown is skipped: a reader
            that skipped a line could replay another run and call it the
            same.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        del lines[-1]
    if not lines:
        raise FormatError("the text is empty")
    # First, and alone: what the other lines may hold depends on it.
    version, scenario = _opening(_parse(1, lines[0]))
    records = [
        _parse(number, line) for number, line in enumerate(lines[1:], start=2)]
    if not records:
        raise FormatError("the file has no end line")
    steps: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    for number, record in enumerate(records[:-1], start=2):
        if _expect_kind(number, record, *_KINDS[version]) == "step":
            steps.append(_fields(number, record, _STEP))
        else:
            # Where its line stands is all a file says of when it was made.
            draws.append(
                {**_fields(number, record, _DRAW), "step": len(steps)})
    return Saved(
        scenario, tuple(steps), _end(len(lines), records[-1]), version,
        tuple(draws))


def replay(
    scenario: scheduler.Scenario, saved: Saved
) -> scheduler.RunResult:
    """Runs the schedule a trace file holds and checks the run against it.

    Each draw the file holds is given back to the agent that made it. The
    run is checked once it is over, so a scenario that is not the file's
    can make every call of the schedule before it is refused.

    Args:
        scenario: What to run. A file names its scenario and not what the
            scenario was built with, so the caller brings it.
        saved: What the file holds.

    Returns:
        The run. Written as a trace file, it holds what the file holds.

    Raises:
        ValueError: The file holds a run of a scenario with another name.
            Nothing has run.
        ReplayDivergenceError: The run left the file's schedule or its
            draws, or a step, a draw or the violation is not written as
            the file holds it. Or a draw holds the text of no value, and
            then nothing has run.
        UnsupportedValueError: The scenario makes a call with a value no
            call carries, under this schedule. That ends its run, so
            there is none to hold against the file.
    """
    if saved.scenario != scenario.name:
        raise ValueError(
            f"the file holds a run of {saved.scenario!r}, not of "
            f"{scenario.name!r}")
    result = explore.replay(scenario, saved.schedule, _given_back(saved.draws))
    try:
        steps, draws, violation = _written(result)
    except TypeError as error:
        # A run that no file holds is not the run this file holds.
        raise scheduler.ReplayDivergenceError(str(error)) from None
    # As many steps and as many draws on both sides: explore.replay refuses
    # any other run.
    for number, (held, ran) in enumerate(zip(saved.steps, steps), start=1):
        _compare(f"step {number}", held, ran)
    for number, (held, ran) in enumerate(zip(saved.draws, draws), start=1):
        # The sampler held the request against what the agent asked for as
        # Python compares, which takes True for 1.
        _compare(f"draw {number}", held, ran)
    _compare("the violation", saved.violation, violation)
    return result


def _given_back(draws: Sequence[Mapping[str, Any]]) -> list[scheduler.Drawn]:
    """Returns the draws of a file as a run holds them.

    Raises:
        ReplayDivergenceError: What a draw asked for or was given is the
            text of no value, so no run made the draw.
    """
    given = []
    for number, draw in enumerate(draws, start=1):
        parts = {}
        for part in ("request", "value"):
            try:
                parts[part] = _decode(draw[part])
            except ValueError as error:
                raise scheduler.ReplayDivergenceError(
                    f"draw {number}: its {part} holds {error}") from None
        given.append(scheduler.Drawn(
            draw["agent"], parts["request"], parts["value"], draw["step"]))
    return given


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
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None
]:
    """Returns a run's steps, its draws and its violation as a file holds them.

    A draw is returned as Saved holds one, with the number of steps
    before it.

    Raises:
        TypeError: The run is not written: an argument or a result, or
            what a draw asked for or was given, is of a type a file does
            not hold, is nested too deep or holds an integer too long for
            this Python to write, a name or a message is not a string, a
            string in the run holds a surrogate, or a draw is out of
            place among the others.
    """
    steps = []
    for number, step in enumerate(result.steps, start=1):
        try:
            steps.append(_written_step(step))
        except (TypeError, ValueError) as error:
            raise TypeError(f"step {number}: {error}") from None
    draws = []
    earliest = 0
    for number, draw in enumerate(result.draws, start=1):
        try:
            draws.append(_written_draw(draw, earliest, len(steps)))
        except (TypeError, ValueError) as error:
            raise TypeError(f"draw {number}: {error}") from None
        earliest = draw.step
    violation = result.violation
    if violation is None:
        return steps, draws, None
    return steps, draws, {
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


def _written_draw(
    draw: scheduler.Drawn, earliest: int, latest: int
) -> dict[str, Any]:
    """Returns a draw as a file holds it, with the steps before it.

    Args:
        draw: The draw.
        earliest: The steps before the draw that was made before this one.
        latest: The steps of the run.

    Raises:
        TypeError: The draw is not written.
    """
    held = {
        "agent": _string("the agent's name", draw.agent),
        "request": _encode(draw.request, _MAX_DEPTH - 1),
        "value": _encode(draw.value, _MAX_DEPTH - 1),
    }
    # As for a step: the error names the draw.
    json.loads(json.dumps(held))
    if type(draw.step) is not int:
        raise TypeError(
            f"a trace file does not hold {_shown(draw.step)}, of type "
            f"{type(draw.step).__name__}, as the steps before a draw")
    if not earliest <= draw.step <= latest:
        # A file says when a draw was made by where its line stands, and
        # the lines of the draws are in the order of the draws. Put where
        # it does fit, this one would be the draw of another run.
        raise TypeError(
            f"a trace file does not hold a draw made after {draw.step} "
            f"steps, here only after {earliest} to {latest}")
    return {**held, "step": draw.step}


def _lines(
    steps: Sequence[Mapping[str, Any]], draws: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Returns the step and draw lines of a file, each draw where it was made.

    Args:
        steps: The steps as _written gives them.
        draws: The draws as _written gives them: in the order they were
            made, each with the steps before it, and none out of place.
    """
    made: list[list[dict[str, Any]]] = [[] for _ in range(len(steps) + 1)]
    for draw in draws:
        made[draw["step"]].append(
            {"kind": "draw", **{key: draw[key] for key in _DRAW}})
    lines: list[dict[str, Any]] = []
    for before, step in enumerate(steps):
        lines += made[before]
        lines.append({"kind": "step", **step})
    return lines + made[len(steps)]


def _decode(held: Any) -> Any:
    """Returns the value that a JSON value of a file stands for.

    The way back from _encode, for what _encode writes and nothing else.

    Raises:
        ValueError: No value is written so. An object has a key that
            starts with one "$", and it is neither a tuple nor an error
            as they are written.
    """
    kind = type(held)
    if kind is list:
        return [_decode(part) for part in held]
    if kind is not dict:
        return held
    if not any(_is_tag(key) for key in held):
        # One "$" less: the writer gave such a key one more.
        return {
            key[1:] if key.startswith("$") else key: _decode(part)
            for key, part in held.items()}
    if list(held) == ["$tuple"] and type(held["$tuple"]) is list:
        return tuple(_decode(part) for part in held["$tuple"])
    if list(held) == ["$raised"]:
        error = held["$raised"]
        if type(error) is dict and sorted(error) == ["kind", "message"]:
            return scheduler.Raised(
                _decode(error["kind"]), _decode(error["message"]))
    raise ValueError(f"{json.dumps(held)}, which is the text of no value")


def _is_tag(key: str) -> bool:
    """Says whether a key marks an object as a tuple, an error or the like.

    A key of a dict that starts with "$" is written with one more, so it
    starts with two.
    """
    return key.startswith("$") and not key.startswith("$$")


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


def _opening(record: dict[str, Any]) -> tuple[int, str]:
    """Checks the first line and returns the version and the scenario's name."""
    _expect_kind(1, record, _OPENING)
    if "version" not in record:
        raise FormatError("line 1 gives no version")
    version = record["version"]
    # Before the keys: another version may put other keys on this line.
    # By type too: True and 1.0 are equal to 1.
    if type(version) is not int or version not in _VERSIONS:
        raise FormatError(
            f"line 1 gives version {json.dumps(version)}, and this reader "
            f"knows versions {' and '.join(map(str, _VERSIONS))}")
    _expect_keys(1, record, ("version", "scenario"))
    return version, _typed(1, record, {"scenario": str})["scenario"]


def _fields(
    number: int, record: dict[str, Any], kinds: Mapping[str, type]
) -> dict[str, Any]:
    """Returns the fields of a step line or a draw line.

    Args:
        number: The line's number.
        record: The line.
        kinds: The keys a line of its kind has, each with its type.
    """
    _expect_keys(number, record, kinds)
    return _typed(number, record, kinds)


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


def _expect_kind(number: int, record: Mapping[str, Any], *kinds: str) -> str:
    """Returns the kind of a line, which must be one of those given."""
    kind = record.get("kind")
    if kind not in kinds:
        raise FormatError(
            f"line {number} is of kind {json.dumps(kind)}, not "
            f"{' or '.join(map(json.dumps, kinds))}")
    return kind


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
