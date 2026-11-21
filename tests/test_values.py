"""Checks which values a call carries and what taking one gives."""

import ast
import collections
import dataclasses
import enum
import inspect
import random
import sys
from collections.abc import Callable
from typing import Any

import pytest

from anyorder import scheduler
from anyorder import values


def _made(rng: random.Random, depth: int = 0) -> Any:
    """Makes a value a call carries, with parts, nested up to four deep."""
    holders = ["list", "tuple", "dict", "error"]
    kinds = ["none", "bool", "int", "float", "str"]
    if depth == 0:
        # Always a value with parts: those without are tested one by one.
        kinds = holders
    elif depth < 4:
        kinds += holders * 2
    kind = rng.choice(kinds)
    if kind == "none":
        return None
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "int":
        return rng.choice([0, 1, -7, 2**70])
    if kind == "float":
        return rng.choice([0.0, -0.0, 1.0, 0.5])
    if kind == "str":
        return rng.choice(["", "a", "$tuple", "caf\u00e9"])
    parts = [_made(rng, depth + 1) for _ in range(rng.randrange(4))]
    if kind == "list":
        return parts
    if kind == "tuple":
        return tuple(parts)
    if kind == "dict":
        # In the order made, which is not the order of the keys.
        return {f"{rng.choice('cba')}{len(parts) - place}": part
                for place, part in enumerate(parts)}
    return scheduler.Raised(
        _made(rng, depth + 1), _made(rng, depth + 1))


def _holders(value: Any) -> list[Any]:
    """Returns every list, dict and error in a value, itself included."""
    found = []
    waiting = [value]
    while waiting:
        part = waiting.pop()
        if type(part) in (list, tuple):
            waiting.extend(part)
        elif type(part) is dict:
            waiting.extend(part.values())
        elif type(part) is scheduler.Raised:
            waiting.extend((part.kind, part.message))
        if type(part) in (list, dict, scheduler.Raised):
            found.append(part)
    return found


def _spoil(value: Any) -> None:
    """Changes every list and dict in a value."""
    for holder in _holders(value):
        if type(holder) is list:
            holder.append("spoilt")
        elif type(holder) is dict:
            holder["spoilt"] = True


def _listed(value: Any) -> list[Any]:
    return [value]


def _nested(wrap: Callable[[Any], Any], times: int, inside: Any = 0) -> Any:
    value = inside
    for _ in range(times):
        value = wrap(value)
    return value


@pytest.mark.parametrize(
    "value",
    [
        None, True, False, 0, -7, 2**70, 0.5, -0.0, "", "caf\u00e9", (),
        (1, "a", None, True), ((1, 2), ("a", (None,))), (0.5, (1.5, ())),
    ],
    ids=repr)
def test_value_that_cannot_change_is_handed_back_as_it_is(value):
    assert values.take(value) is value


@pytest.mark.parametrize("seed", range(200))
def test_taken_value_is_written_alike_and_shares_nothing_that_can_change(
    seed
):
    value = _made(random.Random(seed))
    # As text: Python takes 1, True and 1.0 for equal, and a list for
    # equal to no tuple but a dict for equal whatever the order of its
    # keys. The text keeps all of these apart.
    shown = repr(value)
    taken = values.take(value)
    assert repr(taken) == shown
    theirs = {id(holder) for holder in _holders(value)}
    assert not any(id(holder) in theirs for holder in _holders(taken))
    _spoil(value)
    assert repr(taken) == shown
    again = values.take(taken)
    _spoil(taken)
    assert repr(again) == shown


def test_tuple_with_a_part_that_can_change_is_built_anew():
    value = (1, [2], ({"a": [3]},))
    taken = values.take(value)
    assert taken == value and type(taken) is tuple
    assert taken is not value
    assert taken[1] is not value[1]
    assert taken[2][0] is not value[2][0]
    assert taken[2][0]["a"] is not value[2][0]["a"]


def test_tool_error_is_built_anew():
    # An error is frozen and still an object like another: whoever holds
    # it can go around that. So the taker gets one of its own.
    assert values.Raised is scheduler.Raised
    error = scheduler.Raised("ValueError", "only 1 left")
    taken = values.take(error)
    assert taken == error and type(taken) is scheduler.Raised
    assert taken is not error
    listed = scheduler.Raised("K", ["m"])
    assert values.take(listed) == listed
    assert values.take(listed).message is not listed.message


def test_dict_keeps_the_order_of_its_keys():
    plain = {"b": 1, "a": 2}
    assert list(values.take(plain)) == ["b", "a"]
    assert values.take(plain) is not plain
    holding = {"b": [1], "a": 2}
    assert list(values.take(holding)) == ["b", "a"]


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


# The values tests/test_tracefile.py has the writer refuse, and some more.
_NOT_CARRIED = [
    ({1, 2}, "set"),
    (frozenset(), "frozenset"),
    (b"x", "bytes"),
    (bytearray(b"x"), "bytearray"),
    (object(), "object"),
    (1j, "complex"),
    (range(3), "range"),
    (float("nan"), "float"),
    (float("inf"), "float"),
    (float("-inf"), "float"),
    (_Colour.RED, "_Colour"),
    (_Point(1, 2), "_Point"),
    (_Name("a"), "_Name"),
    (_Stack([1]), "_Stack"),
    (collections.OrderedDict(), "OrderedDict"),
    ([{"a": ({2},)}], "set"),
    ((1, {2}), "set"),
    ((1, float("nan")), "float"),
    ({1: "a"}, "dict"),
    ({_Name("a"): 1}, "dict"),
    ({"a": 1, 2: [3]}, "dict"),
    (_Ratio(0.5), "_Ratio"),
    (_Failure("K", "m"), "_Failure"),
    (scheduler.Raised("K", {1}), "set"),
    (scheduler.Raised(b"K", "m"), "bytes"),
]


@pytest.mark.parametrize("value,kind", _NOT_CARRIED)
def test_value_no_call_carries_is_not_taken(value, kind):
    with pytest.raises(
            values.NotTakenError,
            match=f"^a call does not carry .*, of type {kind}$"):
        values.take(value)


def test_refusal_is_a_type_error():
    # As the refusal of a trace file's writer is.
    assert issubclass(values.NotTakenError, TypeError)


def _answering_as(kind: type) -> type:
    """Makes a class that answers as kind does, through its metaclass."""

    class Liar(type):
        def __eq__(cls, other: object) -> bool:
            return other is kind or type.__eq__(cls, other)

        def __hash__(cls) -> int:
            return hash(kind)

    class _Box(metaclass=Liar):
        pass

    return _Box


def _masked_as(kind: type) -> Any:
    """Makes a value that names kind when it is asked for its class."""

    class _Masked:
        @property
        def __class__(self) -> type:
            return kind

    return _Masked()


class _Agreeable:
    """Equal to whatever it is compared with, None too."""

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


_AROUND = [
    lambda value: value,
    lambda value: [value],
    lambda value: (value,),
    lambda value: {"a": value},
    lambda value: (1, [value]),
]
_AROUND_IDS = ["bare", "list", "tuple", "dict", "deeper"]


@pytest.mark.parametrize("around", _AROUND, ids=_AROUND_IDS)
@pytest.mark.parametrize(
    "kind",
    [str, int, bool, type(None), tuple, list, dict, float, scheduler.Raised],
    ids=lambda kind: kind.__name__)
def test_class_that_answers_as_a_type_a_call_carries_is_not_taken(
    kind, around
):
    # Every type is matched by identity. A set of types, or an ==, takes
    # the word of such a class.
    box = _answering_as(kind)
    assert box == kind and box in {kind} and box is not kind
    with pytest.raises(values.NotTakenError, match=", of type _Box$"):
        values.take(around(box()))


@pytest.mark.parametrize("around", _AROUND, ids=_AROUND_IDS)
def test_value_that_names_another_class_is_not_taken(around):
    # The type is asked of Python and not of the value, which can say what
    # it likes.
    masked = _masked_as(int)
    assert masked.__class__ is int and isinstance(masked, int)
    with pytest.raises(values.NotTakenError, match=", of type _Masked$"):
        values.take(around(masked))


@pytest.mark.parametrize("around", _AROUND, ids=_AROUND_IDS)
def test_value_that_is_equal_to_none_is_not_taken(around):
    assert _Agreeable().__eq__(None)
    with pytest.raises(values.NotTakenError, match=", of type _Agreeable$"):
        values.take(around(_Agreeable()))


@pytest.mark.parametrize(
    "key", [_answering_as(str)(), _masked_as(str)],
    ids=["answers-as-str", "names-str"])
def test_dict_with_a_key_that_passes_for_a_string_is_not_taken(key):
    assert isinstance(key, str) or type(key) == str
    with pytest.raises(values.NotTakenError, match=", of type dict$"):
        values.take({key: 1})


@pytest.mark.parametrize(
    "wrap,most",
    [
        (_listed, 99),
        (lambda value: {"k": value}, 99),
        (lambda value: (value,), 49),
        (lambda value: scheduler.Raised("K", value), 49),
        (lambda value: scheduler.Raised(value, "m"), 49),
    ],
    ids=["list", "dict", "tuple", "error", "kind"])
def test_value_is_taken_as_deep_as_a_file_holds_a_result(wrap, most):
    # A line of a trace file may be nested 100 deep and is one level
    # itself, which leaves 99 to a result: 99 lists or dicts, or 49 tuples
    # or errors, which a file writes with two levels each.
    assert values.ROOM == 99
    fits = _nested(wrap, most)
    assert repr(values.take(fits)) == repr(fits)
    with pytest.raises(
            values.NotTakenError,
            match="^a call does not carry a value nested so deep: its line "
                  "in a trace file would have more than 100 levels of "
                  "lists and objects$"):
        values.take(_nested(wrap, most + 1))


def test_arguments_are_taken_one_level_less_deep_than_a_result():
    # The arguments of a call are a dict, which takes one of the levels.
    fits = {"value": _nested(_listed, 98)}
    assert repr(values.take(fits)) == repr(fits)
    with pytest.raises(values.NotTakenError, match="nested so deep"):
        values.take({"value": _nested(_listed, 99)})


@pytest.mark.parametrize(
    "empty,most", [([], 98), ({}, 98), ((), 97)],
    ids=["list", "dict", "tuple"])
def test_empty_value_at_the_bottom_takes_its_levels(empty, most):
    # An empty list or dict takes one level and an empty tuple two, as a
    # file writes them, so 98 or 97 lists fit around it.
    fits = _nested(_listed, most, empty)
    assert repr(values.take(fits)) == repr(fits)
    with pytest.raises(values.NotTakenError, match="nested so deep"):
        values.take(_nested(_listed, most + 1, empty))


def test_value_that_holds_itself_is_not_taken():
    # A copy must end, and here the bound ends it.
    looped: list[Any] = []
    looped.append(looped)
    keyed: dict[str, Any] = {}
    keyed["me"] = keyed
    first: list[Any] = []
    second = (first,)
    first.append(second)
    for value in (looped, keyed, first, second):
        with pytest.raises(values.NotTakenError, match="nested so deep"):
            values.take(value)


def test_value_nested_far_deeper_than_python_calls_is_not_taken():
    deep = _nested(_listed, sys.getrecursionlimit() * 5)
    with pytest.raises(values.NotTakenError, match="nested so deep"):
        values.take(deep)


@dataclasses.dataclass
class _Link:
    """One link of a chain, as a tool might return one."""

    rest: Any


class _Mute:
    def __repr__(self) -> str:
        raise RuntimeError("no repr today")


class _Dumb:
    def __repr__(self) -> str:
        raise ValueError("not to be shown")


@pytest.mark.parametrize(
    "value,kind",
    [
        (_nested(_Link, 5000, None), "_Link"),
        (_Mute(), "_Mute"),
        ([_Mute()], "_Mute"),
        (_Dumb(), "_Dumb"),
    ],
    ids=["too-deep-to-show", "repr-raises", "inside-a-list", "another-error"])
def test_refused_value_that_cannot_be_shown_is_still_refused(value, kind):
    # The refusal shows the value, and showing this one raises. It is
    # refused all the same, and with the same error, whatever repr raised.
    with pytest.raises(Exception):
        repr(value)
    with pytest.raises(
            values.NotTakenError,
            match="^a call does not carry a value that cannot be shown, "
                  f"of type {kind}$"):
        values.take(value)


class _Halt(BaseException):
    """Not an Exception, as an interrupt is not.

    Not an interrupt itself: pytest shows the values of a failing test, and
    an interrupt raised by a repr would end the whole session.
    """


class _Stopped:
    def __repr__(self) -> str:
        raise _Halt


def test_error_that_is_no_exception_is_not_made_a_refusal():
    # What ends a program, such as an interrupt that arrives while a value
    # is shown, must still end it.
    with pytest.raises(_Halt):
        values.take(_Stopped())


def test_refusal_names_a_class_made_in_a_function_by_its_short_name():
    class Local:
        pass

    with pytest.raises(values.NotTakenError, match=", of type Local$"):
        values.take(Local())


def test_tool_error_is_frozen_and_can_be_hashed():
    error = scheduler.Raised("ValueError", "only 1 left")
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.kind = "KeyError"  # type: ignore[misc]
    # Equal errors are one member of a set, the taken one too.
    assert len({error, values.take(error)}) == 1


@pytest.mark.parametrize(
    "twice",
    [lambda part: [part, part], lambda part: (part, part),
     lambda part: {"a": part, "b": part}, lambda part: [0, [part], part]],
    ids=["list", "tuple", "dict", "deeper"])
def test_part_held_twice_is_copied_once_for_each_place(twice):
    # A file writes the part once for each place, so a taken value in which
    # the two places were one list would be a value that no file tells
    # apart from this one. The price is in the README's limits.
    value = twice(["a"])
    taken = values.take(value)
    assert repr(taken) == repr(value)
    held = [part for part in _holders(taken) if part == ["a"]]
    assert len(held) == 2 and held[0] is not held[1]


def test_values_imports_nothing_of_the_package():
    # Its docstring says so: the scheduler and the trace file both rest on
    # it, and it rests on neither.
    imported = []
    for node in ast.walk(ast.parse(inspect.getsource(values))):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    assert imported
    assert not [
        name for name in imported if name.startswith((".", "anyorder"))]


class _Loud(str):
    """A text that will not be put into another."""

    def __format__(self, spec: str) -> str:
        raise ZeroDivisionError("not in a message")


class _Sly:
    def __repr__(self) -> str:
        return _Loud("sly")


class _Unnamed(type):
    """A metaclass whose classes raise when asked for their name."""

    @property
    def __name__(cls) -> str:
        raise LookupError("no name today")


class _Anon(metaclass=_Unnamed):
    pass


def test_refusal_is_a_refusal_whatever_the_value_says_of_itself():
    # A repr may be of a subclass of str, and a class may say anything
    # when it is asked for its name. Neither has a say in the refusal.
    with pytest.raises(ZeroDivisionError):
        f"{_Sly()!r}"
    with pytest.raises(
            values.NotTakenError,
            match="^a call does not carry a value that cannot be shown, "
                  "of type _Sly$"):
        values.take(_Sly())
    with pytest.raises(LookupError):
        _Anon.__name__
    with pytest.raises(
            values.NotTakenError,
            match="^a call does not carry <.*>, of type _Anon$"):
        values.take(_Anon())


def test_refusal_shows_a_long_value_cut():
    # b and two quotes around the bytes make 200 characters of the first
    # value and 201 of the second.
    whole = b"x" * 197
    with pytest.raises(values.NotTakenError) as caught:
        values.take(whole)
    assert str(caught.value) == (
        f"a call does not carry {whole!r}, of type bytes")
    with pytest.raises(values.NotTakenError) as caught:
        values.take(b"x" * 198)
    assert str(caught.value) == (
        "a call does not carry b'" + "x" * 198 + "... (201 characters), "
        "of type bytes")
    with pytest.raises(values.NotTakenError) as caught:
        values.take(b"x" * 10 ** 6)
    assert len(str(caught.value)) < 300


@pytest.mark.parametrize(
    "value",
    [chr(0xD83D) + chr(0xDE00), chr(0xD83D), ["a" + chr(0xDFFF)],
     10 ** 5000, (10 ** 5000,)],
    ids=["pair", "lone", "in-a-list", "long-integer", "in-a-tuple"])
def test_value_only_a_file_refuses_is_taken(value):
    # No trace file holds a string with a surrogate or an integer too long
    # for Python to write as text. Both are rules of the file, and a call
    # carries the two.
    assert values.take(value) == value
