"""The values a call carries, and how one is taken.

A call carries None, booleans, integers, finite floats, strings, lists,
tuples, dicts whose keys are strings and tool errors, with parts of the
same kinds. Those are the kinds of value a trace file holds, and a file
holds every such value but two: a string with a surrogate, and an integer
too long for Python to write. A type is matched exactly, as a trace file
matches it: an instance of a subclass can compare equal to one of them and
still be more.

take gives a value as it is at that moment, apart from whoever gave it. A
list or a dict is copied, at any depth, so nothing done to the value
afterwards shows in what was taken. A string, a number or a tuple of such
cannot change and is handed back as it is. A tool error is built anew: it
is frozen, and whoever holds one can still go around that.

A value is nested at most as deep as a line of a trace file lets a result
be. The bound is here as well as in the file: a copy must end on a list
that holds itself, and it must end at the same depth under every Python.
take calls itself once for each level, so a value near the bound needs
about a hundred calls of Python's stack, and twice as many before Python
3.12: taken from deep inside other calls it can end in a RecursionError.
key needs about two calls for each level under every Python.

key gives a hashable that stands for a value as a trace file writes it:
two values have one key exactly when a file writes them alike. Python
takes True, 1 and 1.0 for equal, and two dicts that hold the same under
keys in another order, and an agent that is given one can go on otherwise
than an agent that is given the other. So whatever looks a value up, such
as a search that asks whether it has stood here before, goes by the key
and not by the value. A key is finer than the reader of a file in one
thing: a file is held against a run with no eye for the order of a dict's
keys, and a key keeps that order.

Nothing here imports the rest of the package.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Hashable, Iterable
from typing import Any

# A line of a trace file is nested at most 100 levels of lists and objects
# and is one level itself. That leaves 99 to a result and to the dict of
# the arguments. A list or a dict takes one level, and a tuple or an error
# two, as a file writes them.
_LINE_LEVELS = 100
ROOM = _LINE_LEVELS - 1

# A refusal shows the value it refuses, and no more of it than this.
_SHOWN = 200

# The name of a class as type itself holds it, whatever the class or its
# metaclass says its name is.
_held_name = type.__dict__["__name__"].__get__


def name_of(kind: type) -> str:
    """Returns the name of a class, for a refusal or a tool's error to give.

    As type itself holds it, and as a str and nothing more: a class can
    be made with a name of a subclass of str, which would have its own
    say in how it is put into a message.
    """
    name = _held_name(kind)
    return name if type(name) is str else str.__str__(name)


class NotTakenError(TypeError):
    """A value is not one a call carries."""


@dataclasses.dataclass(frozen=True)
class Raised:
    """What a call gives its agent when the tool raised.

    A value and not the exception: two exceptions alike do not compare
    equal, and a run must compare equal to its replay.

    Attributes:
        kind: Name of the exception's class.
        message: The exception as text.
    """

    kind: str
    message: str


def take(value: Any, room: int = ROOM) -> Any:
    """Returns a value as it is now, apart from whoever gave it.

    Args:
        value: An argument, a result or a part of one.
        room: How many levels the value may take, a list or a dict taking
            one and a tuple or an error two. It is how take counts for
            the parts of a value: a caller that gives more than ROOM
            gets values that no trace file holds.

    Returns:
        The value itself if nothing in it can change. Otherwise a copy
        that shares no list, no dict and no error with it.

    Raises:
        NotTakenError: The value or a part of it is not one a call
            carries: it is of another type, a float that is not finite
            or a dict with a key that is not a string, or it is nested
            too deep.
    """
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    # A tuple, a list or a dict whose parts are all plain is looked
    # through in one pass, with no call of take for each part: most are
    # like that, and this runs at every call.
    if kind is tuple:
        if room < 2:
            raise _too_deep()
        if _all_plain(value):
            return value
        parts = [take(part, room - 2) for part in value]
        if all(map(operator.is_, parts, value)):
            return value
        return tuple(parts)
    if kind is list:
        if room < 1:
            raise _too_deep()
        if _all_plain(value):
            return value.copy()
        return [take(part, room - 1) for part in value]
    if kind is dict and _all_strings(value):
        if room < 1:
            raise _too_deep()
        if _all_plain(value.values()):
            return value.copy()
        return {key: take(part, room - 1) for key, part in value.items()}
    if kind is float and math.isfinite(value):
        return value
    if kind is Raised:
        if room < 2:
            raise _too_deep()
        return Raised(
            take(value.kind, room - 2), take(value.message, room - 2))
    raise NotTakenError(
        f"a call does not carry {shown(value)}, of type {name_of(kind)}")


def key(value: Any, room: int = ROOM) -> Hashable:
    """Returns a hashable that stands for a value as a trace file writes it.

    Two values have equal keys exactly when a file writes them alike. So
    True, 1 and 1.0 have three keys and 0.0 and -0.0 have two, where Python
    takes each of those for equal. So have two dicts that hold the same
    under keys in another order: a dict is handed on in the order it was
    given, and whoever is handed it can go by that. A list and the tuple of
    its parts have two keys as well.

    Args:
        value: A value a call carries, or a part of one.
        room: How many levels the value may take, as for take.

    Returns:
        The value itself for a string, an integer and None. Otherwise a
        tuple that starts with the kind of the value and shares nothing
        with it that can change.

    Raises:
        NotTakenError: The value or a part of it is not one a call
            carries, or it is nested too deep.
    """
    kind = type(value)
    if kind is str or kind is int or value is None:
        return value
    if kind is bool:
        return ("bool", value)
    if kind is float and math.isfinite(value):
        # Written out in full: the two zeros are equal and are two values.
        return ("float", value.hex())
    if kind is list:
        left = _left(room, 1)
        return ("list", tuple(key(part, left) for part in value))
    if kind is tuple:
        left = _left(room, 2)
        return ("tuple", tuple(key(part, left) for part in value))
    if kind is dict and _all_strings(value):
        left = _left(room, 1)
        return ("dict", tuple(
            (name, key(part, left)) for name, part in value.items()))
    if kind is Raised:
        left = _left(room, 2)
        return ("raised", key(value.kind, left), key(value.message, left))
    raise NotTakenError(
        f"a call does not carry {shown(value)}, of type {name_of(kind)}")


def _left(room: int, levels: int) -> int:
    """Returns the room left to the parts of a value that takes levels."""
    if room < levels:
        raise _too_deep()
    return room - levels


def _all_plain(parts: Iterable[Any]) -> bool:
    """Says whether every part is None, a boolean, an integer or a string.

    Each type is matched by identity. A set of types would be quicker to
    ask and would take the word of a class that answers as int does,
    through its metaclass.
    """
    for part in parts:
        kind = type(part)
        if not (kind is str or kind is int or kind is bool or part is None):
            return False
    return True


def _all_strings(keys: Iterable[Any]) -> bool:
    for key in keys:
        if type(key) is not str:
            return False
    return True


def _too_deep() -> NotTakenError:
    return NotTakenError(
        "a call does not carry a value nested so deep: its line in a trace "
        f"file would have more than {_LINE_LEVELS} levels of lists and objects")


def shown(value: Any) -> str:
    """Returns a value as a refusal names it, cut where it is long.

    For any value, also one that no call carries.
    """
    try:
        shown = repr(value)
    except Exception:
        # Python cannot show it, as an integer too long to write, or its
        # own repr raises. The refusal must not be another error.
        return "a value that cannot be shown"
    if type(shown) is not str:
        # A repr can be of a subclass of str, which has its own say in
        # how it is put into a message.
        return "a value that cannot be shown"
    if len(shown) > _SHOWN:
        return f"{shown[:_SHOWN]}... ({len(shown)} characters)"
    return shown
