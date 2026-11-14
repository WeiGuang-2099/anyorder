"""Editors adding a line each to one document.

The naive editor reads the document and writes it back with its own line
added, in separate tool calls, so two editors can both start from the same
lines, and the write of the second then drops the line of the first. The
compare-and-set editor writes only if the document still has the version it
read, and reads again when that write is refused. The two scenarios differ
only in the editor, which makes the compare-and-set one the control: a
failure reported there is a bug in the tester.

The invariant asks that no line the document once held is gone, and knows a
line by its text. That is a rule for editors that only add lines, each a
line of its own, as these do. It does not see an editor that stops after a
refused write: no line is lost, one was never added. The
serial-equivalence oracle sees that.
"""

from __future__ import annotations

from collections.abc import Sequence

from anyorder import scheduler


class Document:
    """Lines, and a version that goes up by one with every write.

    The lines are a tuple that is replaced and never changed, so what a
    read returned stays what was read.
    """

    def __init__(self) -> None:
        self.version = 0
        self.lines: tuple[str, ...] = ()
        # Every line a write stored, once, in the order they came. An
        # invariant is asked about the world alone and keeps no memory.
        self.held: tuple[str, ...] = ()

    def read(self) -> tuple[int, tuple[str, ...]]:
        return self.version, self.lines

    def write(self, lines: Sequence[str]) -> None:
        """Stores the lines, whatever the version is."""
        self.lines = tuple(lines)
        for line in self.lines:
            if line not in self.held:
                self.held += (line,)
        self.version += 1

    def write_if(self, version: int, lines: Sequence[str]) -> bool:
        """Stores the lines if the document still has this version."""
        if self.version != version:
            return False
        self.write(lines)
        return True


async def naive_editor(ctx: scheduler.AgentContext) -> None:
    _, lines = await ctx.call("read")
    await ctx.call("write", lines=(*lines, ctx.name))


async def cas_editor(ctx: scheduler.AgentContext) -> None:
    # A write is refused only when another editor's write went through
    # since the read, and each editor's goes through once, so this ends on
    # a document that does not raise.
    while True:
        version, lines = await ctx.call("read")
        wrote = await ctx.call(
            "write_if", version=version, lines=(*lines, ctx.name))
        # Only True is a write that went through: a tool that raised gives
        # its editor a value that is true as well.
        if wrote is True:
            return


def no_line_lost(document: Document) -> str | None:
    """Fails when a line the document once held is no longer in it."""
    lost = [line for line in document.held if line not in document.lines]
    if not lost:
        return None
    gone = f"the document held {', '.join(lost)}"
    if not document.lines:
        return f"{gone} and is now empty"
    return f"{gone} and now holds only {', '.join(document.lines)}"


def snapshot(document: Document) -> tuple[int, tuple[str, ...]]:
    """Gives the state a run leaves: the version and the lines in order."""
    return document.version, document.lines


def naive(editors: int = 2) -> scheduler.Scenario:
    """Editors that read and write in separate calls."""
    return _scenario("lost-update-naive", naive_editor, editors)


def cas(editors: int = 2) -> scheduler.Scenario:
    """The control: editors that write with a compare-and-set."""
    return _scenario("lost-update-cas", cas_editor, editors)


def _scenario(
    name: str, editor: scheduler.Agent, editors: int
) -> scheduler.Scenario:
    return scheduler.Scenario(
        name=name,
        make_world=Document,
        agents={f"editor-{i}": editor for i in range(1, editors + 1)},
        invariants=(
            scheduler.Invariant("no-line-lost", no_line_lost, every_step=True),
        ),
    )
