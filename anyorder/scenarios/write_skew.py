"""Doctors going off call, where one of them must stay.

A rota has an entry for each doctor: whether the doctor is on call, and a
version that goes up by one with every write of that entry. A doctor may
leave while another doctor is on call. The naive doctor reads the rota and,
if it may, takes itself off call in a second tool call, so two doctors can
each see the other on call, and both leave. The compare-and-set doctor
leaves only if its own entry still has the version it read. That fix fails
where the naive doctor fails: the doctors write different entries, so no
write is refused. The atomic doctor makes one call that looks at the other
entries and writes its own. The three scenarios differ only in the doctor,
which makes the atomic one the control: a failure reported there is a bug in
the tester.

The rota is read whole, in one call, so these scenarios do not show a doctor
that reads the entries one at a time.

The invariant asks that somebody is on call. That is a rule for a rota that
starts with somebody on call, and a scenario is built with one doctor at
least.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable

from anyorder import scheduler

# A rota as a read gives it: for each doctor the name, the version of the
# entry and whether the doctor is on call.
Entries = tuple[tuple[str, int, bool], ...]


class Rota:
    """Who is on call, with a version for each doctor's entry.

    The version of an entry goes up by one with every write of it, whether
    or not the write changes who is on call. A read gives tuples of its
    own, so what it returned stays what was read.
    """

    def __init__(self, people: Iterable[str]) -> None:
        # By name: the version, and whether the doctor is on call.
        self.entries: dict[str, tuple[int, bool]] = {
            name: (0, True) for name in sorted(people)}

    def read(self) -> Entries:
        """Gives every entry in name order: name, version, on call."""
        return tuple((name, *entry) for name, entry in self.entries.items())

    def leave(self, person: str) -> None:
        """Takes the person off call, whoever else is on call."""
        version, _ = self.entries[person]
        self.entries[person] = (version + 1, False)

    def leave_if(self, person: str, version: int) -> bool:
        """Takes the person off call if their entry still has this version."""
        if self.entries[person][0] != version:
            return False
        self.leave(person)
        return True

    def leave_if_covered(self, person: str) -> bool:
        """Takes the person off call if somebody else is on call.

        The check and the write are one call, so no other call comes
        between them.
        """
        if not any(
                on_call for name, (_, on_call) in self.entries.items()
                if name != person):
            return False
        self.leave(person)
        return True


def _may_leave(rota: Entries, doctor: str) -> bool:
    """Says whether the rota as it was read lets this doctor leave."""
    on_call = {name for name, _, on in rota if on}
    return doctor in on_call and on_call != {doctor}


def _version(rota: Entries, doctor: str) -> int:
    """Gives the version of this doctor's entry in the rota as it was read."""
    return next(version for name, version, _ in rota if name == doctor)


async def naive_doctor(ctx: scheduler.AgentContext) -> None:
    rota = await ctx.call("read")
    if _may_leave(rota, ctx.name):
        await ctx.call("leave", person=ctx.name)


async def cas_doctor(ctx: scheduler.AgentContext) -> None:
    # A write is refused only when the doctor's own entry was written since
    # the read. No doctor writes the entry of another, so that does not
    # happen in these scenarios: the second read is for a rota that others
    # write as well.
    while True:
        rota = await ctx.call("read")
        if not _may_leave(rota, ctx.name):
            return
        left = await ctx.call(
            "leave_if", person=ctx.name, version=_version(rota, ctx.name))
        # Only True is a write that went through: a tool that raised gives
        # its doctor a value that is true as well.
        if left is True:
            return


async def atomic_doctor(ctx: scheduler.AgentContext) -> None:
    await ctx.call("leave_if_covered", person=ctx.name)


def someone_on_call(rota: Rota) -> str | None:
    """Fails when nobody is on call."""
    if any(on_call for _, on_call in rota.entries.values()):
        return None
    return "nobody is on call"


def snapshot(rota: Rota) -> Entries:
    """Gives the state a run leaves: every entry, in name order."""
    # From the entries and not by a call of read: a snapshot is no tool
    # call, and a world may count the calls it is given.
    return tuple((name, *entry) for name, entry in rota.entries.items())


def naive(doctors: int = 2) -> scheduler.Scenario:
    """Doctors that read the rota and leave in separate calls.

    Raises:
        ValueError: Fewer than 1 doctor was asked for.
    """
    return _scenario("write-skew-naive", naive_doctor, doctors)


def cas(doctors: int = 2) -> scheduler.Scenario:
    """A fix that fails: doctors that leave with a compare-and-set.

    Raises:
        ValueError: Fewer than 1 doctor was asked for.
    """
    return _scenario("write-skew-cas", cas_doctor, doctors)


def atomic(doctors: int = 2) -> scheduler.Scenario:
    """The control: doctors that check the rota and leave in one call.

    Raises:
        ValueError: Fewer than 1 doctor was asked for.
    """
    return _scenario("write-skew-atomic", atomic_doctor, doctors)


def _scenario(
    name: str, doctor: scheduler.Agent, doctors: int
) -> scheduler.Scenario:
    if doctors < 1:
        # With nobody on the rota the invariant fails before any call, in
        # the control as well.
        raise ValueError(f"a rota needs at least 1 doctor, not {doctors}")
    names = tuple(f"doctor-{i}" for i in range(1, doctors + 1))
    return scheduler.Scenario(
        name=name,
        make_world=functools.partial(Rota, names),
        agents=dict.fromkeys(names, doctor),
        invariants=(
            scheduler.Invariant(
                "someone-on-call", someone_on_call, every_step=True),
        ),
        snapshot=snapshot,
    )
