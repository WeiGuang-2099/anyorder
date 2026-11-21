"""Checks the write skew scenario, its fix that fails and its control.

The rota's tools come first, then the naive doctors with their counts
worked out by hand, the compare-and-set doctors, who fail in the same runs,
and the control. After them come shrinking, trace files and worlds and
agents of the tests' own: a rota that starts otherwise, a rota with one
more tool, agents that write a doctor's entry, rotas that raise, a rota
that lets nobody leave and two doctors that are wrong controls.
"""

import collections
import dataclasses
import itertools
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from anyorder import explore
from anyorder import oracle
from anyorder import scheduler
from anyorder import shrink
from anyorder import tracefile
from anyorder.scenarios import write_skew

_D1, _D2, _D3 = "doctor-1", "doctor-2", "doctor-3"
_ALL_ON_CALL = ((_D1, 0, True), (_D2, 0, True))
_BOTH_READ_FIRST = (_D1, _D2, _D1, _D2)
# One doctor reads, the other runs to its end, the first leaves. The file
# of the second of these is written out by hand below: line 1 opens it,
# lines 2 to 5 are the four calls, and line 6 holds the violation, seen at
# the fourth call, where doctor-2 leaves on what it read.
_SMALLEST = {(_D1, _D2, _D2, _D1), (_D2, _D1, _D1, _D2)}
_SHRUNK = (_D2, _D1, _D1, _D2)
_SHRUNK_LINES = (
    '{"kind": "anyorder-trace", "version": 2, '
    '"scenario": "write-skew-naive"}',
    '{"kind": "step", "agent": "doctor-2", "tool": "read", "args": {}, '
    '"result": {"$tuple": [{"$tuple": ["doctor-1", 0, true]}, '
    '{"$tuple": ["doctor-2", 0, true]}]}}',
    '{"kind": "step", "agent": "doctor-1", "tool": "read", "args": {}, '
    '"result": {"$tuple": [{"$tuple": ["doctor-1", 0, true]}, '
    '{"$tuple": ["doctor-2", 0, true]}]}}',
    '{"kind": "step", "agent": "doctor-1", "tool": "leave", '
    '"args": {"person": "doctor-1"}, "result": null}',
    '{"kind": "step", "agent": "doctor-2", "tool": "leave", '
    '"args": {"person": "doctor-2"}, "result": null}',
    '{"kind": "end", "violation": {"invariant": "someone-on-call", '
    '"message": "nobody is on call", "step": 4}}',
)
# One doctor after the other. The file of this run is written out as well:
# its last call is a read made after a write, so it holds an entry that
# was written. The two other tools that write do so through leave.
_SERIAL = (_D1, _D1, _D2)
_SERIAL_LINES = (
    '{"kind": "anyorder-trace", "version": 2, '
    '"scenario": "write-skew-naive"}',
    '{"kind": "step", "agent": "doctor-1", "tool": "read", "args": {}, '
    '"result": {"$tuple": [{"$tuple": ["doctor-1", 0, true]}, '
    '{"$tuple": ["doctor-2", 0, true]}]}}',
    '{"kind": "step", "agent": "doctor-1", "tool": "leave", '
    '"args": {"person": "doctor-1"}, "result": null}',
    '{"kind": "step", "agent": "doctor-2", "tool": "read", "args": {}, '
    '"result": {"$tuple": [{"$tuple": ["doctor-1", 1, false]}, '
    '{"$tuple": ["doctor-2", 0, true]}]}}',
    '{"kind": "end", "violation": null}',
)


class _StepLimitError(scheduler.SchedulerError):
    """A run made more calls than its test allows."""


class _Limited:
    """A world that refuses the call after the last one it allows.

    A doctor that handles a refusal wrongly, or whose tool raises every
    time, may never end. The scheduler passes its own errors on and gives
    them to no agent, so such a run fails its test where it would hang it.
    """

    def __init__(self, world: Any, calls: int) -> None:
        self._world = world
        self._left = calls

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._world, name)
        if not callable(value):
            return value

        def tool(**args: Any) -> Any:
            if not self._left:
                raise _StepLimitError(f"no call is left for {name!r}")
            self._left -= 1
            return value(**args)

        return tool


def _limited(scenario: scheduler.Scenario, calls: int) -> scheduler.Scenario:
    """The scenario with a world that allows this many calls in a run."""
    return dataclasses.replace(
        scenario, make_world=lambda: _Limited(scenario.make_world(), calls))


def _fix(doctors: int = 2) -> scheduler.Scenario:
    """The fix that fails, allowed the most calls its doctors can need.

    A write is refused only when somebody else wrote the doctor's entry,
    which no doctor does. So each doctor reads once and leaves once at
    most: 2 n calls.
    """
    return _limited(write_skew.cas(doctors=doctors), 2 * doctors)


def _control(doctors: int = 2) -> scheduler.Scenario:
    """The control, allowed the one call each of its doctors makes."""
    return _limited(write_skew.atomic(doctors=doctors), doctors)


class _RaisesOnce(write_skew.Rota):
    """A rota whose first compare-and-set raises and writes nothing."""

    def __init__(self, people: Iterable[str]) -> None:
        super().__init__(people)
        self.raised = False

    def leave_if(self, person: str, version: int) -> bool:
        if not self.raised:
            self.raised = True
            raise RuntimeError("try again")
        return super().leave_if(person, version)


class _AlwaysRaises(write_skew.Rota):
    """A rota whose compare-and-set raises every time."""

    def leave_if(self, person: str, version: int) -> bool:
        raise RuntimeError("try again")


class _Renewable(write_skew.Rota):
    """A rota with one more tool: a write that takes nobody off call."""

    def renew(self, person: str) -> None:
        version, on_call = self.entries[person]
        self.entries[person] = (version + 1, on_call)


class _LetsNobodyLeave(write_skew.Rota):
    """A rota that declines whoever asks."""

    def leave_if_covered(self, person: str) -> bool:
        return False


async def _registrar(ctx: scheduler.AgentContext) -> None:
    """Sends doctor-1 home."""
    await ctx.call("leave", person=_D1)


async def _clerk(ctx: scheduler.AgentContext) -> None:
    """Renews the entry of doctor-2 three times."""
    for _ in range(3):
        await ctx.call("renew", person=_D2)


async def _defiant_doctor(ctx: scheduler.AgentContext) -> None:
    """Leaves when it is declined."""
    if await ctx.call("leave_if_covered", person=ctx.name) is not True:
        await ctx.call("leave", person=ctx.name)


async def _waiting_doctor(ctx: scheduler.AgentContext) -> None:
    """Asks again until it may leave."""
    while await ctx.call("leave_if_covered", person=ctx.name) is not True:
        pass


def _on(
    scenario: scheduler.Scenario, make_world: Callable[[], Any]
) -> scheduler.Scenario:
    """The scenario on another world."""
    return dataclasses.replace(
        scenario, name="write-skew-test", make_world=make_world)


def _with_doctor_3_off_call(scenario: scheduler.Scenario) -> scheduler.Scenario:
    """The scenario on a rota where doctor-3 has left before the run."""
    def make_world() -> write_skew.Rota:
        rota = scenario.make_world()
        rota.leave(_D3)
        return rota

    return _on(scenario, make_world)


def _staffed_by(doctor: scheduler.Agent) -> scheduler.Scenario:
    """The control of two doctors with another doctor in place of its own."""
    return _limited(
        dataclasses.replace(
            write_skew.atomic(), name="write-skew-wrong",
            agents={_D1: doctor, _D2: doctor}),
        20)


def _one_stays(names: Iterable[str], stays: str) -> write_skew.Entries:
    """The state where every doctor but one has left, once."""
    return tuple(
        (name, 0, True) if name == stays else (name, 1, False)
        for name in names)


def _orders_ending_with(
    names: Iterable[str], last: str
) -> tuple[oracle.Order, ...]:
    """The serial orders in which this doctor comes last."""
    return tuple(
        order for order in itertools.permutations(names) if order[-1] == last)


def _runs(
    scenario: scheduler.Scenario, limit: int = 1000
) -> list[scheduler.RunResult]:
    """Runs a whole exhaustive search, up to a limit."""
    search = explore.exhaust(scenario)
    runs = list(itertools.islice(search, limit))
    assert next(search, None) is None, f"more than {limit} runs"
    return runs


def _broke(runs: list[scheduler.RunResult]) -> list[scheduler.RunResult]:
    """Keeps the runs that broke the invariant."""
    return [run for run in runs if run.violation is not None]


def _failed(
    scenario: scheduler.Scenario, runs: list[scheduler.RunResult]
) -> list[scheduler.RunResult]:
    """Keeps the runs that end as no serial order ends."""
    serial = oracle.serial_outcomes(scenario, write_skew.snapshot)
    return [run for run in runs if oracle.judge(serial, run).failed]


def _calls(run: scheduler.RunResult) -> list[tuple[str, Any, Any]]:
    """The tool, the arguments and the result of each step."""
    return [(step.tool, step.args, step.result) for step in run.steps]


def test_read_gives_every_entry_of_that_moment_in_name_order():
    rota = write_skew.Rota(["b", "c", "a"])
    before = rota.read()
    rota.leave("b")
    assert before == (("a", 0, True), ("b", 0, True), ("c", 0, True))
    assert rota.read() == (("a", 0, True), ("b", 1, False), ("c", 0, True))
    # The order is that of the names as strings.
    mixed = write_skew.Rota(["doctor-2", "doctor-10"])
    assert mixed.read() == (("doctor-10", 0, True), ("doctor-2", 0, True))


def test_leave_writes_the_entry_of_the_person_whoever_is_on_call():
    rota = write_skew.Rota([_D1, _D2])
    assert rota.leave(_D1) is None
    assert rota.read() == ((_D1, 1, False), (_D2, 0, True))
    # A write that changes nothing is a write: the version goes up.
    rota.leave(_D1)
    assert rota.read() == ((_D1, 2, False), (_D2, 0, True))
    # The last doctor on call may leave as well.
    rota.leave(_D2)
    assert rota.read() == ((_D1, 2, False), (_D2, 1, False))


def test_leave_if_looks_at_the_version_of_the_entry_it_writes_and_no_other():
    rota = write_skew.Rota([_D1, _D2])
    rota.leave(_D2)
    # Not at the version of another entry: 1 is that of doctor-2.
    assert rota.leave_if(_D1, 1) is False
    # The entry of doctor-2 was written and that of doctor-1 was not, so
    # the write goes through, and nobody is on call.
    assert rota.leave_if(_D1, 0) is True
    assert rota.read() == ((_D1, 1, False), (_D2, 1, False))
    # Not at a version the entry had before, nor at one it has not
    # reached.
    assert rota.leave_if(_D1, 0) is False
    assert rota.leave_if(_D1, 2) is False
    assert rota.read() == ((_D1, 1, False), (_D2, 1, False))
    assert rota.leave_if(_D1, 1) is True
    assert rota.read() == ((_D1, 2, False), (_D2, 1, False))


def test_leave_if_covered_declines_the_last_doctor_on_call():
    rota = write_skew.Rota([_D1, _D2, _D3])
    assert rota.leave_if_covered(_D2) is True
    assert rota.leave_if_covered(_D3) is True
    # doctor-1 is on call, and does not cover for itself. A call that is
    # declined writes nothing.
    assert rota.leave_if_covered(_D1) is False
    assert rota.read() == ((_D1, 0, True), (_D2, 1, False), (_D3, 1, False))
    # The tool looks at the rota and does not count who asked before: here
    # doctor-3 left by another tool.
    other = write_skew.Rota([_D1, _D2, _D3])
    other.leave(_D3)
    assert other.leave_if_covered(_D1) is True
    assert other.leave_if_covered(_D2) is False
    # Somebody who is off call and asks while another is on call is
    # written again, as by leave.
    assert other.leave_if_covered(_D1) is True
    assert other.read() == ((_D1, 2, False), (_D2, 0, True), (_D3, 1, False))


def test_invariant_asks_that_somebody_is_on_call():
    rota = write_skew.Rota([_D1, _D2])
    assert write_skew.someone_on_call(rota) is None
    rota.leave(_D2)
    assert write_skew.someone_on_call(rota) is None
    rota.leave(_D1)
    assert write_skew.someone_on_call(rota) == "nobody is on call"
    # Nor is anybody on call on a rota with nobody on it.
    assert write_skew.someone_on_call(write_skew.Rota([])) == (
        "nobody is on call")


def test_snapshot_of_a_rota_is_every_entry_with_its_version():
    rota = write_skew.Rota([_D2, _D1])
    rota.leave(_D2)
    rota.leave(_D2)
    assert write_skew.snapshot(rota) == ((_D1, 0, True), (_D2, 2, False))


def test_snapshot_makes_no_tool_call():
    # A rota that allows no call at all still gives its state. With a call
    # of read here, judging a run that used every call it was allowed
    # would be refused.
    rota = write_skew.Rota([_D1, _D2])
    assert write_skew.snapshot(_Limited(rota, 0)) == _ALL_ON_CALL


@pytest.mark.parametrize("doctors", [0, -1])
@pytest.mark.parametrize(
    "build", [write_skew.naive, write_skew.cas, write_skew.atomic])
def test_scenario_is_not_built_with_fewer_than_one_doctor(build, doctors):
    # With nobody on the rota nobody is on call before any call is made,
    # and the control would fail as the others do.
    with pytest.raises(ValueError, match=f"at least 1 doctor, not {doctors}$"):
        build(doctors=doctors)


@pytest.mark.parametrize(
    "build", [write_skew.naive, write_skew.cas, write_skew.atomic])
def test_one_doctor_alone_stays_on_call(build):
    # Each of the three makes one call when it is alone.
    scenario = _limited(build(doctors=1), 1)
    serial = oracle.serial_outcomes(scenario, write_skew.snapshot)
    (run,) = _runs(scenario)
    assert len(run.steps) == 1
    assert run.violation is None
    assert oracle.judge(serial, run) == oracle.Verdict(
        ((_D1, 0, True),), ((_D1,),))


def test_two_naive_doctors_have_6_schedules_and_4_leave_nobody_on_call():
    # A run starts with a read. Then either that doctor leaves before the
    # other reads, or both read before either leaves.
    # One leaves first: 2 choices of the doctor. The other then reads,
    # sees nobody else on call and stays. 2 schedules of 3 calls, the
    # serial orders.
    # Both read a rota with everybody on call: the reads in 2 orders and
    # the leaves in 2 orders. 2 * 2 = 4 schedules of 4 calls. Nobody is on
    # call from the second leave on, the last of the four calls.
    runs = _runs(write_skew.naive())
    broke = _broke(runs)
    assert collections.Counter(len(run.steps) for run in runs) == {3: 2, 4: 4}
    assert broke == [run for run in runs if len(run.steps) == 4]
    assert broke == [
        run for run in runs
        if [step.tool for step in run.steps[:2]] == ["read", "read"]]
    assert {run.violation for run in broke} == {
        scheduler.Violation("someone-on-call", "nobody is on call", 4)}


def test_two_reads_before_a_leave_take_both_doctors_off_call():
    run = explore.replay(write_skew.naive(), _BOTH_READ_FIRST)
    assert _calls(run) == [
        ("read", {}, _ALL_ON_CALL),
        ("read", {}, _ALL_ON_CALL),
        ("leave", {"person": _D1}, None),
        ("leave", {"person": _D2}, None),
    ]
    assert run.violation == scheduler.Violation(
        "someone-on-call", "nobody is on call", 4)
    # One after the other, the second doctor reads that the first left.
    serial = explore.replay(write_skew.naive(), (_D1, _D1, _D2))
    assert _calls(serial) == [
        ("read", {}, _ALL_ON_CALL),
        ("leave", {"person": _D1}, None),
        ("read", {}, ((_D1, 1, False), (_D2, 0, True))),
    ]
    assert serial.violation is None


def test_three_naive_doctors_have_90_schedules_and_72_leave_nobody_on_call():
    # A doctor's entry is written by that doctor alone, so a doctor is on
    # call when it reads, and it leaves unless both others have left by
    # then.
    # One doctor reads after both others left, and stays: 3 choices of
    # that doctor, and the others' four calls before its read, 4! / (2! 2!)
    # = 6 interleavings. Both of them leave, since the third is on call
    # throughout. 3 * 6 = 18 schedules of 5 calls, which pass.
    # Otherwise all three leave, in 6 calls. Of the 6! / (2! 2! 2!) = 90
    # interleavings of three doctors with two calls each, the 18 where one
    # doctor's two calls come after the four of the others are not runs:
    # that doctor does not leave. 90 - 18 = 72 schedules. Nobody is on
    # call from the third leave on, which is the last call: a run ends
    # with the last call of some doctor.
    # 18 + 72 = 90 schedules.
    runs = _runs(write_skew.naive(doctors=3))
    broke = _broke(runs)
    assert collections.Counter(len(run.steps) for run in runs) == {
        5: 18, 6: 72}
    assert broke == [run for run in runs if len(run.steps) == 6]
    assert {run.violation.step for run in broke} == {6}
    for run in runs:
        left = [step.agent for step in run.steps if step.tool == "leave"]
        assert len(left) == len(run.steps) - 3
        # In a run that passes the doctor that stays reads last.
        assert set(left) | {run.schedule[-1]} == {_D1, _D2, _D3}


@pytest.mark.parametrize("doctors", [2, 3])
def test_naive_doctors_fail_the_oracle_exactly_where_nobody_is_on_call(
        doctors):
    # One after another, every doctor but the last sees somebody on call
    # and leaves, and the last sees nobody else and stays. So a serial order
    # ends with its last doctor on call and the others off call, each
    # written once: n outcomes, each that of the (n - 1)! orders with that
    # doctor last. A run that leaves nobody on call ends as none of them.
    scenario = write_skew.naive(doctors=doctors)
    names = tuple(scenario.agents)
    serial = oracle.serial_outcomes(scenario, write_skew.snapshot)
    assert serial.orders == {
        _one_stays(names, last): _orders_ending_with(names, last)
        for last in names}
    runs = _runs(scenario)
    broke = _broke(runs)
    assert _failed(scenario, runs) == broke
    assert {oracle.judge(serial, run).outcome for run in broke} == {
        tuple((name, 1, False) for name in names)}


@pytest.mark.parametrize("doctors", [2, 3])
def test_compare_and_set_on_the_own_entry_fails_as_the_naive_doctors_do(
        doctors):
    # A doctor's entry is written by that doctor alone, so it still has
    # the version the doctor read, and no compare-and-set is refused. The
    # doctors check a rule over all the entries, and each writes one of
    # its own. So every run is a run of the naive doctors with leave_if in
    # the place of leave: the same schedules and the same violations.
    scenario = _fix(doctors)
    runs = _runs(scenario)
    naive = _runs(write_skew.naive(doctors=doctors))
    assert (len(runs), len(_broke(runs))) == {2: (6, 4), 3: (90, 72)}[doctors]
    assert [run.schedule for run in runs] == [run.schedule for run in naive]
    assert [run.violation for run in runs] == [run.violation for run in naive]
    writes = [
        step for run in runs for step in run.steps if step.tool != "read"]
    assert {step.tool for step in writes} == {"leave_if"}
    assert all(
        step.args == {"person": step.agent, "version": 0}
        and step.result is True for step in writes)
    assert _failed(scenario, runs) == _broke(runs)


@pytest.mark.parametrize("doctors", [2, 3])
def test_control_lets_every_doctor_but_the_last_leave(doctors):
    # A doctor makes one call, so the schedules are the n! orders of the
    # doctors, and each is a serial order. The serial-equivalence oracle
    # passes any such run, and the invariant holds while nobody leaves at
    # all. So what the control does is asked here: every call but the last
    # takes its doctor off call, and the last is declined.
    scenario = _control(doctors)
    names = tuple(scenario.agents)
    serial = oracle.serial_outcomes(scenario, write_skew.snapshot)
    runs = _runs(scenario)
    assert sorted(run.schedule for run in runs) == sorted(
        itertools.permutations(names))
    for run in runs:
        assert [(step.tool, step.args) for step in run.steps] == [
            ("leave_if_covered", {"person": name}) for name in run.schedule]
        results = [step.result for step in run.steps]
        assert all(result is True for result in results[:-1])
        assert results[-1] is False
        assert run.violation is None
        last = run.schedule[-1]
        assert oracle.judge(serial, run) == oracle.Verdict(
            _one_stays(names, last), _orders_ending_with(names, last))


@pytest.mark.parametrize("doctors", [2, 3])
def test_guards_allow_the_calls_of_the_longest_runs_and_no_more(doctors):
    assert max(len(run.steps) for run in _runs(_fix(doctors))) == 2 * doctors
    assert {len(run.steps) for run in _runs(_control(doctors))} == {doctors}
    # With one call fewer the search is refused, and does not hang.
    for short in (
            _limited(write_skew.cas(doctors=doctors), 2 * doctors - 1),
            _limited(write_skew.atomic(doctors=doctors), doctors - 1)):
        with pytest.raises(_StepLimitError, match="no call is left"):
            _runs(short)
    # The worlds of the two helpers refuse the call after the last of a
    # longest run.
    for guarded, calls in (
            (_fix(doctors), 2 * doctors), (_control(doctors), doctors)):
        world = guarded.make_world()
        for _ in range(calls):
            world.read()
        with pytest.raises(_StepLimitError, match="no call is left"):
            world.read()


@pytest.mark.parametrize("doctors", [2, 3, 5])
def test_control_passes_both_oracles_under_random_schedules(doctors):
    scenario = _control(doctors)
    serial = oracle.serial_outcomes(scenario, write_skew.snapshot)
    for seed in range(200):
        run = scheduler.run(scenario, scheduler.RandomPolicy(seed))
        assert run.violation is None
        assert not oracle.judge(serial, run).failed


def test_control_passes_the_search_that_finds_nobody_on_call():
    for scenario in (write_skew.naive(), _fix()):
        failure = explore.explore(scenario, range(20))
        assert failure is not None
        assert failure.result.violation.invariant == "someone-on-call"
    assert explore.explore(_control(), range(20)) is None


def test_failure_shrinks_to_one_doctor_inside_the_other():
    # Two doctors that both act make at least 1 switch, and a run with 1
    # is serial: the second doctor reads that the first left, and stays.
    # A run with 2 is a, then b, then a. For both to leave both must read
    # before either leaves, so a's first block is its read alone, b's
    # block is both its calls, and a's last block is its leave. That is
    # one schedule for each choice of a.
    # The search's failure is doctor-1 doctor-2 doctor-1 doctor-2, with 3
    # switches.
    scenario = write_skew.naive()
    failure = explore.explore(scenario, range(100))
    assert failure is not None
    assert failure.result.schedule == _BOTH_READ_FIRST
    assert shrink.size(failure.result) == (3, 2)
    shrunk = shrink.shrink(scenario, failure.result).result
    assert shrunk.schedule == _SHRUNK
    assert shrink.size(shrunk) == (2, 2)
    assert shrunk.violation == scheduler.Violation(
        "someone-on-call", "nobody is on call", 4)


@pytest.mark.parametrize(
    "build", [write_skew.naive, _fix], ids=["naive", "cas"])
def test_every_failure_of_two_doctors_shrinks_to_2_context_switches(build):
    # A failing run is both reads and then both leaves. With the reads a
    # then b, the leaves a then b give a b a b, 3 switches, and b then a
    # give a b b a, 2 switches. With 2 choices of a that is 2 runs of
    # each.
    scenario = build()
    broke = _broke(_runs(scenario))
    assert collections.Counter(map(shrink.size, broke)) == {
        (2, 2): 2, (3, 2): 2}
    assert {
        run.schedule for run in broke if shrink.size(run) == (2, 2)
    } == _SMALLEST
    for run in broke:
        shrunk = shrink.shrink(scenario, run).result
        assert shrunk.schedule in _SMALLEST
        assert shrunk in broke


@pytest.mark.parametrize(
    "build", [write_skew.naive, _fix], ids=["naive", "cas"])
def test_every_failure_of_three_doctors_shrinks_to_3_context_switches(build):
    # Nobody is on call only once all three have left, so no failure has
    # fewer than 3 doctors, and three doctors make at least 2 switches. A
    # run with 2 is serial, and its last doctor stays.
    # A run with 3 has one doctor, a, in two blocks, its read and its
    # leave, and the others, b and c, in one block each. a b c a fails: b
    # and c each read with a on call. b a c a fails: a reads with c on
    # call, and c with a. In a b a c the last doctor reads that nobody
    # else is on call and stays, and a's blocks can stand nowhere else. 3
    # choices of a, 2 orders of b and c, 2 places for a's read: 12 runs.
    scenario = build(doctors=3)
    broke = _broke(_runs(scenario))
    sizes = collections.Counter(map(shrink.size, broke))
    assert min(sizes) == (3, 3)
    assert sizes[3, 3] == 12
    for run in broke:
        shrunk = shrink.shrink(scenario, run).result
        assert shrink.size(shrunk) == (3, 3)
        assert shrunk.violation == scheduler.Violation(
            "someone-on-call", "nobody is on call", 6)
        assert shrunk in broke


@pytest.mark.parametrize(
    "schedule,lines", [(_SHRUNK, _SHRUNK_LINES), (_SERIAL, _SERIAL_LINES)],
    ids=["shrunk", "serial"])
def test_run_is_the_run_of_the_file_written_by_hand(schedule, lines):
    # The file holds the name of each tool and of each argument, what each
    # call returned, and the invariant's name and message: a change to any
    # of them makes a file saved before it replay to another trace. A file
    # also tells False from 0 and 1 from 1.0, which two runs that are
    # compared do not.
    scenario = write_skew.naive()
    run = explore.replay(scenario, schedule)
    text = "".join(f"{line}\n" for line in lines)
    assert tracefile.dumps(scenario, run) == text
    assert tracefile.replay(scenario, tracefile.loads(text)) == run
    # A file saved under version 1 holds the same lines, tuples and
    # booleans among them, and still replays.
    old = text.replace('"version": 2', '"version": 1')
    assert old != text
    assert tracefile.replay(scenario, tracefile.loads(old)) == run


def test_calls_of_the_fix_and_of_the_control_go_through_a_trace_file():
    fix = _fix()
    failed = explore.replay(fix, _SHRUNK)
    assert _calls(failed) == [
        ("read", {}, _ALL_ON_CALL),
        ("read", {}, _ALL_ON_CALL),
        ("leave_if", {"person": _D1, "version": 0}, True),
        ("leave_if", {"person": _D2, "version": 0}, True),
    ]
    saved = tracefile.loads(tracefile.dumps(fix, failed))
    assert saved.scenario == "write-skew-cas"
    assert tracefile.replay(fix, saved) == failed
    control = _control()
    passed = explore.replay(control, (_D2, _D1))
    assert _calls(passed) == [
        ("leave_if_covered", {"person": _D2}, True),
        ("leave_if_covered", {"person": _D1}, False),
    ]
    saved = tracefile.loads(tracefile.dumps(control, passed))
    assert saved.scenario == "write-skew-atomic"
    assert tracefile.replay(control, saved) == passed


@pytest.mark.parametrize(
    "build", [write_skew.naive, write_skew.cas], ids=["naive", "cas"])
def test_doctor_off_call_from_the_start_is_not_needed_for_a_failure(build):
    # doctor-3 reads that it is off call and stops: one call. It covers
    # for nobody, so doctor-1 and doctor-2 are the two doctors of the
    # scenario of two: 4 interleavings of 4 calls that fail and 2 of 3
    # calls that pass. The read of doctor-3 goes into any of the 5 places
    # among 4 calls and of the 4 among 3: 4 * 5 = 20 schedules that fail
    # and 2 * 4 = 8 that pass.
    # Nobody is on call from the second leave on, the last call of the two
    # doctors. In 4 of the 20 the read of doctor-3 comes after it, and the
    # violation is seen at step 4. In the other 16 it is seen at step 5.
    # An invariant asked only at the end would see all 20 at step 5.
    # The shrinker puts the read of doctor-3 off to the end, after the
    # violation, and ends with 2 switches and 2 doctors.
    scenario = _limited(_with_doctor_3_off_call(build(doctors=3)), 5)
    runs = _runs(scenario)
    broke = _broke(runs)
    assert (len(runs), len(broke)) == (28, 20)
    assert collections.Counter(run.violation.step for run in broke) == {
        4: 4, 5: 16}
    assert _failed(scenario, runs) == broke
    for run in broke:
        shrunk = shrink.shrink(scenario, run).result
        assert shrink.size(shrunk) == (2, 2)
        assert shrunk.schedule[:4] in _SMALLEST
        assert shrunk.schedule[4:] == (_D3,)
        assert shrunk.violation.step == 4
        assert shrunk in broke


def test_doctor_refused_while_on_call_asks_again_at_the_version_it_reads():
    # No tool of the rota writes an entry and leaves its doctor on call,
    # so with the rota's own tools a doctor on call is at version 0, and a
    # doctor that is refused is off call. Here a clerk renews the entry of
    # doctor-2, once before the doctor reads and twice between its read
    # and its write. The doctor asks at the version it read, which is 1.
    # It is refused, reads again and leaves at the version it reads then,
    # which is 3 and not the next one.
    scenario = _limited(
        dataclasses.replace(
            _on(write_skew.cas(), lambda: _Renewable([_D1, _D2])),
            agents={_D2: write_skew.cas_doctor, "clerk": _clerk}),
        7)
    run = explore.replay(
        scenario, ("clerk", _D2, "clerk", "clerk", _D2, _D2, _D2))
    renewed = ("renew", {"person": _D2}, None)
    assert _calls(run) == [
        renewed,
        ("read", {}, ((_D1, 0, True), (_D2, 1, True))),
        renewed,
        renewed,
        ("leave_if", {"person": _D2, "version": 1}, False),
        ("read", {}, ((_D1, 0, True), (_D2, 3, True))),
        ("leave_if", {"person": _D2, "version": 3}, True),
    ]


def test_doctor_that_was_sent_home_is_refused_and_does_not_leave_again():
    # A compare-and-set is refused when somebody else wrote the doctor's
    # entry since the doctor read it. Here a registrar sends doctor-1
    # home, and doctor-2 is on the rota and does nothing. 3 schedules, by
    # where the registrar's call falls among those of doctor-1.
    # Before its read: the doctor reads that it is off call and stops.
    # After its write: the doctor left, and is sent home as well.
    # Between the two: the write is refused. The doctor reads again, finds
    # itself off call and stops, with its entry written once.
    scenario = _limited(
        dataclasses.replace(
            write_skew.cas(), name="write-skew-registrar",
            agents={_D1: write_skew.cas_doctor, "registrar": _registrar}),
        4)
    runs = _runs(scenario)
    assert sorted(run.schedule for run in runs) == [
        (_D1, _D1, "registrar"),
        (_D1, "registrar", _D1, _D1),
        ("registrar", _D1),
    ]
    assert _broke(runs) == []
    sent_home = ((_D1, 1, False), (_D2, 0, True))
    assert _calls(explore.replay(scenario, ("registrar", _D1))) == [
        ("leave", {"person": _D1}, None),
        ("read", {}, sent_home),
    ]
    between = explore.replay(scenario, (_D1, "registrar", _D1, _D1))
    assert _calls(between) == [
        ("read", {}, _ALL_ON_CALL),
        ("leave", {"person": _D1}, None),
        ("leave_if", {"person": _D1, "version": 0}, False),
        ("read", {}, sent_home),
    ]


def test_doctor_whose_write_raised_reads_again_and_leaves():
    # The scheduler gives the doctor the error as a value, and that value
    # is true. A doctor that read the result by truth would take the error
    # for a write that went through, and stay on call.
    scenario = _limited(
        _on(write_skew.cas(), lambda: _RaisesOnce([_D1, _D2])), 5)
    run = explore.replay(scenario, (_D1,) * 4 + (_D2,))
    assert [(step.tool, step.result) for step in run.steps] == [
        ("read", _ALL_ON_CALL),
        ("leave_if", scheduler.Raised("RuntimeError", "try again")),
        ("read", _ALL_ON_CALL),
        ("leave_if", True),
        ("read", ((_D1, 1, False), (_D2, 0, True))),
    ]
    # It goes by what it reads after the error, not by what it read
    # before: here doctor-2 leaves in between, and doctor-1 stays.
    other = explore.replay(scenario, (_D1, _D1, _D2, _D2, _D1))
    assert _calls(other)[2:] == [
        ("read", {}, _ALL_ON_CALL),
        ("leave_if", {"person": _D2, "version": 0}, True),
        ("read", {}, ((_D1, 0, True), (_D2, 1, False))),
    ]
    assert other.violation is None


def test_doctor_whose_write_raises_every_time_never_ends():
    # What taking only True for a write costs: the doctor tries again
    # after an error as after a refusal, and does not look at what the
    # error says. Cut at the 20 calls the test allows.
    scenario = _limited(
        _on(write_skew.cas(), lambda: _AlwaysRaises([_D1, _D2])), 20)
    with pytest.raises(_StepLimitError, match="no call is left"):
        explore.replay(scenario, (_D1,) * 21)


@pytest.mark.parametrize("doctors", [2, 3])
def test_rota_that_lets_nobody_leave_passes_both_oracles(doctors):
    # Why the test of the control counts the doctors that leave: with a
    # tool that declines everybody the invariant holds, and every schedule
    # is still a serial order.
    control = write_skew.atomic(doctors=doctors)
    scenario = _limited(
        _on(control, lambda: _LetsNobodyLeave(control.agents)), doctors)
    runs = _runs(scenario)
    assert len(runs) == len(list(itertools.permutations(control.agents)))
    assert all(
        step.result is False for run in runs for step in run.steps)
    assert _broke(runs) == []
    assert _failed(scenario, runs) == []


def test_doctor_that_leaves_when_declined_fails_only_the_invariant():
    # One doctor leaves, the other is declined and leaves all the same: 2
    # schedules of 3 calls, and nobody is on call after the third. Each is
    # a serial order, so the serial orders end as the runs end, and the
    # oracle sees nothing. A failure that the serial orders share takes an
    # invariant.
    scenario = _staffed_by(_defiant_doctor)
    runs = _runs(scenario)
    assert sorted(run.schedule for run in runs) == [
        (_D1, _D2, _D2), (_D2, _D1, _D1)]
    assert {run.violation for run in runs} == {
        scheduler.Violation("someone-on-call", "nobody is on call", 3)}
    assert _failed(scenario, runs) == []


def test_doctor_that_asks_again_until_it_may_leave_never_ends():
    # The last doctor on call is declined whenever it asks: the others
    # have left and do not come back. One after the other the second
    # doctor is that one, so not even a serial order ends: each is cut at
    # the 20 calls the test allows.
    scenario = _staffed_by(_waiting_doctor)
    with pytest.raises(_StepLimitError, match="'leave_if_covered'"):
        explore.replay(scenario, (_D1,) + (_D2,) * 20)
    with pytest.raises(_StepLimitError):
        oracle.serial_outcomes(scenario, write_skew.snapshot)
    with pytest.raises(_StepLimitError):
        explore.explore(scenario, range(20))
