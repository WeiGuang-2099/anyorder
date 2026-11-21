"""Checks the lost update scenario and its control under both oracles.

The document's tools come first, then the naive editors with their counts
worked out by hand, then the control. After them come shrinking, trace
files and three wrong controls of the tests' own, which show what each
oracle sees of an editor that handles a refused write badly. Documents
whose compare-and-set raises come last.
"""

import collections
import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any

import pytest

from anyorder import explore
from anyorder import oracle
from anyorder import scheduler
from anyorder import shrink
from anyorder import tracefile
from anyorder.scenarios import lost_update

_E1, _E2 = "editor-1", "editor-2"
_BOTH_READ_FIRST = (_E1, _E2, _E1, _E2)
# One editor reads, the other runs to its end, the first writes. The file
# of the second of these is written out by hand below: line 1 opens it,
# lines 2 to 5 are the four calls, and line 6 holds the violation, seen at
# the fourth call, where editor-2 stores what it read with its own line.
_SMALLEST = {(_E1, _E2, _E2, _E1), (_E2, _E1, _E1, _E2)}
_SHRUNK = (_E2, _E1, _E1, _E2)
_SHRUNK_LINES = (
    '{"kind": "anyorder-trace", "version": 2, '
    '"scenario": "lost-update-naive"}',
    '{"kind": "step", "agent": "editor-2", "tool": "read", "args": {}, '
    '"result": {"$tuple": [0, {"$tuple": []}]}}',
    '{"kind": "step", "agent": "editor-1", "tool": "read", "args": {}, '
    '"result": {"$tuple": [0, {"$tuple": []}]}}',
    '{"kind": "step", "agent": "editor-1", "tool": "write", '
    '"args": {"lines": {"$tuple": ["editor-1"]}}, "result": null}',
    '{"kind": "step", "agent": "editor-2", "tool": "write", '
    '"args": {"lines": {"$tuple": ["editor-2"]}}, "result": null}',
    '{"kind": "end", "violation": {"invariant": "no-line-lost", '
    '"message": "the document held editor-1 and now holds only editor-2", '
    '"step": 4}}',
)


class _StepLimitError(scheduler.SchedulerError):
    """A run made more calls than its test allows."""


class _Limited:
    """A world that refuses the call after the last one it allows.

    An editor that handles a refused write wrongly, or whose tool raises
    every time, may never end. The scheduler passes its own errors on and
    gives them to no agent, so such a run fails its test where it would
    hang it.
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


def _control(editors: int = 2) -> scheduler.Scenario:
    """The control, allowed the most calls its editors can need.

    A write is refused only when another editor's write went through since
    the read, and the refused editor then reads after that write. Each
    editor's write goes through once, so the k-th of them refuses at most
    the n - k editors still at work: n (n - 1) / 2 refusals at most. With
    a read before each of the n writes that go through and before each
    refused one, a run makes at most 2 n + n (n - 1) = n (n + 1) calls.
    """
    return _limited(
        lost_update.cas(editors=editors), editors * (editors + 1))


async def _meek_editor(ctx: scheduler.AgentContext) -> None:
    """Gives up when its write is refused."""
    version, lines = await ctx.call("read")
    await ctx.call("write_if", version=version, lines=(*lines, ctx.name))


async def _hasty_editor(ctx: scheduler.AgentContext) -> None:
    """Tries the next version when it is refused, with the lines it read."""
    version, lines = await ctx.call("read")
    while not await ctx.call(
            "write_if", version=version, lines=(*lines, ctx.name)):
        version += 1


async def _stubborn_editor(ctx: scheduler.AgentContext) -> None:
    """Makes the same call again when it is refused."""
    version, lines = await ctx.call("read")
    while not await ctx.call(
            "write_if", version=version, lines=(*lines, ctx.name)):
        pass


class _RaisesOnce(lost_update.Document):
    """A document whose first compare-and-set raises and stores nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.raised = False

    def write_if(self, version: int, lines: Sequence[str]) -> bool:
        if not self.raised:
            self.raised = True
            raise RuntimeError("try again")
        return super().write_if(version, lines)


class _AlwaysRaises(lost_update.Document):
    """A document whose compare-and-set raises every time."""

    def write_if(self, version: int, lines: Sequence[str]) -> bool:
        raise RuntimeError("try again")


def _edited_by(editor: scheduler.Agent) -> scheduler.Scenario:
    """The control of two editors with another editor in place of its own."""
    return _limited(
        dataclasses.replace(
            lost_update.cas(), name="lost-update-wrong",
            agents={_E1: editor, _E2: editor}),
        20)


def _runs(
    scenario: scheduler.Scenario, limit: int = 1000
) -> list[scheduler.RunResult]:
    """Runs a whole exhaustive search, up to a limit."""
    search = explore.exhaust(scenario)
    runs = list(itertools.islice(search, limit))
    assert next(search, None) is None, f"more than {limit} runs"
    return runs


def _lost(runs: list[scheduler.RunResult]) -> list[scheduler.RunResult]:
    """Keeps the runs that broke the invariant."""
    return [run for run in runs if run.violation is not None]


def _failed(
    scenario: scheduler.Scenario, runs: list[scheduler.RunResult]
) -> list[scheduler.RunResult]:
    """Keeps the runs that end as no serial order ends."""
    serial = oracle.serial_outcomes(scenario, lost_update.snapshot)
    return [run for run in runs if oracle.judge(serial, run).failed]


def _refusals(run: scheduler.RunResult) -> collections.Counter[str]:
    """Counts the refused writes of each editor."""
    return collections.Counter(
        step.agent for step in run.steps
        if step.tool == "write_if" and step.result is False)


def test_read_gives_the_version_and_the_lines_of_that_moment():
    document = lost_update.Document()
    before = document.read()
    document.write(("a", "b"))
    assert before == (0, ())
    assert document.read() == (1, ("a", "b"))


def test_document_keeps_lines_of_its_own_and_not_the_list_it_is_given():
    document = lost_update.Document()
    given = ["a"]
    document.write(given)
    given.append("b")
    assert document.read() == (1, ("a",))


def test_write_if_stores_only_at_the_version_it_is_given():
    # Two lines at version 1, so that a count of the lines cannot stand
    # for the version.
    document = lost_update.Document()
    assert document.write_if(0, ("a", "b")) is True
    assert document.write_if(0, ("c",)) is False
    # Nor at a version the document has not reached.
    assert document.write_if(2, ("c",)) is False
    assert document.read() == (1, ("a", "b"))
    assert document.write_if(1, ("a", "b", "c")) is True
    assert document.read() == (2, ("a", "b", "c"))


def test_version_goes_up_with_every_write_whatever_it_stores():
    # Also with a write that changes nothing: a version tells a document
    # that was written from one that was not, which its lines cannot.
    document = lost_update.Document()
    document.write(())
    assert document.read() == (1, ())
    document.write(("a",))
    document.write(("a",))
    assert document.read() == (3, ("a",))


def test_document_remembers_each_line_once_in_the_order_they_came():
    document = lost_update.Document()
    for lines in (("b", "a"), ("c",), ("a", "d", "d")):
        document.write(lines)
    assert document.held == ("b", "a", "c", "d")
    # The lines themselves are kept as they were given, twice if so.
    assert document.read() == (3, ("a", "d", "d"))


def test_invariant_names_the_lines_that_are_gone_and_those_that_are_left():
    document = lost_update.Document()
    assert lost_update.no_line_lost(document) is None
    document.write(("b", "a"))
    assert lost_update.no_line_lost(document) is None
    document.write(("c", "a"))
    assert lost_update.no_line_lost(document) == (
        "the document held b and now holds only c, a")
    document.write(())
    assert lost_update.no_line_lost(document) == (
        "the document held b, a, c and is now empty")
    # A line that comes back is not gone.
    document.write(("a", "c", "b"))
    assert lost_update.no_line_lost(document) is None


def test_snapshot_of_a_document_is_its_version_and_its_lines_in_order():
    # Three lines at version 2, and not in the order of the alphabet.
    document = lost_update.Document()
    document.write(("c",))
    document.write(("b", "a", "c"))
    assert lost_update.snapshot(document) == (2, ("b", "a", "c"))


def test_two_naive_editors_have_6_schedules_and_4_lose_a_line():
    # An editor makes two calls whatever it reads, so the schedules are
    # the 4! / (2! 2!) = 6 interleavings of two editors.
    # Both read before either writes: the reads in 2 orders and the writes
    # in 2 orders. 2 * 2 = 4 schedules. Both read an empty document, and
    # the second write stores one line where the first had stored another.
    # Otherwise one editor reads and writes before the other reads: the 2
    # serial orders, which end with both lines.
    # The line is lost at the second write, the last of the four calls.
    runs = _runs(lost_update.naive())
    lost = _lost(runs)
    assert (len(runs), len(lost)) == (6, 4)
    assert lost == [
        run for run in runs
        if [step.tool for step in run.steps[:2]] == ["read", "read"]]
    assert {run.violation.invariant for run in lost} == {"no-line-lost"}
    assert {run.violation.step for run in lost} == {4}


def test_two_reads_before_a_write_lose_the_line_of_the_first_writer():
    run = explore.replay(lost_update.naive(), _BOTH_READ_FIRST)
    assert [(step.tool, step.args, step.result) for step in run.steps] == [
        ("read", {}, (0, ())),
        ("read", {}, (0, ())),
        ("write", {"lines": (_E1,)}, None),
        ("write", {"lines": (_E2,)}, None),
    ]
    assert run.violation == scheduler.Violation(
        "no-line-lost",
        "the document held editor-1 and now holds only editor-2",
        4)


def test_three_naive_editors_have_90_schedules_and_84_lose_a_line():
    # 6! / (2! 2! 2!) = 90 interleavings of three editors.
    # A write keeps every line exactly when its editor read after the
    # write before it, so a run keeps every line exactly when it is one of
    # the 3! = 6 serial orders. The other 84 lose a line, at the first
    # write whose editor read before the write before it. Say the editors
    # write in the order p, q, r: 6 choices.
    # q read before p's write and r reads after q's write: p and q read in
    # 2 orders, p writes, q writes, r reads and writes. 6 * 2 = 12 runs,
    # which see the loss at step 4.
    # q read before p's write and r reads before q's write: p and q read
    # in 2 orders and p writes, and r's read goes into one of the 4 places
    # among those three calls. 6 * 2 * 4 = 48 runs, at step 5.
    # q read after p's write and r read before q's write: p reads and
    # writes, q reads and writes, and r's read goes into one of the 4
    # places before q's write. 6 * 4 = 24 runs, at r's write, step 6.
    # 12 + 48 + 24 = 84.
    runs = _runs(lost_update.naive(editors=3))
    lost = _lost(runs)
    assert (len(runs), len(lost)) == (90, 84)
    assert collections.Counter(run.violation.step for run in lost) == {
        4: 12, 5: 48, 6: 24}
    for run in lost:
        assert run.steps[run.violation.step - 1].tool == "write"
        # A run goes on to its end after the violation.
        assert len(run.steps) == 6


@pytest.mark.parametrize("editors", [2, 3])
def test_naive_editors_fail_the_oracle_exactly_where_a_line_is_lost(editors):
    # One after another, each editor adds its line below those before it,
    # so a serial order ends at version n with the lines in its own order.
    # A run that lost a line ends with fewer lines than n.
    scenario = lost_update.naive(editors=editors)
    serial = oracle.serial_outcomes(scenario, lost_update.snapshot)
    assert serial.orders == {
        (editors, order): (order,)
        for order in itertools.permutations(scenario.agents)}
    runs = _runs(scenario)
    lost = _lost(runs)
    assert _failed(scenario, runs) == lost
    assert all(
        len(oracle.judge(serial, run).outcome[1]) < editors for run in lost)


def test_two_cas_editors_have_6_schedules_and_4_hold_a_refused_write():
    # A run starts with a read. Then either that editor writes before the
    # other reads, or both read before either writes.
    # One writes first: 2 choices of the editor, and the other then reads
    # and writes. 2 schedules of 4 calls, the serial orders.
    # Both read version 0: the reads in 2 orders, and either editor writes
    # first and is done. The other is refused, reads again and writes.
    # 2 * 2 = 4 schedules of 6 calls.
    # A write is refused in the 4, which happens in no serial order. Each
    # still ends as a serial order ends: both lines, in the order the
    # writes went through.
    scenario = _control()
    serial = oracle.serial_outcomes(scenario, lost_update.snapshot)
    runs = _runs(scenario)
    assert collections.Counter(len(run.steps) for run in runs) == {4: 2, 6: 4}
    assert collections.Counter(
        sum(_refusals(run).values()) for run in runs) == {0: 2, 1: 4}
    assert _lost(runs) == []
    for run in runs:
        wrote = tuple(step.agent for step in run.steps if step.result is True)
        assert oracle.judge(serial, run) == oracle.Verdict((2, wrote), (wrote,))


def test_three_cas_editors_have_498_schedules_and_all_pass():
    # Write (r f s) for the number of ways to finish with r editors that
    # read next, f that hold a read of the version as it is, and s that
    # hold a read of an older one. A read makes its editor one of the f. A
    # write by one of the f goes through, ends that editor and makes the
    # other f stale. A write by one of the s is refused, and that editor
    # reads next.
    # One editor: (1 0 0) = 1 and (0 0 1) = 1.
    # Two: (0 2 0) = 2 * (0 0 1) = 2. (1 1 0) = (0 2 0) + (1 0 0) = 3, by
    # the read or the write. (2 0 0) = 2 * (1 1 0) = 6, as counted in the
    # test above. (0 1 1) = (0 0 1) + (1 1 0) = 4, by the write or the
    # refusal. (1 0 1) = (0 1 1) + (2 0 0) = 10, by the read or the
    # refusal. (0 0 2) = 2 * (1 0 1) = 20.
    # Three, by the reads before the first write:
    # 1 read: 3 choices of the editor, then (2 0 0). 3 * 6 = 18.
    # 2 reads: in 3 * 2 orders, either of the two writes, then (1 0 1).
    # 6 * 2 * 10 = 120.
    # 3 reads: in 3! orders, any of the three writes, then (0 0 2).
    # 6 * 3 * 20 = 360.
    # 18 + 120 + 360 = 498 schedules. No write is refused in the 3! = 6
    # serial orders, and at least one is in each of the other 492.
    scenario = _control(editors=3)
    serial = oracle.serial_outcomes(scenario, lost_update.snapshot)
    runs = _runs(scenario)
    assert len(runs) == 498
    assert collections.Counter(
        [step.tool for step in run.steps].index("write_if")
        for run in runs) == {1: 18, 2: 120, 3: 360}
    assert sum(not _refusals(run) for run in runs) == 6
    assert _lost(runs) == []
    for run in runs:
        wrote = tuple(step.agent for step in run.steps if step.result is True)
        assert oracle.judge(serial, run) == oracle.Verdict((3, wrote), (wrote,))


@pytest.mark.parametrize("editors", [2, 3])
def test_longest_run_of_the_control_makes_n_times_n_plus_1_calls(editors):
    # The docstring of _control gives the bound. It is reached: every
    # editor reads the empty document, and they then write in turn, each
    # write refusing all those still at work, who read again together.
    most = editors * (editors + 1)
    runs = _runs(_control(editors))
    assert max(len(run.steps) for run in runs) == most
    assert max(max(_refusals(run).values(), default=0) for run in runs) == (
        editors - 1)
    # With one call fewer the search is refused, and does not hang.
    short = _limited(lost_update.cas(editors=editors), most - 1)
    with pytest.raises(_StepLimitError, match="no call is left"):
        _runs(short)


@pytest.mark.parametrize("editors", [2, 3, 5])
def test_control_passes_both_oracles_under_random_schedules(editors):
    scenario = _control(editors)
    serial = oracle.serial_outcomes(scenario, lost_update.snapshot)
    for seed in range(200):
        run = scheduler.run(scenario, scheduler.RandomPolicy(seed))
        assert run.violation is None
        assert not oracle.judge(serial, run).failed


def test_control_passes_the_search_that_finds_the_lost_line():
    failure = explore.explore(lost_update.naive(), range(20))
    assert failure is not None
    assert failure.result.violation.invariant == "no-line-lost"
    assert explore.explore(_control(), range(20)) is None


def test_lost_line_shrinks_to_one_editor_inside_the_other():
    # Two editors that both act make at least 1 switch, and a run with 1
    # is serial and keeps both lines. A run with 2 is a, then b, then a.
    # To lose a line both must read the empty document, so a's first block
    # is its read alone, b's block is both its calls, and a's last block
    # is its write. That is one schedule for each choice of a.
    # The search's failure is editor-1 editor-2 editor-1 editor-2, with 3
    # switches.
    scenario = lost_update.naive()
    failure = explore.explore(scenario, range(100))
    assert failure is not None
    assert failure.result.schedule == _BOTH_READ_FIRST
    assert shrink.size(failure.result) == (3, 2)
    shrunk = shrink.shrink(scenario, failure.result).result
    assert shrunk.schedule == _SHRUNK
    assert shrink.size(shrunk) == (2, 2)
    assert shrunk.violation == scheduler.Violation(
        "no-line-lost",
        "the document held editor-1 and now holds only editor-2",
        4)


def test_every_lost_line_of_two_editors_shrinks_to_2_context_switches():
    # A failing run is both reads and then both writes. With the reads a
    # then b, the writes a then b give a b a b, 3 switches, and b then a
    # give a b b a, 2 switches. With 2 choices of a that is 2 runs of
    # each.
    scenario = lost_update.naive()
    lost = _lost(_runs(scenario))
    assert collections.Counter(map(shrink.size, lost)) == {
        (2, 2): 2, (3, 2): 2}
    assert {
        run.schedule for run in lost if shrink.size(run) == (2, 2)
    } == _SMALLEST
    for run in lost:
        shrunk = shrink.shrink(scenario, run).result
        assert shrunk.schedule in _SMALLEST
        assert shrunk in lost


def test_every_lost_line_of_three_editors_shrinks_to_two_editors():
    # A line is lost at a write, after the two reads and the first write
    # of two editors, so two editors make 4 calls, the last of which loses
    # it, and as above 2 switches are the least for them. The third editor
    # is not needed: put off to the end, it reads and writes after the
    # violation.
    scenario = lost_update.naive(editors=3)
    lost = _lost(_runs(scenario))
    assert min(map(shrink.size, lost)) == (2, 2)
    for run in lost:
        shrunk = shrink.shrink(scenario, run).result
        assert shrink.size(shrunk) == (2, 2)
        assert shrunk.violation.invariant == "no-line-lost"
        assert shrunk.violation.step == 4
        assert len({step.agent for step in shrunk.steps[4:]}) == 1
        assert shrunk in lost


def test_shrunk_failure_is_the_run_of_the_file_written_by_hand():
    # The file holds the name of each tool and of each argument, what each
    # call returned, and the invariant's name and message: a change to any
    # of them makes a file saved before it replay to another trace.
    scenario = lost_update.naive()
    run = explore.replay(scenario, _SHRUNK)
    text = "".join(f"{line}\n" for line in _SHRUNK_LINES)
    assert tracefile.dumps(scenario, run) == text
    assert tracefile.replay(scenario, tracefile.loads(text)) == run
    # A file saved under version 1 holds the same lines, a tuple among
    # them, and still replays.
    old = text.replace('"version": 2', '"version": 1')
    assert old != text
    assert tracefile.replay(scenario, tracefile.loads(old)) == run


def test_refused_write_goes_through_a_trace_file():
    scenario = _control()
    run = explore.replay(scenario, (_E1, _E2, _E2, _E1, _E1, _E1))
    assert [step.result for step in run.steps] == [
        (0, ()), (0, ()), True, False, (1, (_E2,)), True]
    saved = tracefile.loads(tracefile.dumps(scenario, run))
    assert saved.scenario == "lost-update-cas"
    assert tracefile.replay(scenario, saved) == run


def test_editor_that_gives_up_loses_no_line_and_fails_the_oracle():
    # The 2 serial orders pass. In the other 4 schedules both editors read
    # version 0, one write goes through and the other is refused, and the
    # refused editor stops. No line that was held is gone, so the
    # invariant holds. The document ends at version 1 with one line, as no
    # serial order ends: only the oracle sees it.
    scenario = _edited_by(_meek_editor)
    runs = _runs(scenario)
    assert collections.Counter(len(run.steps) for run in runs) == {4: 6}
    assert _lost(runs) == []
    failed = _failed(scenario, runs)
    assert failed == [run for run in runs if _refusals(run)]
    assert len(failed) == 4
    # A seeded search looks for a broken invariant, and finds none.
    assert explore.explore(scenario, range(100)) is None


def test_editor_that_tries_the_next_version_without_reading_loses_a_line():
    # The 2 serial orders pass. In the other 4 schedules both editors read
    # version 0, one write goes through and the other is refused. The
    # refused editor then asks for version 1 with the lines it read at
    # version 0, and that write goes through, 5 calls in all, and drops
    # the other editor's line. Both oracles see it.
    scenario = _edited_by(_hasty_editor)
    runs = _runs(scenario)
    assert collections.Counter(len(run.steps) for run in runs) == {4: 2, 5: 4}
    lost = _lost(runs)
    assert lost == [run for run in runs if _refusals(run)]
    assert {run.violation.step for run in lost} == {5}
    assert _failed(scenario, runs) == lost


def test_editor_that_tries_again_without_reading_never_ends():
    # Once the other editor's write went through, the same call is refused
    # every time. One after the other no write is refused, so the serial
    # orders end, and a search is the first to meet such a run: here it is
    # cut at the 20 calls the test allows.
    scenario = _edited_by(_stubborn_editor)
    serial = oracle.serial_outcomes(scenario, lost_update.snapshot)
    assert set(serial.orders) == {(2, (_E1, _E2)), (2, (_E2, _E1))}
    with pytest.raises(_StepLimitError, match="'write_if'"):
        explore.replay(scenario, (_E1, _E2, _E1) + (_E2,) * 18)
    with pytest.raises(_StepLimitError):
        explore.explore(scenario, range(20))


def test_editor_whose_write_raised_reads_again_and_writes():
    # The scheduler gives the editor the error as a value, and that value
    # is true. An editor that read the result by truth would take the
    # error for a write that went through and stop. No line is lost and
    # the one serial order ends the same way, so neither oracle would see
    # that its line was never added.
    scenario = _limited(
        dataclasses.replace(
            lost_update.cas(editors=1), make_world=_RaisesOnce),
        4)
    run = explore.replay(scenario, (_E1,) * 4)
    asked = {"version": 0, "lines": (_E1,)}
    assert [(step.tool, step.args, step.result) for step in run.steps] == [
        ("read", {}, (0, ())),
        ("write_if", asked, scheduler.Raised("RuntimeError", "try again")),
        ("read", {}, (0, ())),
        ("write_if", asked, True),
    ]
    assert run.violation is None
    # It writes what it reads after the error, not what it read before:
    # here editor-2 reads and writes in between.
    both = _limited(
        dataclasses.replace(lost_update.cas(), make_world=_RaisesOnce), 6)
    last = explore.replay(both, (_E1, _E1, _E2, _E2, _E1, _E1)).steps[-1]
    assert (last.agent, last.tool, last.args, last.result) == (
        _E1, "write_if", {"version": 1, "lines": (_E2, _E1)}, True)


def test_editor_whose_write_raises_every_time_never_ends():
    # What taking only True for a write costs: the editor tries again
    # after an error as after a refusal, and does not look at what the
    # error says. Cut at the 20 calls the test allows.
    scenario = _limited(
        dataclasses.replace(
            lost_update.cas(editors=1), make_world=_AlwaysRaises),
        20)
    with pytest.raises(_StepLimitError, match="no call is left"):
        explore.replay(scenario, (_E1,) * 21)
