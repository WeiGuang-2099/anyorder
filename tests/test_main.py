"""Checks that the demo prints what the README says it prints."""

import pathlib
import subprocess
import sys

import pytest

from anyorder import __main__ as demo
from anyorder import explore
from anyorder import scheduler
from anyorder import tracefile
from anyorder.scenarios import lost_update
from anyorder.scenarios import oversell

_README = pathlib.Path(__file__).parent.parent / "README.md"
# Each scenario the demo knows by name, as the README says it is built,
# with a schedule of it. The numbers are written out: a name must not
# come to stand for another build when a default changes.
_KNOWN = [
    (oversell.naive(stock=1, buyers=2),
     ("buyer-1", "buyer-2", "buyer-1", "buyer-1", "buyer-2", "buyer-2")),
    (oversell.atomic(stock=1, buyers=2), ("buyer-2", "buyer-1")),
    (lost_update.naive(editors=2),
     ("editor-1", "editor-2", "editor-1", "editor-2")),
    (lost_update.cas(editors=2),
     ("editor-1", "editor-2", "editor-2", "editor-1", "editor-1",
      "editor-1")),
]
_NAMES = [scenario.name for scenario, _ in _KNOWN]


def _assert_readme_shows(output: str) -> None:
    # The README shows the output as a block indented by four spaces.
    printed = [f"    {line}" for line in output.splitlines()]
    readme = _README.read_text(encoding="utf-8").splitlines()
    start = readme.index(printed[0])
    end = start + len(printed)
    assert readme[start:end] == printed
    # The block starts where the output starts and ends where it ends.
    assert not readme[start - 1].startswith("    ")
    assert not readme[end].startswith("    ")


def _saved_failure(tmp_path, capsys):
    """Runs the demo with --save and returns the file and what it printed."""
    path = tmp_path / "failure.jsonl"
    assert demo.main(["--save", str(path)]) == 1
    return path, capsys.readouterr().out


def _exit_status(arguments, capsys):
    """Runs the demo, which must end itself, and returns how and its text."""
    with pytest.raises(SystemExit) as ended:
        demo.main(arguments)
    return ended.value.code, capsys.readouterr()


def _names_in_the_refusal(tmp_path, capsys):
    """Returns the scenarios the demo names when a file has none of them."""
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        '{"kind": "anyorder-trace", "version": 1, "scenario": "unknown"}\n'
        '{"kind": "end", "violation": null}\n', encoding="utf-8")
    status, printed = _exit_status(["--replay", str(path)], capsys)
    assert status == 2
    _, _, names = printed.err.strip().rpartition("It knows ")
    return names.split(", ")


def test_demo_prints_the_trace_in_the_readme(capsys):
    assert demo.main() == 1
    _assert_readme_shows(capsys.readouterr().out)


def test_demo_with_save_prints_the_same_and_writes_the_shrunk_failure(
    tmp_path, capsys
):
    assert demo.main() == 1
    plain = capsys.readouterr().out
    path, printed = _saved_failure(tmp_path, capsys)
    assert printed == plain
    data = path.read_bytes()
    # The same bytes on every system: no carriage return, nothing to decode.
    assert data.isascii() and b"\r" not in data
    saved = tracefile.loads(data.decode("ascii"))
    assert saved.scenario == "oversell-naive"
    assert saved.violation is not None
    # The last schedule printed is the shrunk one.
    schedules = [
        line for line in plain.splitlines() if line.startswith("schedule: ")]
    assert schedules[-1] == f"schedule: {' '.join(saved.schedule)}"


def test_demo_with_save_writes_in_place_of_a_file_of_that_name(
    tmp_path, capsys
):
    path, _ = _saved_failure(tmp_path, capsys)
    data = path.read_bytes()
    path.write_bytes(b"an older file\n" + data)
    assert demo.main(["--save", str(path)]) == 1
    assert path.read_bytes() == data


def test_demo_replays_a_saved_failure_and_prints_what_the_readme_says(
    tmp_path, capsys
):
    path, searched = _saved_failure(tmp_path, capsys)
    # 0 and not 1: the file replayed. The violation is what it holds.
    assert demo.main(["--replay", str(path)]) == 0
    replayed = capsys.readouterr().out
    # After its first line a replay prints the shrunk failure as the search
    # did: the message, the schedule and the six steps.
    assert replayed.splitlines()[1:] == searched.splitlines()[-8:]
    _assert_readme_shows(replayed)


def test_demo_replays_a_file_with_other_line_ends(tmp_path, capsys):
    path, _ = _saved_failure(tmp_path, capsys)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert demo.main(["--replay", str(path)]) == 0


def test_demo_replay_of_a_run_that_passed_returns_0(tmp_path, capsys):
    scenario = oversell.naive(stock=1, buyers=2)
    serial = explore.replay(
        scenario, ("buyer-1", "buyer-1", "buyer-1", "buyer-2"))
    path = tmp_path / "serial.jsonl"
    path.write_text(tracefile.dumps(scenario, serial), encoding="utf-8")
    assert demo.main(["--replay", str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "oversell-naive: the file replays to the same trace",
        "  no violation",
        "schedule: buyer-1 buyer-1 buyer-1 buyer-2",
        "  1. buyer-1: read_stock() -> 1",
        "  2. buyer-1: write_stock(value=0) -> None",
        "  3. buyer-1: place_order(buyer='buyer-1') -> None",
        "  4. buyer-2: read_stock() -> 0",
    ]


def test_demo_does_not_catch_a_file_that_replays_differently(
    tmp_path, capsys
):
    path, _ = _saved_failure(tmp_path, capsys)
    text = path.read_text(encoding="utf-8")
    # Both buyers read 1. The file now says the first read gave 2.
    path.write_text(
        text.replace('"result": 1', '"result": 2', 1), encoding="utf-8")
    with pytest.raises(scheduler.ReplayDivergenceError, match="step 1 is "):
        demo.main(["--replay", str(path)])
    # A failure that did not replay is not reported.
    assert capsys.readouterr().out == ""


def test_demo_exits_with_2_on_a_file_that_is_not_a_trace_file(
    tmp_path, capsys
):
    path, _ = _saved_failure(tmp_path, capsys)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]), encoding="utf-8")
    status, printed = _exit_status(["--replay", str(path)], capsys)
    assert status == 2
    assert printed.out == ""
    assert 'line 7 is of kind "step", not "end"' in printed.err


def test_demo_exits_with_2_on_a_file_that_is_not_utf_8(tmp_path, capsys):
    path, _ = _saved_failure(tmp_path, capsys)
    # One byte that no UTF-8 text holds, inside the violation's message.
    data = path.read_bytes()
    assert data.count(b"orders placed") == 1
    path.write_bytes(data.replace(b"orders placed", b"orders plac\xffd"))
    status, printed = _exit_status(["--replay", str(path)], capsys)
    assert status == 2
    assert printed.out == ""
    assert f"error: {path}: " in printed.err


def test_demo_exits_with_2_on_a_file_that_is_missing(tmp_path, capsys):
    missing = tmp_path / "none.jsonl"
    status, printed = _exit_status(["--replay", str(missing)], capsys)
    assert status == 2
    assert printed.out == ""
    # The demo names the file, whether or not the system's error does.
    assert f"error: {missing}: " in printed.err


@pytest.mark.parametrize("scenario,schedule", _KNOWN, ids=_NAMES)
def test_demo_replays_a_file_on_the_scenario_it_names(
    scenario, schedule, tmp_path, capsys
):
    run = explore.replay(scenario, schedule)
    path = tmp_path / "run.jsonl"
    path.write_text(tracefile.dumps(scenario, run), encoding="utf-8")
    assert demo.main(["--replay", str(path)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        f"{scenario.name}: the file replays to the same trace")


def test_demo_knows_the_scenarios_the_readme_lists(tmp_path, capsys):
    known = _names_in_the_refusal(tmp_path, capsys)
    assert known == _NAMES
    _assert_readme_shows("\n".join(known))


def test_demo_exits_with_2_on_a_file_of_a_scenario_it_does_not_know(
    tmp_path, capsys
):
    path, _ = _saved_failure(tmp_path, capsys)
    text = path.read_text(encoding="utf-8")
    assert text.count('"oversell-naive"') == 1
    path.write_text(
        text.replace('"oversell-naive"', '"oversell-other"'),
        encoding="utf-8")
    status, printed = _exit_status(["--replay", str(path)], capsys)
    assert status == 2
    assert printed.out == ""
    assert f"error: {path}: " in printed.err
    assert (
        "the file holds a run of 'oversell-other', which the demo does not "
        "know. It knows oversell-naive, oversell-atomic, lost-update-naive, "
        "lost-update-cas") in printed.err


def test_demo_does_not_replay_a_file_of_a_scenario_built_otherwise(
    tmp_path, capsys
):
    # The file names the naive buyer scenario, and its run is of two
    # units. The demo brings the build it knows by that name, of one unit,
    # where the second buyer reads 0 and stops.
    scenario = oversell.naive(stock=2, buyers=2)
    run = explore.replay(scenario, ("buyer-1",) * 3 + ("buyer-2",) * 3)
    path = tmp_path / "two-units.jsonl"
    path.write_text(tracefile.dumps(scenario, run), encoding="utf-8")
    with pytest.raises(
            scheduler.ReplayDivergenceError,
            match="the run ended after 4 steps but the schedule has 6"):
        demo.main(["--replay", str(path)])
    assert capsys.readouterr().out == ""


def test_demo_exits_with_2_when_the_file_cannot_be_written(tmp_path, capsys):
    path = tmp_path / "none" / "failure.jsonl"
    status, printed = _exit_status(["--save", str(path)], capsys)
    assert status == 2
    assert f"error: {path}: " in printed.err
    assert not path.exists()


def test_demo_refuses_save_together_with_replay(tmp_path, capsys):
    # The file to replay is a good one, so that only the two flags
    # together can be what ends the demo.
    good, _ = _saved_failure(tmp_path, capsys)
    other = tmp_path / "other.jsonl"
    status, printed = _exit_status(
        ["--save", str(other), "--replay", str(good)], capsys)
    assert status == 2
    assert printed.out == ""
    assert "not allowed with" in printed.err
    assert not other.exists()


def test_demo_run_as_a_program_takes_its_command_line(tmp_path, capsys):
    path, _ = _saved_failure(tmp_path, capsys)
    root = pathlib.Path(__file__).parent.parent

    def run(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "anyorder", *arguments],
            cwd=root, capture_output=True, text=True, check=False)

    replayed = run("--replay", str(path))
    assert replayed.returncode == 0
    assert replayed.stdout.splitlines()[0] == (
        "oversell-naive: the file replays to the same trace")
    assert run("--replay", str(tmp_path / "none.jsonl")).returncode == 2
    # The search found a violation, and the program ends with what main
    # returned.
    assert run().returncode == 1


def test_demo_refuses_an_argument_it_does_not_know(capsys):
    status, printed = _exit_status(["--seeds", "5"], capsys)
    assert status == 2
    assert printed.out == ""
