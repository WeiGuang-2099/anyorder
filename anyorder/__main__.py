"""Finds the oversell in the naive buyer scenario, shrinks it and prints it.

With --save the shrunk failure is also written to a trace file. With
--replay nothing is searched: the run a trace file holds is replayed and
printed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

from anyorder import explore
from anyorder import scheduler
from anyorder import shrink
from anyorder import tracefile
from anyorder.scenarios import oversell

_SEEDS = range(100)


def _format_step(index: int, step: scheduler.Step) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in step.args.items())
    return f"  {index}. {step.agent}: {step.tool}({args}) -> {step.result!r}"


def _print_run(result: scheduler.RunResult) -> None:
    message = "no violation"
    if result.violation is not None:
        message = result.violation.message
    print(f"  {message}")
    print(f"schedule: {' '.join(result.schedule)}")


def _print_steps(result: scheduler.RunResult) -> None:
    for index, step in enumerate(result.steps, start=1):
        print(_format_step(index, step))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m anyorder")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save", metavar="FILE", type=pathlib.Path,
        help="also write the shrunk failure to FILE as a trace file")
    group.add_argument(
        "--replay", metavar="FILE", type=pathlib.Path,
        help="search nothing: replay the run the trace file FILE holds")
    return parser


def _load(
    parser: argparse.ArgumentParser, path: pathlib.Path, name: str
) -> tracefile.Saved:
    """Reads a trace file of the scenario with this name.

    A file that cannot be read, is not a trace file or holds a run of
    another scenario ends the demo with status 2.
    """
    try:
        # As bytes: a text file read as text may have its line ends changed.
        saved = tracefile.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as error:
        parser.error(f"{path}: {error}")
    if saved.scenario != name:
        parser.error(
            f"{path}: the file holds a run of {saved.scenario!r}, and the "
            f"demo replays {name!r}")
    return saved


def _save(
    parser: argparse.ArgumentParser, path: pathlib.Path, text: str
) -> None:
    """Writes a trace file. A file that cannot be written ends the demo."""
    try:
        # As bytes, so that the file is the same bytes on every system.
        path.write_bytes(text.encode("utf-8"))
    except OSError as error:
        parser.error(f"{path}: {error}")


def main(argv: Sequence[str] = ()) -> int:
    """Prints the first failing schedule, shrunk, or replays a trace file.

    Args:
        argv: The command line after the program's name.

    Returns:
        1 if the search found a violation and 0 if it found none. With
        --replay, 0: the file replayed to the trace it holds, whether or
        not that run broke an invariant.

    Raises:
        ReplayDivergenceError: The file given to --replay replays to
            another trace than it holds. Nothing is printed to standard
            output.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    scenario = oversell.naive()
    if arguments.replay is not None:
        saved = _load(parser, arguments.replay, scenario.name)
        result = tracefile.replay(scenario, saved)
        print(f"{scenario.name}: the file replays to the same trace")
        _print_run(result)
        _print_steps(result)
        return 0
    failure = explore.explore(scenario, _SEEDS)
    if failure is None:
        print(f"{scenario.name}: no violation in {len(_SEEDS)} schedules")
        return 0
    found = failure.result
    print(f"{scenario.name}: violation found at seed {failure.seed}")
    _print_run(found)
    shrunk = shrink.shrink(scenario, found).result
    before, _ = shrink.size(found)
    after, _ = shrink.size(shrunk)
    print(f"shrunk from {before} context switches to {after}")
    _print_run(shrunk)
    _print_steps(shrunk)
    if arguments.save is not None:
        _save(parser, arguments.save, tracefile.dumps(scenario, shrunk))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
