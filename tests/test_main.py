"""Checks that the demo prints what the README says it prints."""

import pathlib

from anyorder import __main__ as demo

_README = pathlib.Path(__file__).parent.parent / "README.md"


def test_demo_prints_the_trace_in_the_readme(capsys):
    assert demo.main() == 1
    # The README shows the output as a block indented by four spaces.
    printed = [f"    {line}" for line in capsys.readouterr().out.splitlines()]
    readme = _README.read_text(encoding="utf-8").splitlines()
    start = readme.index(printed[0])
    end = start + len(printed)
    assert readme[start:end] == printed
    # The block ends where the output ends.
    assert not readme[end].startswith("    ")
