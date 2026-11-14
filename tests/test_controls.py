"""Checks that each control differs from its scenario only in the agent.

What a control does is asked in the tests of each scenario, not here.
"""

import pytest

from anyorder.scenarios import lost_update
from anyorder.scenarios import oversell
from anyorder.scenarios import write_skew


@pytest.mark.parametrize("scenario,control", [
    (oversell.naive(), oversell.atomic()),
    (oversell.naive(stock=2, buyers=3), oversell.atomic(stock=2, buyers=3)),
    (lost_update.naive(), lost_update.cas()),
    (lost_update.naive(editors=3), lost_update.cas(editors=3)),
    (write_skew.naive(), write_skew.atomic()),
    (write_skew.cas(doctors=3), write_skew.atomic(doctors=3)),
], ids=["oversell", "oversell-of-two-units", "lost-update",
        "lost-update-of-three", "write-skew",
        "write-skew-fix-that-fails-of-three"])
def test_control_differs_from_its_scenario_only_in_the_agent(
        scenario, control):
    # Every scenario with a bug has a control that differs only in the
    # fix: here the same world, the same invariants and agents of the same
    # names, with one other agent function behind all of them.
    assert type(control.make_world()) is type(scenario.make_world())
    assert vars(control.make_world()) == vars(scenario.make_world())
    assert control.invariants == scenario.invariants
    assert list(control.agents) == list(scenario.agents)
    (broken,) = set(scenario.agents.values())
    (fixed,) = set(control.agents.values())
    assert fixed is not broken
    assert control.name != scenario.name
