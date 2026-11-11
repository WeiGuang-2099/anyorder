# anyorder

By [WeiGuang-2099](https://github.com/WeiGuang-2099).

A tester for concurrency bugs in multi-agent systems. When several agents
share state through tools, the result can depend on the order their tool
calls happen to run in. anyorder controls that order, searches for
an order that breaks an invariant, and reports it as a schedule that replays
the failure exactly.

## Status

Early. What works today:

* A deterministic scheduler. Agents are coroutines, every tool call is a
  scheduling point, and nothing reads the clock or starts a thread.
* Seeded random search over schedules.
* Exhaustive search over every schedule of a small scenario.
* Replay of a recorded schedule. A run that leaves its schedule is refused,
  not tolerated.
* One example scenario, an oversell, with a control that must pass.

Agents are scripted Python for now. Model calls, fault injection, schedule
shrinking and adapters for agent frameworks are not built yet.

## Run

    pip install -e ".[test]"
    python -m pytest -q
    python -m anyorder

The last command searches the naive buyer scenario and prints the first
schedule that oversells:

    oversell-naive: violation found at seed 4
      2 orders placed against a stock of 1: buyer-2, buyer-1
    schedule: buyer-1 buyer-2 buyer-1 buyer-2 buyer-2 buyer-1
      1. buyer-1: read_stock() -> 1
      2. buyer-2: read_stock() -> 1
      3. buyer-1: write_stock(value=0) -> None
      4. buyer-2: write_stock(value=0) -> None
      5. buyer-2: place_order(buyer='buyer-2') -> None
      6. buyer-1: place_order(buyer='buyer-1') -> None

Both buyers read a stock of 1 before either writes, so both place an order.
Run one after the other, the same two buyers sell one unit.

## How it works

* An agent reaches its world only through `AgentContext.call`. Each call
  suspends the agent and hands the scheduler one pending tool call.
* The scheduler holds one pending call per unfinished agent and asks a
  policy which runs next. The list of chosen agent names is the schedule.
* `explore.explore` runs one random schedule per seed until an invariant
  fails. It replays the failing schedule and checks that the trace is the
  same before returning it.
* `explore.exhaust` runs every complete schedule of a scenario, depth
  first, and replays each run before yielding it.
* `explore.replay` runs a scenario under a given schedule.

Seeds find schedules. The schedule is what a failure is stored and reported
as.

## Writing a scenario

A scenario is a world (any object whose public methods are the tools), a set
of agents, and invariants checked once every agent has finished. See
`anyorder/scenarios/oversell.py`. Every scenario with a bug has a
control that differs only in the fix.

## Limits

* Invariants are checked only when all agents have finished.
* A tool that raises ends the run.
* Search is uniform random or exhaustive. Neither prunes equivalent
  schedules or shrinks a failing one.
* Exhaustive search has no bound. Every agent must finish under every
  schedule, and the number of schedules can grow exponentially with the
  number of calls and factorially with the number of agents.
* Replaying a run checks that it gives the same trace twice. That probes
  whether a scenario is deterministic and does not prove it.
* The example world is synthetic. Nothing here has been run against a real
  agent framework or a real model.

## License

MIT. See [LICENSE](LICENSE).
