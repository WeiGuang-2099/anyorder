# anyorder

By [WeiGuang-2099](https://github.com/WeiGuang-2099).

A tester for concurrency bugs in multi-agent systems. When several agents
share state through tools, the result can depend on the order their tool
calls happen to run in. anyorder controls that order, searches for
an order that breaks an invariant, shrinks it, and reports it as a schedule
that replays the failure exactly.

## Status

Early. What works today:

* A deterministic scheduler. Agents are coroutines, every tool call is a
  scheduling point, and nothing reads the clock or starts a thread.
* Seeded random search over schedules.
* Exhaustive search over every schedule of a small scenario.
* Replay of a recorded schedule. A run that leaves its schedule is refused,
  not tolerated.
* Shrinking of a failing schedule toward fewer context switches, then fewer
  agents.
* One example scenario, an oversell, with a control that must pass.

Agents are scripted Python for now. Model calls, fault injection and
adapters for agent frameworks are not built yet.

## Run

    pip install -e ".[test]"
    python -m pytest -q
    python -m anyorder

The last command searches the naive buyer scenario for a schedule that
oversells, shrinks it and prints the trace of the shrunk one:

    oversell-naive: violation found at seed 4
      2 orders placed against a stock of 1: buyer-2, buyer-1
    schedule: buyer-1 buyer-2 buyer-1 buyer-2 buyer-2 buyer-1
    shrunk from 4 context switches to 2
      2 orders placed against a stock of 1: buyer-1, buyer-2
    schedule: buyer-2 buyer-1 buyer-1 buyer-1 buyer-2 buyer-2
      1. buyer-2: read_stock() -> 1
      2. buyer-1: read_stock() -> 1
      3. buyer-1: write_stock(value=0) -> None
      4. buyer-1: place_order(buyer='buyer-1') -> None
      5. buyer-2: write_stock(value=0) -> None
      6. buyer-2: place_order(buyer='buyer-2') -> None

Both buyers read a stock of 1 before either writes, so both place an order.
Run one after the other, the same two buyers sell one unit. In the shrunk
schedule one buyer reads, the other runs to its end, and the one that read
finishes on what it read.

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
* `shrink.shrink` looks for a smaller failure of the same invariant: fewer
  context switches before the invariant breaks, then fewer agents among
  the steps up to there. It moves steps of the failing schedule, runs each
  candidate, and replays the one it keeps before going on from it.

Seeds find schedules. The schedule is what a failure is stored and reported
as.

## Writing a scenario

A scenario is a world (any object whose public methods are the tools), a set
of agents, and invariants. Each invariant has a name and says whether it
must hold from the start and after every step, or only once every agent has
finished. See `anyorder/scenarios/oversell.py`. Every scenario with a bug
has a control that differs only in the fix.

## Limits

* A run goes on to its end after an invariant fails, and keeps only the
  first violation.
* A tool that raises does not end the run. Its agent gets the error's
  class name and text as the result of the call, and may ignore it. A call
  with the wrong arguments is such an error. A call to a tool the world
  lacks still ends the run, and so does an interrupt or one of the
  scheduler's own errors raised inside a tool.
* A call's arguments and a tool's result are kept in the trace by
  reference, so a list that is passed in or returned shows its final
  content.
* Search is uniform random or exhaustive. Neither prunes equivalent
  schedules.
* Shrinking ends at a failure that none of its moves makes smaller. That
  is not always the smallest one. The tests compare it with exhaustive
  search: for the sizes of the oversell scenario they cover, it reaches
  the smallest failure from every failing schedule, and they hold small
  scenarios where it ends above the smallest. A candidate that breaks
  another invariant before the one being shrunk is passed over.
* Exhaustive search has no bound. Every agent must finish under every
  schedule, and the number of schedules can grow exponentially with the
  number of calls and factorially with the number of agents.
* Shrinking has no bound either: every agent must finish under every
  candidate. Once a candidate's order runs out, the first waiting agent by
  name runs, so an agent that waits for another by trying again, as for a
  lock, can keep a candidate from ending although the given failure ended.
* Replaying a run checks that it gives the same trace twice. That probes
  whether a scenario is deterministic and does not prove it.
* The example world is synthetic. Nothing here has been run against a real
  agent framework or a real model.

## License

MIT. See [LICENSE](LICENSE).
