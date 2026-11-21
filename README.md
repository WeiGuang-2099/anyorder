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
* Calls by value. Every argument and every result is taken at its call, so
  a trace holds each value as it was then, and an agent and its world
  share nothing but what a call carries.
* Seeded random search over schedules.
* Exhaustive search over every schedule of a small scenario.
* Replay of a recorded schedule. A run that leaves its schedule is refused,
  not tolerated.
* Shrinking of a failing schedule toward fewer context switches, then fewer
  agents.
* Trace files. A run is written as JSON lines, read back and replayed. A
  file that replays to another trace than it holds is refused.
* A serial-equivalence oracle. A run fails when it leaves its world in a
  state that no serial order of the agents leaves, whether or not it broke
  an invariant.
* Example scenarios, an oversell, a lost update and a write skew, each
  with a control that must pass. The write skew also has a fix that must
  fail.

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

To keep the shrunk failure and replay it later:

    python -m anyorder --save oversell.jsonl
    python -m anyorder --replay oversell.jsonl

The first prints the same and also writes the shrunk run to a trace file,
in place of any file of that name. The second searches nothing. It runs
the schedule the file holds, checks every step and the violation against
the file, and prints:

    oversell-naive: the file replays to the same trace
      2 orders placed against a stock of 1: buyer-1, buyer-2
    schedule: buyer-2 buyer-1 buyer-1 buyer-1 buyer-2 buyer-2
      1. buyer-2: read_stock() -> 1
      2. buyer-1: read_stock() -> 1
      3. buyer-1: write_stock(value=0) -> None
      4. buyer-1: place_order(buyer='buyer-1') -> None
      5. buyer-2: write_stock(value=0) -> None
      6. buyer-2: place_order(buyer='buyer-2') -> None

A replay exits with 0 when the file replays, also when the run it holds
broke an invariant, where the search exits with 1 on a violation. A file
that cannot be read, is not a trace file or names a scenario the demo does
not know ends it with 2. A file that replays to another trace than it
holds ends it with an error that says where the two differ, at a step, at
the violation or in the number of steps, and the run is not printed.

A replay runs the scenario the file names. The demo knows these by name:

    oversell-naive
    oversell-atomic
    lost-update-naive
    lost-update-cas
    write-skew-naive
    write-skew-cas
    write-skew-atomic

It builds the oversell with a stock of 1 and 2 buyers, the lost update
with 2 editors and the write skew with 2 doctors. `oversell-atomic`,
`lost-update-cas` and `write-skew-atomic` are the controls. A file names
its scenario and nothing else of it. A file of a scenario built with other
numbers is replayed on the build above, and where its run differs it ends
the demo with the error of a file that replays to another trace.

The search runs the first of them. To search another, name it:

    python -m anyorder --scenario lost-update-naive

The search of a control finds no violation and exits with 0. `--save`
then writes no file, and a file of that name is left as it was.
`--scenario` is not taken together with `--replay`, whose file names its
scenario.

`write-skew-cas` is not a control but a fix that fails. Its search finds
the failure of `write-skew-naive` again and exits with 1:

    write-skew-cas: violation found at seed 4
      nobody is on call
    schedule: doctor-1 doctor-2 doctor-1 doctor-2
    shrunk from 3 context switches to 2
      nobody is on call
    schedule: doctor-2 doctor-1 doctor-1 doctor-2
      1. doctor-2: read() -> (('doctor-1', 0, True), ('doctor-2', 0, True))
      2. doctor-1: read() -> (('doctor-1', 0, True), ('doctor-2', 0, True))
      3. doctor-1: leave_if(person='doctor-1', version=0) -> True
      4. doctor-2: leave_if(person='doctor-2', version=0) -> True

Each doctor leaves if its own entry still has the version it read. Both
entries do, so both writes go through.

## How it works

* An agent reaches its world only through `AgentContext.call`. Each call
  suspends the agent and hands the scheduler one pending tool call.
* The scheduler holds one pending call per unfinished agent and asks a
  policy which runs next. The list of chosen agent names is the schedule.
* A call is by value. An argument is taken when the agent awaits the call
  and a result when the tool returns. The tool, the agent and the trace
  each get a copy of their own of every list and dict in it, at any depth,
  and share what cannot change, such as a string or a tuple of numbers.
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
* `tracefile.dumps` gives a run as the text of a trace file: JSON lines,
  one that names the format, its version and the scenario, one for each
  step with its agent, its call and the result, and one with the
  violation. It reads the text back and replays it before it returns it.
  `tracefile.loads` reads such a text, of the version that is written or
  of the one before it, and `tracefile.replay` runs the schedule it holds
  and refuses the file unless every step and the violation come out as
  the file holds them.
* `oracle.serial_outcomes` runs every serial order of a scenario's agents,
  each agent making all its calls before the next makes one, and keeps the
  state each order leaves its world in, as a snapshot function gives it.
  `oracle.judge` takes those outcomes and a run. It replays the run to
  look at its world and names the serial orders that end in the same
  state. With none the run failed, and it is replayed once more before
  that is reported: both replays must end in the same state.

Seeds find schedules. The schedule is what a failure is stored and reported
as.

## Writing a scenario

A scenario is a world (any object whose public methods are the tools), a set
of agents, and invariants. Each invariant has a name and says whether it
must hold from the start and after every step, or only once every agent has
finished. Every scenario with a bug has a control that differs only in the
fix. The examples are in `anyorder/scenarios`:

* `oversell.py`: buyers of limited stock. The naive buyer reads the stock
  and writes it back one lower in separate calls, so two buyers can both
  take the last unit. The control checks and decrements in one call.
* `lost_update.py`: editors that each add a line to one document. The
  naive editor reads the lines and writes them back with its own, so when
  two read the same lines the second write drops the line of the first.
  The control writes with a compare-and-set on the version it read, and
  reads again when that write is refused.
* `write_skew.py`: doctors on a rota, where one must stay on call. The
  naive doctor reads the rota and, if another doctor is on call, takes
  itself off call in a second call, so two doctors can each see the other
  on call and both leave. A compare-and-set on the doctor's own entry is a
  fix that fails: the doctors write different entries, so no write is
  refused. The control looks at the other entries and writes its own in
  one call.

A tool takes and returns the values a call carries: None, booleans,
integers, finite floats, strings, lists, tuples, dicts with string keys
and tool errors. It may return a list of its own, since the caller gets a
copy. A tuple of strings and numbers is cheaper: nothing in it can change,
so none of it is copied.

An invariant's check is asked about the world alone. One that kept a
memory of its own would carry it from one run into the next, and a replay
could then come out otherwise. A rule about the past needs a world that
keeps that past: the document of the lost update keeps every line it has
held, so that its invariant can ask whether one of them is gone.

For the serial-equivalence oracle a scenario also needs a snapshot
function, such as `oversell.snapshot` or `lost_update.snapshot`, which is
given to `oracle.serial_outcomes` with the scenario. It gives the state of
a world as a value that can be hashed and that is equal for two worlds in
the same state, such as a tuple of numbers and strings, and so says what
counts as ending the same.

## Limits

* A run goes on to its end after an invariant fails, and keeps only the
  first violation.
* A tool that raises does not end the run. Its agent gets the error's
  class name and text as the result of the call, and may ignore it. That
  result is true, so an agent that reads a result by truth takes an error
  for a yes. A call with the wrong arguments is such an error. A call to a
  tool the world lacks still ends the run, and so does an interrupt or one
  of the scheduler's own errors raised inside a tool.
* A call carries None, booleans, integers, finite floats, strings, lists,
  tuples, dicts with string keys and tool errors, each of exactly that
  type. A call with any other value among its arguments, such as a set,
  bytes, a dataclass, an enum or a NaN, ends the run with an error when
  the agent awaits it, before its tool runs. A tool that returns one ends
  the run after it ran. Neither error is handed to the agent. A search, a
  shrink, the oracle or the replay of a file that meets such a call under
  a schedule ends with that error, which names no schedule, and a failure
  that was found before it is not reported.
* A call is by value. A tool that fills a list it was given hands nothing
  back through it, and a list that is used as a handle is another object
  on the other side. The copies are made at every call: a large list or
  dict costs time whenever a call carries it, and a tuple of strings and
  numbers costs a look.
* A part that a value holds twice, as in `[part, part]`, is taken once for
  each place: a list is copied twice, and a tuple that needs no copy is
  looked through twice. So a value nested that way can take time that
  doubles with every level. Nothing bounds the size of a value.
* The nesting of a value is bounded: none is nested deeper than a line of
  a trace file lets it be. Taking a value near that bound needs about a
  hundred calls of Python's stack, and twice as many before Python 3.12,
  so a run that is started from deep inside other calls can end in a
  RecursionError where a shallower value would run.
* Agents that share an object outside every call, such as a list both
  were built with, still share it. The scheduler sees calls and nothing
  else.
* Search is uniform random or exhaustive. Neither prunes equivalent
  schedules.
* Shrinking ends at a failure that none of its moves makes smaller. That
  is not always the smallest one. The tests compare it with exhaustive
  search: for the sizes of the three examples that they cover, it reaches
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
* The control of the lost update tries again after a refused write, and
  still ends under every schedule: it reads again first, a write is
  refused only when another editor's went through since the read, and each
  editor's goes through once. An editor that made the same refused call
  again would never end, and nothing here would cut its run. Nor would the
  control end on a document whose compare-and-set raised every time: only
  True is a write that went through to it, and it tries again.
* The invariant of the lost update, that no line the document once held is
  gone, knows a line by its text. It is a rule for editors that only add
  lines, each a line of its own: two that add the same text can lose an
  update that it does not see. Nor does it see an editor that stops after
  a refused write, which loses no line and never adds its own. The
  serial-equivalence oracle sees both.
* The write skew reads the whole rota in one call, so it does not show a
  doctor that reads the entries one at a time. No compare-and-set is
  refused in it, so what its compare-and-set doctor does after a refusal
  is seen only in tests, on rotas that somebody else writes. Like the
  control of the lost update, that doctor, when it may leave, would ask
  again without end on a rota whose compare-and-set raised every time.
  The control of the write skew passes the serial-equivalence oracle
  whatever its tool does: each of its doctors makes one call, so every
  schedule is a serial order. What it does is checked by a test that
  counts the doctors that leave.
* The serial-equivalence oracle compares the state a run leaves and
  nothing else. What an agent read or was told is not compared, so a read
  that no serial order gives is seen only where a tool call wrote what
  came of it to the world. A failure that the serial orders share is not
  seen at all: that takes an invariant.
* A failure that only the oracle sees is reported as it was found.
  `shrink.shrink` takes a run that broke an invariant, and
  `explore.explore` and the demo search for one. The oracle has no bound
  either: it runs the n! serial orders of n agents, each twice, and every
  agent must finish in each of them.
* Replaying a run checks that it gives the same trace twice. That probes
  whether a scenario is deterministic and does not prove it.
* A trace file holds the values a call carries, but for two: a string that
  holds a surrogate, and an integer too long for Python to write as text.
  A call carries both, and a run with either is not written.
* A trace file of version 1 was written when a trace kept its values by
  reference, so that a list changed after its call was written with its
  final content. Such a file is still read. It replays where its run did
  not rest on a value being shared: none was changed after its call, and
  neither an agent nor a tool went by a list that the other still held.
* A trace file names its scenario and not what the scenario was built
  with, such as the stock or the number of editors. Whoever replays it
  brings the scenario, and the demo brings the build it knows by that
  name. The replay tells a wrong one only where its run of the schedule is
  written differently: a scenario built with other numbers that gives the
  same trace replays the file.
* A trace file is checked against a run of its schedule once that run is
  over, not call by call. A scenario that is not the file's can make
  every call of the schedule before it is refused.
* A tool's error is kept as the text Python gives it. That text can
  differ between versions of Python, and then a file that holds one does
  not replay on the other version.
* The example worlds are synthetic. Nothing here has been run against a
  real agent framework or a real model.

## License

MIT. See [LICENSE](LICENSE).
