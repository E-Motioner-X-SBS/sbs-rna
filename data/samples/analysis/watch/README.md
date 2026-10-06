# The training watch — two layers, and why

**Layer 1, durable.** `scripts/training_watch.sh` → `scripts/training_watch.py`,
fired by the system crontab at **:13 and :43** every hour. It reads the run
telemetry under `data/samples/analysis/runs/` plus the live checkpoints, and
names what is broken. It knows nothing about assistant sessions and does not
care whether one is alive, so it survives a closed terminal, a killed
session, a usage limit and a reboot.

It writes:

| file | what it is |
|---|---|
| `latest.md` | the most recent report, overwritten each fire |
| `report-<stamp>.md` | every report, kept |
| `history.jsonl` | one line per fire: counts and every CRIT/WARN — this is the **backlog** a session reads after being away |
| `ALERT` | exists **only while something is wrong**, so its absence is a real all-clear and not an absence of information |
| `heartbeat` | an ISO timestamp per fire, written even if the python half dies |
| `watch.log` | raw stdout/stderr |

`touch HOLD` in this directory stops it; remove the file to resume. Same
convention as the GPU runner, and for the same reason: a deliberate stop has
to survive the scheduler.

**Layer 2, in-session.** A `CronCreate` job at **:16 and :46** — three
minutes after layer 1, so the report is fresh — wakes the assistant to read
the report and do the judgement work: is this finding real, what caused it,
fix it, test it, record it.

**Layer 2 does not survive.** `CronCreate` holds its jobs in memory and
loses them when the session ends, which is exactly what a usage limit does.
That is the whole reason layer 1 exists separately. Nothing is lost while
the assistant is away — layer 1 keeps running and keeps appending to
`history.jsonl`, and the next session reads the backlog.

**To restore layer 2** after a limit reset or a new session, say *"resume the
training watch"*. Layer 1 needs no restoring; check it with
`crontab -l | grep training_watch` and `cat heartbeat`.

## What it checks

Each is aimed at the defect class this project keeps finding — a component
that exists, is measured, and whose output nobody checks for validity.

- **liveness / stalled** — a trainer running whose own telemetry has stopped
  moving. Scoped to the stage the live process actually owns.
- **non-finite** — any NaN or Inf in any logged column.
- **at the floor** — every `X_acc` against the `X_major` beside it. This is
  finding 75 generalised: head 2 reported 0.47 accuracy for a whole stage
  while its majority rate was 0.47, and nothing compared them.
- **macro at chance** — `X_macro` at 1/`X_n_class`.
- **collapsed** — `X_pred_sd` ≈ 0, which makes any correlation beside it 0
  by construction rather than by failure to correlate.
- **no lift** — `X_lift` ≤ 0 across a window.
- **never written** — a declared column no row of any kind carries, checked
  only once the run has produced an epoch row.
- **loss rising / loss spike** — second half against first; and a latest
  value 5 sd above its own mean.
- **frozen gate** — a zero-initialised gate still exactly 0.0 after 500
  steps, scoped to the stages that can train it and skipped on checkpoints
  that predate the parameter.
- **oom**, **disk**, **gpu**, **unreadable checkpoint**.

## False positives are the thing to watch

The first run produced 23 warnings and 22 were false: it reported stage 1
stalled while stage 2/3 was the live trainer, reported eleven validation
columns "never written" in a run that had not finished an epoch, reported
`coev_proj` frozen in stages that are not supposed to train it, and
reported **itself** as a trainer because its own `bash -c` line contained
the script names. All four are fixed and commented at the point of the fix.
A watcher that cries wolf is one nobody reads.
