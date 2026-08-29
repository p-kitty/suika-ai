# AGENTS.md — how to work in this repo

**Only two things to check before touching anything.** If you are on `master`, cut a branch
(→[git](#git)). If the work falls under [When in doubt, ask before touching anything](#when-in-doubt-ask-before-touching-anything),
ask instead of starting. The remaining sections are conventions per area,
so read them when you touch that area.

## Who owns which document

Each of the three has a separate role. **Do not write the same thing in two places**.

- **AGENTS.md (this file)** … how to work. Commands, code conventions, how changes proceed,
  repo-specific traps. Write it so "the next person can follow the same steps".
  **Do not write measured numbers (score, n, significance, measured run time).** What you may write is
  defaults and thresholds you would stop to look up when typing a command (`--episodes 50`, `--skip 60`), and
  the order of magnitude of run cost that changes whether to ask (seconds / minutes / hours).
  Link to NOTES.md for the measurement behind such a value
- **[NOTES.md](NOTES.md)** … domain knowledge and experiment log for the game and the policy. What measurements showed,
  failed attempts, the current penalty rules, open tasks. "Why this design", "what did it score"
- **[README.md](README.md)** … the entry point for humans (English). Setup, how to run, layout

Where things go when unsure:

| What you want to write | Where it goes |
|---|---|
| How to run tests / lint / scripts, default arguments | AGENTS.md |
| Why a default has that value | NOTES.md (link from AGENTS.md without the numbers) |
| Measurement results and conclusions such as score, n, CI, run time | NOTES.md |
| Penalty weights and their meaning, policy design decisions | NOTES.md |
| Procedures such as "plug A/B changes into `_apply_variant`" | AGENTS.md (the resulting numbers go to NOTES.md) |
| Attempts that had no effect, things decided against | NOTES.md |
| The history of changing one physics or drawing constant | commit message (no NOTES.md section) |
| A module / script was added or renamed | README.md Layout (AGENTS.md if it needs a convention) |

If you feel like writing details in AGENTS.md, do not; link to a NOTES.md section instead.

### Do not only append to NOTES.md

The bar for keeping something is "**does it stop the next person who starts down the same road**".

**First, fixes that go straight into `master` get no section.** Bug fixes,
changes that close a hole in a rule, and new rules can be found if the commit message records the symptom, the cause and how it was checked,
because `git log` keeps them. NOTES.md keeps only **what stays unresolved** (symptoms not yet fixed,
ideas decided against, traps the next person is likely to hit).
**Indexes of the current state are different**, such as [Current penalty rules](NOTES.md#current-penalty-rules):
when you add a rule, update them in the same diff.

- **Keep** … what investigation showed, measured numbers, **failures likely to recur**. Ideas that had no effect,
  traps stepped in, reasons for deciding against something. Without these the same experiment gets run twice
- **Delete** … **problems that are settled and will not come back**. Investigation history of fixed bugs, reverted ideas
  no one will bring back, old text that contradicts the current implementation. The history stays in commit messages and `git log`,
  so it may be removed from NOTES.md

If closed problems keep piling up, NOTES.md grows without bound and the crucial "do not step here"
gets buried. **When adding a section, check at the same time for sections that have served their purpose.**

[CLAUDE.md](CLAUDE.md) is the file Claude Code reads automatically; its content is a single line that
loads AGENTS.md. **It is an entry point, not a place to put things.**

**Do not use the agent-side memory feature.** Anything memorized outside the repo (`~/.claude/**/memory/`)
is invisible to other agents and to `git log`, and the same thing ends up scattered in two places.
Everything meant to last, including instructions and policies received from the user, goes into one of the three files above.

## When in doubt, ask before touching anything

**Confirm once before starting anything that is expensive to redo.** Asking after you start
wastes all the time already spent.

Ask first:

- **When running a script that takes `--workers`.** The four scripts are `compare_policy.py`,
  `compare_b_only.py`, `eval_policy.py` and `train_sim.py`; **always ask first, without exception**.
  **The line is the script name, not the run time**. Judging by "it will probably be short"
  wastes exactly the amount by which the estimate is wrong. Background runs are the same.
  Include two things when asking:
  - The expected time from `--episodes` × `--max-steps`, and what that n can tell us
    (→[How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in). Examples that take hours
    are listed there)
  - An option with fewer `--workers`. The user actually plays the real Suika Game on the same machine,
    and when every core is busy the game stutters and becomes unplayable
- **When overturning a decision settled in NOTES.md.** To bring back or delete something marked
  "made permanent", "reverted" or "won't do", present the evidence and confirm
- **When changing the design.** Splitting or renaming modules, large refactors, rebuilding features or
  the network architecture, adding keys to `config/config.json`, adding dependencies
  (scipy is deliberately not installed)
- **When touching detection thresholds or physics constants.** They ripple into the ground truth in `screenshots/` and the fidelity of `SimEnv`.
  Fixing one place breaks another scene
- **When deleting.** Even things that look unused, like `src/ladder.py`, can be investigation groundwork
  kept on purpose. Confirm before deleting anything vulture lists too
- **When a request can be read two ways and misreading it wastes the whole job**

OK to proceed without asking:

- Bug fixes, adding tests, fixing lint, recording in NOTES.md
- Cutting a branch, committing ([git](#git))
- Runs that do not take `--workers` — `pytest`, and screening that calls `choose_x` serially on a single thread
  (`band_escape.py` and the like)
- When a request reads two ways but the deliverable is nearly the same either way

Ask everything in one go. Include the options and a recommendation, and keep doing the parts that do not depend on the answer.

## Environment

- Windows / PowerShell. `.venv` sits at the repo root, and `python` and `pytest` point there
- **`ruff` / `basedpyright` / `vulture` are not in the venv**. They live in the global
  `Python310\Scripts`, so the bare names work even with the venv active
- `main.py` grabs the real game screen and input. Agents do not launch it on their own.
  Check behavior on the sim side under `scripts/`

## Common commands

```powershell
pytest                 # finishes in seconds. Every time you change something
basedpyright           # types. Pre-existing errors may remain, so check whether they come from your diff
ruff check --fix       # unused imports / variables only (select = F401, F841)
vulture                # unreferenced functions and constants. False positives; check callers before deleting
```

Examples of sim evaluation, A/B and training runs are in the README Scripts section and
[Training in NOTES.md](NOTES.md#training). Long ones take hours.

## Code conventions

- Docstrings, comments, identifiers and commit messages are all in **English**
- Comments say **why this value or this shape**, not "what it does"
  (measured values, traps hit, alternatives discarded). The top of `src/policy.py` and its constants are the model
- But **do not accumulate change history in comments**. When changing a constant, replace only the value;
  "it used to be X" and "why it was lowered" go in the commit message
- `from __future__ import annotations` + type annotations. `typeCheckingMode = "basic"`
- Prefix module-internal names with `_`. Only names called across modules are public
- **Do not bind weights or functions of `src/penalties.py` with `from .penalties import X`.**
  Reference them as `pen.X` like `src/policy.py` does. The A/B in `compare_policy.py`
  swaps them by rewriting module attributes, so binding makes A and B run the same policy
- `scripts/*.py` follow the pattern of `sys.path.insert` → `from scripts._bootstrap import ROOT`
  at the top (do not write a function that adds the path)
- Tests do not pin concrete procedures. Assert **properties of the policy** such as merging, danger avoidance and
  accident prevention (the policy at the top of `tests/test_policy.py`). Fall physics lives in `tests/sim/test_sim_physics.py`

## When touching the policy or training

Changes that alter play go through the order below. Only a pure refactor that is not meant to change play
is different: it is enough to confirm that **x / score / penalties / merge match over a few hundred moves on the same seeds**
(if they do not match it is not a refactor, so it goes through the order).

### Order of steps

**Top to bottom. Do not move on with a step skipped.** Steps 1-3 each take minutes,
and many changes actually fail here.

1. **Read [How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in).**
   Score noise is large; a rise or fall in the mean alone says nothing
2. **Check in view_sim that no obvious blunders remain.** If they do, do not go to the A/B;
   fix them first (→[Do not run an A/B while obvious blunders remain](#do-not-run-an-ab-while-obvious-blunders-remain))
3. **Screen with `python scripts/band_escape.py`.** Inside the tie band the choice is indifferent, so
   ([measured](NOTES.md#settled-the-tie-band-really-is-indifferent-2026-08-19)), so "what fraction of moves change"
   is not a screen. Look at **the fraction that escapes the band**. If that is a few %, running the A/B
   score does not move
4. **Run the A/B** (it takes `--workers`, so ask first →
   [When in doubt, ask before touching anything](#when-in-doubt-ask-before-touching-anything)). Plug the change into
   `_apply_variant` in `scripts/compare_policy.py`. The default is
   `--episodes 50 --max-steps 400`, and **a change without a significant difference here is
   considered not worth adding**. Do not keep piling up n until it becomes significant (what this screen
   throws away is in [How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in)).
   **Side A is the same policy across variants, so do not rerun it.** Once saved with `--out`,
   run only B from then on with `scripts/compare_b_only.py --baseline <that json>`.
   It halves the compute. Discard the baseline when the policy itself changes
   (a warning appears when `baseline_commit` in the JSON differs from HEAD)
5. **Make it permanent.** Revert the variant in `_apply_variant`, and **leave no ON/OFF toggle
   in the code**. To compare with another commit, see the worktree item under [git](#git)
6. **Fix NOTES.md in the same diff.** If you added or removed a rule or changed a weight, update
   [Current penalty rules](NOTES.md#current-penalty-rules).
   **Keep attempts that had no effect too** — the content, n, conclusion and the fact it was reverted.
   The record exists so nobody walks the same road twice; keeping only the successes makes it pointless

### Do not run an A/B while obvious blunders remain

**If view_sim shows plainly strange moves, measuring score in that state
tells you nothing.** Fix the blunders first. Score noise always swallows defects at this granularity
(→[How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in)).

When a blunder seed and symptom come up, replay that one game and follow it move by move:

1. Note the move number from **move** in the `view_sim.py --seed <value>` footer (1-based).
   The `SimEnv` replay loop is 0-based, so move number `move` is `i = move - 1`
2. Replay with `SimEnv(seed=...)` and record the board, inversion rate, trapped count and crown per move
3. **The move where the collapse becomes visible is often the result, not the cause.**
   Go back to "the last move where the board was clean" and read forward from there
4. Lay out **all candidates** of the suspicious move with eval broken down per term. They fall into two kinds,
   and **the remedies are completely different, so always tell them apart**:
   - **The top candidates are tied** … inside the band. eval has no term that makes a difference. Tuning weights
     does not fix it (→[Settled](NOTES.md#settled-the-tie-band-really-is-indifferent-2026-08-19))
   - **A single term decides the order at the top** … suspect that term's definition. Not the weight:
     check whether its firing condition matches reality
5. Whether the move you want to fix gets fixed is checked **by tuning weights on that one position**. Deterministic, seconds.
   Look at the move chosen when the term is cut and at its outcome (where the big fruits go)
6. For a broad view, measure agreement and structural metrics over the position set of several traces.
   It is a screen before betting hours on an A/B, and it runs on a single thread

**Do not compare score by playing one game through.** Changing one move makes the board diverge completely,
so the score difference cannot be told apart from draw luck. All one game can tell you is "was that move fixed"
and "did a new pathology appear".

### How to write a rule

- **Do not penalize board properties the current move cannot change.** A term with the same value for every candidate
  adds equally to all of them and does not change the ranking. No matter how much you multiply the weight.
  Look at the shape the dropped fruit itself creates
  (→[Properties that cannot be changed](NOTES.md#do-not-penalize-board-properties-the-current-move-cannot-change-2026-08-21))
- **One rule per term. Split by the number of weights, not by the complexity of the condition.**
  Something like the corner pocket that looks at "wall-anchored + outside + below + depth" is fine as one rule.
  **If it needs two weights there are two rules**, so split the function, or at least make the weights
  separate module constants. Left combined, `_apply_variant` cannot cut just one of them,
  and the A/B can only measure them "together". **A dead rule hides in the shadow of
  a live one** (→[Split composite terms into sub-terms](NOTES.md#split-composite-terms-into-sub-terms-2026-08-21))
- **Put rule weights in module constants in `src/penalties.py`.** Written as function locals,
  `_apply_variant` cannot reach them and that rule alone cannot be put through an A/B

### Before deleting a rule

- **Some terms work in the early game even when their band escape is 0%.** The screening default is
  `--skip 60`, which collects only the late game. Terms divided by the number of fruits are stronger on sparse boards,
  so deleting based on that view alone breaks early-game behavior. **Before deleting, pass `pytest` and
  rerun with `--skip 0` too** (→[Terms divided by an average](NOTES.md#terms-divided-by-an-average-thin-out-as-the-board-fills-2026-08-21))
- **If an `early_*` metric is significant in an A/B, that is evidence that "something changed".** It is a separate matter from
  "it does not correlate with score, so it is not a proxy". Do not ignore it and go delete things

## git

All git operations are collected in this section. Do not scatter them into other sections.

**branch**

- **Do not work on `master`.** Cut a branch before touching anything:
  `git switch -c <topic>` (e.g. `fix-wall-friction`, `docs-agents-md`)
- PRs target `main`

**commit**

- As long as you are on a branch, **you may commit without asking**. Commit at each checkpoint.
  Conversely, if you notice you are on `master`, do not commit; cut a branch first
- Conventional Commits (`feat:` `fix:` `refactor:` `docs:` `perf:`), in English,
  with an imperative one-line summary
- The body says **why** and **how it was checked**. For a pure refactor, include
  how play invariance was verified. Recent commits are the model
- Run `pytest` before committing (seconds). If you changed the policy, report numbers following
  [How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in)

**merge**

- **Do not fast-forward.** When merging into `master`, always use
  `git merge --no-ff <topic>`. Create a merge commit even if the branches have not diverged
- Why: it shows afterwards where one piece of work began and ended. Squashing with ff
  just lines the topic commits up on master, and neither the target for reverting as a unit (`git revert -m 1`)
  nor the extent of that work can be read any more
- The merge commit summary can be the default `Merge branch '<topic>'`.
  If the series of changes needs explaining, write it in the body
- Run `pytest` on the topic branch before merging

**What is tracked and what is not**

- `screenshots/` is ground truth transcribed by eye, so it is tracked
- When adding or retaking `screenshots/`, paint over the player display name at the top left with the sky color
  before committing. The repo is public, so do not show real names or account names
- `artifacts/` `debug/` change on every run, so they are ignored

**When tracing the history of a constant**

- `git log -L <line>,<line>:<file>`. Shows the history of just those lines with diffs
- **Do not use `-S`.** It looks at changes in the number of occurrences, so rewriting a value like `14.0` → `7.0`
  slips through, and **it returns a different commit without any error**.
  To follow a string, use `-G` (add `--follow` to cross renames)

**When comparing with another commit**

- Do not write automation that runs `git stash` / `git checkout` in the live working tree.
  It once caused an accident that left uncommitted changes stranded in the stash
  (recovered with `git fsck --unreachable`)
- Use a separate isolated tree with `git worktree add`
