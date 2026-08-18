# AGENTS.md — how to work in this repo

## Who owns which document

Each of the three has a separate role. **Do not write the same thing in two places**.

- **AGENTS.md (this file)** … how to work. Commands, code conventions, how changes proceed,
  repo-specific traps. Write it so "the next person can follow the same steps". No numbers
- **[NOTES.md](NOTES.md)** … domain knowledge and experiment log for the game and the policy. What measurements showed,
  failed attempts, the current penalty rules, open tasks. "Why this design", "what did it score"
- **[README.md](README.md)** … the entry point for humans (English). Setup, how to run, layout

Where things go when unsure:

| What you want to write | Where it goes |
|---|---|
| How to run tests / lint / scripts | AGENTS.md |
| Measurement results and conclusions such as score, n, CI | NOTES.md |
| Penalty weights and their meaning, policy design decisions | NOTES.md |
| Procedures such as "plug A/B changes into `_apply_variant`" | AGENTS.md (the resulting numbers go to NOTES.md) |
| Attempts that had no effect, things decided against | NOTES.md |
| The history of changing one physics or drawing constant | commit message (no NOTES.md section) |
| A module / script was added or renamed | README.md Layout (AGENTS.md if it needs a convention) |

If you feel like writing details in AGENTS.md, do not; link to a NOTES.md section instead.

[CLAUDE.md](CLAUDE.md) is the file Claude Code reads automatically; its content is a single line that
loads AGENTS.md. **It is an entry point, not a place to put things.**

**Do not use the agent-side memory feature.** Anything memorized outside the repo (`~/.claude/**/memory/`)
is invisible to other agents and to `git log`, and the same thing ends up scattered in two places.
Everything meant to last, including instructions and policies received from the user, goes into one of the three files above.

## When in doubt, ask before touching anything

**Confirm once before starting anything that is expensive to redo.** Asking after you start
wastes all the time already spent.

Ask first:

- **When running a measurement that takes hours.** Estimate the time from `--episodes` × `--max-steps`,
  and confirm together with what that n can tell us ([How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in)).
  There are examples such as 2.6 hours at n=100 and n=529 = 14 hours to reach significance
- **When running something that fills the CPU. The line is the degree of parallelism, not the run time.** The user actually plays
  the real Suika Game on the same machine. When something taking `--workers` (`compare_policy.py`,
  `eval_policy.py`, `train_sim.py`) fills every core, the game stutters and becomes
  unplayable. Background runs are the same. Ask with an option of fewer `--workers`.
  Screening that only calls `choose_x` serially on a single thread and pytest need no asking
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
- Short sim runs (ones that finish in minutes)
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
pytest                 # about 160 tests, just under 8 seconds. Every time you change something
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

- **Read [How to measure](NOTES.md#how-to-measure-traps-we-keep-stepping-in) before reporting numbers.**
  Score noise is large; a rise or fall in the mean alone says nothing
- Run an A/B by plugging the change into `_apply_variant` in `scripts/compare_policy.py`.
  When making it permanent, revert the variant and **leave no ON/OFF toggle in the code**.
  To compare with another commit, see the worktree item under [git](#git)
- When adding or removing a penalty rule or changing a weight, update NOTES.md's
  [Current penalty rules](NOTES.md#current-penalty-rules) in the same diff
- **Keep attempts that had no effect in NOTES.md too.** Write the content, n, conclusion and that it was reverted.
  The record exists so nobody walks the same road twice; keeping only the successes makes it pointless
- For a pure refactor, confirm play does not change
  (whether x / score / penalties / merge match over a few hundred moves on the same seeds)

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
- Run `pytest` before committing (just under 8 seconds). If you changed the policy, report numbers following
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
