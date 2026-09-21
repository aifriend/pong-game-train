# Handover — Pong reinforcement-learning project

**Written:** 2026-09-21 · **Branch:** `master` · **Last commit:** `625d73a` · **Nothing is committed yet.**

Every change described here is in the working tree only. The test suite passes: 21 tests,
run three times in a row. Each change below was verified by running code, not by reading it.

---

## 1. Where this project was, and what was actually wrong

The agent is trained with Proximal Policy Optimization (PPO, from Stable-Baselines3) over a
five-phase curriculum against progressively harder scripted opponents. Phase 3, against the
medium-difficulty opponent, was reported as stuck below a 30 percent win rate against a
40 percent target, and there is a plan to fix that by tuning hyperparameters
(`.cursor/plans/improve_phase_3_training_2ce0bb56.plan.md`).

That plan targets the wrong thing. Five separate defects were found and fixed, and the
measurements below show the plateau was mostly an artefact of two of them.

## 2. What was changed

All five were confirmed by running code before being changed, and re-measured afterwards.

### 2.1 The per-phase learning rate never reached the optimizer

`scripts/train_ppo_curriculum.py`, in `train_phase`, around line 576.

When a phase continued from the previous phase's model, the code assigned
`model.learning_rate = ...`. Stable-Baselines3 builds its learning-rate schedule once
inside `_setup_model()` and reads that schedule, not the attribute, so the assignment did
nothing. **Every phase after the first trained at the Phase 1 rate of 3e-4.**

Proven on the installed Stable-Baselines3 2.7.1: building a model at 3e-4 and assigning
1.5e-4 left the optimizer at 0.0003. The fix adds `model._setup_lr_schedule()` after the
assignment, which moves the optimizer to 0.00015.

### 2.2 Phase gating measured with sampled actions and too few episodes

`scripts/train_ppo_curriculum.py`, two gate call sites, now at lines 676 and 721.

Both had a comment claiming deterministic evaluation while passing `deterministic=False`,
with 20 and 30 episodes. At a true win rate near 30 percent, 20 episodes give a standard
error of about 10 percentage points, so a 40 percent gate was about one standard error away
from the observed value — the decision was mostly noise.

Both now use greedy actions and 100 episodes, which brings the standard error to about
4.6 percentage points. The two *reporting* call sites (lines 913 and 975) were deliberately
left alone; they are not decisions.

### 2.3 The ball-trajectory prediction was wrong by the court aspect ratio

`pong/env/wrappers.py`, `_predict_ball_y_at_opponent`.

The observation normalizes horizontal position by the court width of 960 pixels and
vertical position by the height of 720, but divides *both* velocity components by one
shared scale. The predicted vertical travel therefore came out divided by the width where
it should have been divided by the height, making it too small by exactly four thirds.

The fix reads the real court dimensions from the wrapped environment in `__init__` and
applies the ratio. **Measured over 211 returns, the mean error against where the ball
actually arrived fell from 0.092 to 0.023 of court height.** For reference, always guessing
the middle of the court scores 0.201.

### 2.4 The pressure reward was paid on every step instead of once per return

`pong/env/wrappers.py`, `step`.

The offensive "pressure" shaping term was added on every single step the ball travelled
toward the opponent — a few hundred steps per crossing. At the Phase 3 scale of 0.12 per
step this paid a mean of **7.31 per crossing against a `point_reward` of 5.0**, so one
rally was worth more than scoring.

The consequence was decisive. Measured over 20 games with the reward decomposed (the
decomposition reproduces the wrapper's own output to within 0.07 percent):

| Reward component | Per episode, before |
|---|---|
| Pressure shaping | +27.80 |
| Tracking shaping | +16.30 |
| Hit reward | +0.39 |
| Points scored and conceded | −25.00 |
| Win or loss bonus | −10.00 |
| Step penalty | −2.19 |
| **Net** | **+7.30** |

**An agent that lost all twenty games five-nil still earned +7.30 per episode.** After the
fix the same run scores **−20.40**. Losing is finally penalised.

Pressure is now paid once, inside the existing hit-detection branch, and withheld on a
scoring step where a negative horizontal velocity belongs to a fresh serve rather than to a
shot the player hit. Verified: exactly 79 payments for 79 returns.

### 2.5 Draws and unfinished games were counted as defeats

`scripts/train_ppo_curriculum.py` (`evaluate_model`) and `scripts/evaluate_agent.py`
(`evaluate_against_opponent`) — the same defect, independently duplicated.

Both counted a win when the agent was ahead and reported everything else as a loss, so a
nil-nil game that simply ran out of steps was indistinguishable from a defeat. Both now
count four outcomes separately and expose them.

## 3. The measurement that reframes the whole project

Measured with the trained checkpoint `models/ppo_phase3_final.zip`, greedy actions,
40 episodes per row:

| Opponent | Ball speed | Step cap | Won | Lost | Drew | Unfinished | Reported win rate |
|---|---|---|---|---|---|---|---|
| slow | 0.6 | 5,000 | 37 | 0 | 3 | 40/40 | 92.5% |
| beginner | 0.7 | 5,000 | 20 | 0 | 20 | 40/40 | 50.0% |
| **medium** | **0.8** | **5,000** | **7** | **0** | **33** | **40/40** | **17.5%** |
| medium | 0.8 | 20,000 | 39 | 0 | 1 | 39/40 | 97.5% |
| normal | 0.9 | 5,000 | 3 | 2 | 35 | 40/40 | 7.5% |

**At the Phase 3 gate the agent loses zero games out of forty.** It was recorded at
17.5 percent against a 40 percent target only because no game finishes inside 5,000 steps
and 33 unfinished games were being called defeats. Given 20,000 steps it wins 39 of 40.

Worse: at a 20,000-step cap, twelve out of twelve games were *still* unfinished even though
the agent led in all twelve. Rallies at this ball speed are effectively endless, so nobody
reaches five points. `decided_win_rate` is therefore `None` at every setting tested — no
game in any configuration reached a real conclusion.

**The Phase 3 "plateau" is largely a measurement artefact, not a learning failure.** Any
work that starts from the hyperparameter plan without addressing this will be tuning against
a broken instrument.

## 4. Decisions waiting for Jose — do not make these unilaterally

**4.1 The step cap, or the points needed to win.** This is the big one. Games do not finish.
The options are raising `max_steps` well above 20,000, lowering `max_score` below 5, or
raising ball speed so rallies end. Each changes what the curriculum means. `max_steps` and
`max_score` were deliberately left untouched.

**4.2 Retuning `pressure_scale`.** The per-phase values (0.15 decaying to 0.08) were tuned
when the term accumulated over hundreds of steps. Now the whole payment for one perfectly
placed return is at most 0.15, which is **3 percent of the 5.0 point reward** — so fix 2.4
has effectively switched the offensive placement signal off rather than rebalanced it. A
reviewer that measured this suggested somewhere around 0.5 to 1.0 per return. Left
unchanged on purpose; it is a training-configuration choice, and the code comment says so.

**4.3 Whether to retrain.** Fixes 2.1, 2.3 and 2.4 all change training dynamics. Every
existing checkpoint was produced under the old behaviour, so no win-rate number recorded
before today is comparable to one recorded after.

## 5. Found, deliberately not fixed

- **The phantom hit reward.** The `_ball_approaching` flag goes stale across a point, so the
  hit branch fires about three times more often than the player actually hits the ball
  (37 firings against 12 real hits on one seed). The narrower pressure guard was chosen
  specifically so this behaviour stayed byte-identical rather than changing a second reward
  term unasked. It is a genuine separate defect.
- **The win rule is copied five times.** `evaluate_model`, `evaluate_agent.py`,
  `pong/env/wrappers.py:310` (`episode_stats["won"]`), the `WinRateLoggingCallback` in
  `train_ppo_curriculum.py:231` (which uses a *different* rule — it excludes unfinished
  games), and `pong/env/pong_headless.py:256` (`info["win_rate"]`, which nothing reads).
  Only the first two were fixed. The TensorBoard scalar `curriculum/win_rate` still uses
  the old rule, so it will disagree with the gate.
- **The trajectory prediction aims at the court edge**, `x = 0`, rather than the opponent's
  paddle plane 42.5 pixels in. That accounts for essentially all of the remaining 0.023
  error.
- **The closing summary evaluates every opponent at ball speed 1.0**
  (`train_ppo_curriculum.py:913`), including the slow opponent that Phase 1 trains at 0.6.

## 6. How to work here

The virtual environment did not exist and was created with `uv sync`. It holds
Stable-Baselines3 2.7.1, PyTorch 2.10.0, Gymnasium 1.2.3.

```bash
uv run --with pytest python -m pytest tests/ -q
```

`pytest` is not a project dependency, hence `--with pytest`.

Two traps when writing verification scripts:

- Start any scratch script with `import sys, os; sys.path.insert(0, os.getcwd())`, or the
  `pong` package will not import.
- Episodes at a 20,000-step cap are slow. A sweep of eight configurations at 40 episodes
  each takes roughly ten minutes.

## 7. State of the working tree

Changed by this work:

| File | What |
|---|---|
| `scripts/train_ppo_curriculum.py` | Fixes 2.1, 2.2, 2.5, plus corrected comments |
| `pong/env/wrappers.py` | Fixes 2.3, 2.4 |
| `scripts/evaluate_agent.py` | Fix 2.5 |
| `tests/test_ppo_environment.py` | `test_pressure_reward_bounded` rewritten |
| `docs/PPO_TRAINING.md`, `README.md` | Statements the fixes made false |

`test_pressure_reward_bounded` had to be rewritten: it drove the paddle upward for 2,000
steps, so it almost never returned the ball, and once pressure became one-shot its final
assertion failed in roughly a third of runs. It now tracks the ball, is seeded, and carries
a guard asserting that paid steps stay a small minority — which would catch any revert to
per-step payment.

Already modified before this work started, and untouched by it: `QUICKSTART.md`,
`pyproject.toml`, and the untracked `docs/CLI.md`, `main.py`, `pong/__main__.py`.

## 8. If you do one thing next

Settle section 4.1. Until games can finish, the win rate the gate reads is not measuring
the agent's ability, and neither the existing improvement plan nor any hyperparameter
search will produce a trustworthy result.
