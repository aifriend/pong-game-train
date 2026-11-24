# RL Coach: NeoHorse-Jev-4B as an offline decision advisor

The coach runs [NeoHorse-Jev-4B](https://github.com/TokenRhythm/NeoHorse), a
4B-parameter *structured decision model*, against the project's training metrics
and turns its decisions into bounded proposals for reward shaping, observation
design, and PPO hyperparameters.

**How it works**

NeoHorse-Jev-4B is prefill-only: it cannot write code or free text. The coach
therefore defines a menu of candidate decisions, and the model picks among them:

- **Noul** (yes/no probability) — condition checks: is the agent rally-farming,
  has the win rate plateaued, is the value function diverging, is entropy
  collapsing?
- **Choice** (candidate + full probability distribution) — which reward knob,
  hyperparameter, or observation feature to adjust, and in which direction.
- **Score** (ordered levels) — overall health of the phase being coached.

Every adjustment is bounded to ±10% or ±25% of the current value and clamped to
sane ranges, so a wrong decision can only nudge training, never derail it.

Decisions happen in two stages per session: first the model picks *which* knobs
to touch (plus the condition checks and phase health), then it picks the
adjustment magnitude for each chosen knob with that knob named in the question,
so the magnitude is conditioned on the choice.

**Nothing reaches training without your approval.** A coaching session writes a
proposal JSON containing the exact metrics the model saw, its answers with
probabilities, and the concrete config diff. You review the diff, then apply it:

```bash
# 1. See the metrics summary the coach reasons over
python scripts/coach.py --model models_v6/ppo_final collect

# 2. Run a coaching session (writes coach/proposals/proposal_*.json)
python scripts/coach.py --phase 4 propose --backend neohorse \
    --model-dir models_neohorse/NeoHorse-Jev-4B

# 3. Apply an approved proposal (writes coach/overrides.json)
python scripts/coach.py apply coach/proposals/proposal_*.json

# 4. Train with the overrides
python scripts/train_ppo_curriculum.py --phase 4 --overrides coach/overrides.json
```

`coach.py status` shows the currently active overrides.

## One-time setup

The model bundle (~9 GB) and the native runtime live outside the project venv:

```bash
# Model bundle (backbone + tokenizer + decision head + runtime wheel)
mkdir -p models_neohorse
.venv-neohorse/bin/hf download TokenRhythm/NeoHorse-Jev-4B \
    --local-dir models_neohorse/NeoHorse-Jev-4B

# Runtime wheel into the dedicated venv (create with: uv venv .venv-neohorse --seed)
uv pip install --python .venv-neohorse/bin/python --no-deps \
    models_neohorse/NeoHorse-Jev-4B/dist/neohorse_decision-1.0.0-py3-none-any.whl
uv pip install --python .venv-neohorse/bin/python pydantic
```

The first coaching run loads the model; on this Mac (M3 Max) inference runs on
Apple Silicon via MPS. Use `--backend mock` for a deterministic stand-in that
needs no model (used by the test suite).

## What the coach can change

Overridable keys per phase (`train_ppo_curriculum.py --overrides`):

| Area | Keys |
| --- | --- |
| Reward shaping | `hit_reward`, `tracking_reward`, `pressure_scale`, `step_penalty` |
| Hyperparameters | `learning_rate`, `ent_coef`, `n_steps`, `batch_size`, `clip_range` |
| Observation design | `obs_features`: `paddle_velocity`, `intercept_y`, `relative_offset` |
| Curriculum | `timesteps`, `ball_speed` |

`n_steps`/`batch_size`/`clip_range` only take effect when a phase creates a new
model; a model resumed from a previous phase keeps its original values.

**Observation features change the observation shape.** Enabling one restarts
training from a fresh model — earlier checkpoints cannot be resumed. The coach
warns about this in every proposal that touches `obs_features`.

## Reading a proposal

```json
{
  "backend": "neohorse",
  "phase": 4,
  "state": "...the exact metrics text the model saw...",
  "answers": {
    "rally_farming": {"noul": 0.83},
    "reward_knob": {"choice": "hit_reward", "probabilities": {...}},
    ...
  },
  "derived_changes": [
    {"kind": "reward", "phase": 4, "key": "hit_reward",
     "old": 0.05, "new": 0.06, "confidence": 0.61}
  ],
  "overrides": {"phases": {"4": {"hit_reward": 0.06}}}
}
```

Condition-check probabilities above ~0.8 are worth taking seriously; below ~0.5
they mostly reflect priors. The coach's own model card notes that decision
probabilities are not calibrated on this task — treat proposals as a second
opinion, not ground truth.

## Caveats

- The native runtime is officially tested on Linux + CUDA; it works here on
  macOS via the `mps` device, but that combination is not upstream-tested.
- One coaching session keeps the model resident for its two decision stages
  (~9 GB in memory); the worker exits when the session ends.
- If you evaluate a model that was trained **with** coach observation features,
  pass them to the coach: `python scripts/coach.py --obs-features intercept_y
  --model models_v7/ppo_final propose ...` — otherwise evaluation fails with an
  observation-dimension error.
- `coach/overrides.json` is gitignored; proposals under `coach/proposals/` are
  kept for review.
