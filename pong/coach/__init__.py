"""
RL coach: uses a structured decision model (NeoHorse-Jev-4B) to propose bounded
changes to reward shaping, observation design, and PPO hyperparameters based on
training metrics.

The decision model is prefill-only: it cannot write code or free text, it picks
from candidate answers defined in questions.py. The coach therefore presents a
curated menu of knobs and adjustments, and translates the model's choices into a
reviewable proposal JSON. Nothing reaches training without a human applying the
proposal (scripts/coach.py apply), which writes coach/overrides.json consumed by
train_ppo_curriculum.py --overrides.
"""
