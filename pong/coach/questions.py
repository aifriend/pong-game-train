"""
Decision menu for the RL coach.

NeoHorse-Jev-4B is prefill-only: every question declares its candidate answers
up front. The three decision types map to coach jobs as follows:

- Noul  (yes/no probability)  -> condition checks ("is the agent rally-farming?")
- Choice (candidate + probs)  -> which knob to touch and how
- Score (ordered levels)      -> how healthy a training phase looks
"""

# Bounded adjustments the coach may apply to a numeric knob. The multipliers
# keep every proposal inside +/-25% per coaching round, so a wrong decision can
# only nudge training, never derail it.
ADJUSTMENT_MULTIPLIERS = {
    "increase_10": 1.10,
    "increase_25": 1.25,
    "decrease_10": 0.90,
    "decrease_25": 0.75,
    "keep": None,
}

# Sanity bounds applied after the multiplier, per knob.
KNOB_BOUNDS = {
    "hit_reward": (0.0, 1.0),
    "tracking_reward": (0.0, 0.1),
    "pressure_scale": (0.0, 2.0),
    "step_penalty": (0.0, 0.01),
    "learning_rate": (1e-6, 1e-3),
    "ent_coef": (0.0, 0.2),
    "n_steps": (256, 4096),
    "batch_size": (32, 1024),
    "clip_range": (0.05, 0.4),
}

REWARD_KNOBS = {
    "hit_reward": "Reward paid once per successful paddle hit; bootstraps motor skills but can teach rally-farming",
    "tracking_reward": "Small per-step reward for aligning the paddle with an approaching ball",
    "pressure_scale": "Reward per return proportional to how far the opponent is from the predicted ball arrival",
    "step_penalty": "Per-step penalty encouraging shorter rallies and faster games",
    "none": "Reward shaping is fine; do not change it",
}

HP_KNOBS = {
    "learning_rate": "PPO learning rate; lower when forgetting or instability appears, higher when learning stalls",
    "ent_coef": "Entropy coefficient; higher encourages exploration, lower sharpens the policy",
    "n_steps": "Rollout length per environment before each PPO update",
    "batch_size": "Minibatch size for PPO updates",
    "clip_range": "PPO clipping range; lower makes updates more conservative",
    "none": "Hyperparameters are fine; do not change them",
}

OBS_FEATURES = {
    "paddle_velocity": "Player paddle y-velocity; gives the policy its own movement intent",
    "intercept_y": "Predicted ball y at the player paddle plane with wall bounces; a head start on positioning",
    "relative_offset": "Signed vertical distance between paddle and ball; direct tracking error",
    "none": "The current observation vector is sufficient",
}

PHASE_HEALTH_LEVELS = ["failing", "struggling", "progressing", "on_track"]


ADJUSTMENT_CRITERIA = {
    "increase_10": "Increase by 10%",
    "increase_25": "Increase by 25%",
    "decrease_10": "Decrease by 10%",
    "decrease_25": "Decrease by 25%",
    "keep": "Leave unchanged",
}


def build_stage1_questions(phase: int) -> dict:
    """
    Stage 1: condition checks, knob selection, observation feature choice, and
    phase health. The knob picks feed the stage-2 adjustment questions.
    """
    return {
        "plateaued": {
            "type": "noul",
            "instructions": (
                "Has the win rate plateaued: no meaningful improvement across the "
                "recent training chunks, without having reached the phase target?"
            ),
        },
        "rally_farming": {
            "type": "noul",
            "instructions": (
                "Is the agent rally-farming: a high hit rate and long rallies but a "
                "low win rate, exploiting the hit reward instead of trying to score?"
            ),
        },
        "value_diverging": {
            "type": "noul",
            "instructions": (
                "Is the value function diverging or the policy destabilizing: value "
                "loss or KL growing persistently across recent updates?"
            ),
        },
        "entropy_collapsed": {
            "type": "noul",
            "instructions": (
                "Has policy entropy collapsed too early, leaving too little "
                "exploration for the agent to discover scoring strategies?"
            ),
        },
        "reward_knob": {
            "type": "choice",
            "instructions": (
                "Which reward shaping parameter most needs adjustment to improve "
                "the win rate in this phase?"
            ),
            "criteria": dict(REWARD_KNOBS),
        },
        "hp_knob": {
            "type": "choice",
            "instructions": (
                "Which PPO hyperparameter most needs adjustment given the training "
                "metrics (losses, entropy, KL, win rate trend)?"
            ),
            "criteria": dict(HP_KNOBS),
        },
        "obs_feature": {
            "type": "choice",
            "instructions": (
                "Would adding one of these observation features help the agent "
                "learn faster in this phase? Enabling a feature restarts training "
                "from a fresh model, so only choose one if the current observation "
                "is clearly missing information the agent needs."
            ),
            "criteria": dict(OBS_FEATURES),
        },
        "phase_health": {
            "type": "score",
            "instructions": (
                f"Rate the overall health of training in phase {phase}: is it "
                "failing, struggling, progressing, or on track?"
            ),
            "criteria": list(PHASE_HEALTH_LEVELS),
        },
    }


def build_stage2_questions(stage1_answers: dict) -> dict:
    """
    Stage 2: adjustment magnitude for each knob chosen in stage 1. Asked
    separately so the magnitude decision knows which knob it applies to.
    Knobs the model answered "none" for are skipped entirely.
    """
    questions = {}
    reward_knob = stage1_answers.get("reward_knob", {}).get("choice", "none")
    if reward_knob != "none":
        questions["reward_adjustment"] = {
            "type": "choice",
            "instructions": (
                f"The reward parameter '{reward_knob}' ({REWARD_KNOBS[reward_knob]}) "
                "was chosen for adjustment. In which direction and by how much? "
                "Prefer small changes unless the metric evidence is severe."
            ),
            "criteria": dict(ADJUSTMENT_CRITERIA),
        }
    hp_knob = stage1_answers.get("hp_knob", {}).get("choice", "none")
    if hp_knob != "none":
        questions["hp_adjustment"] = {
            "type": "choice",
            "instructions": (
                f"The hyperparameter '{hp_knob}' ({HP_KNOBS[hp_knob]}) was chosen "
                "for adjustment. In which direction and by how much? Prefer small "
                "changes unless the metric evidence is severe."
            ),
            "criteria": dict(ADJUSTMENT_CRITERIA),
        }
    return questions
