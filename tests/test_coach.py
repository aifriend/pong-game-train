"""
Tests for the RL coach: proposal generation, override plumbing, observation
features, and the apply round-trip. All tests use the mock backend and never
load the NeoHorse model.
"""

import json
import pytest

from pathlib import Path

from pong.coach.apply import apply_proposal
from pong.coach.engine import MockBackend
from pong.coach.propose import _adjust_value, derive_changes, run_proposal
from pong.env.pong_headless import PongHeadlessEnv, SUPPORTED_OBS_FEATURES
from scripts.train_ppo_curriculum import apply_overrides

# --- helpers ---------------------------------------------------------------


def make_metrics(phase=2):
    return {
        "latest_phase": phase,
        "config": {
            2: {
                "hit_reward": 0.3,
                "tracking_reward": 0.02,
                "pressure_scale": 1.0,
                "step_penalty": 0.001,
                "learning_rate": 2.5e-4,
                "ent_coef": 0.04,
                "obs_features": None,
            }
        },
    }


def uniform_probs(keys, pick):
    return {k: 1.0 / len(keys) for k in keys}


def make_answers(
    reward_knob="none",
    reward_adjustment="keep",
    hp_knob="none",
    hp_adjustment="keep",
    obs_feature="none",
):
    return {
        "plateaued": {"noul": 0.4},
        "rally_farming": {"noul": 0.2},
        "value_diverging": {"noul": 0.1},
        "entropy_collapsed": {"noul": 0.3},
        "reward_knob": {
            "choice": reward_knob,
            "probabilities": uniform_probs(
                [
                    "hit_reward",
                    "tracking_reward",
                    "pressure_scale",
                    "step_penalty",
                    "none",
                ],
                reward_knob,
            ),
        },
        "reward_adjustment": {
            "choice": reward_adjustment,
            "probabilities": uniform_probs(
                ["increase_10", "increase_25", "decrease_10", "decrease_25", "keep"],
                reward_adjustment,
            ),
        },
        "hp_knob": {
            "choice": hp_knob,
            "probabilities": uniform_probs(
                [
                    "learning_rate",
                    "ent_coef",
                    "n_steps",
                    "batch_size",
                    "clip_range",
                    "none",
                ],
                hp_knob,
            ),
        },
        "hp_adjustment": {
            "choice": hp_adjustment,
            "probabilities": uniform_probs(
                ["increase_10", "increase_25", "decrease_10", "decrease_25", "keep"],
                hp_adjustment,
            ),
        },
        "obs_feature": {
            "choice": obs_feature,
            "probabilities": uniform_probs(
                ["paddle_velocity", "intercept_y", "relative_offset", "none"],
                obs_feature,
            ),
        },
        "phase_health": {"score": 1.5, "probabilities": [0.25, 0.25, 0.25, 0.25]},
    }


# --- adjustment math -------------------------------------------------------


def test_adjust_value_bounds():
    # A 25% decrease below the floor clamps to the floor instead of rounding
    # back to the old value
    result = _adjust_value("hit_reward", 0.05, "decrease_25")
    assert result == pytest.approx(0.038)  # 0.0375 rounded to two significant digits

    # keep is a no-op
    assert _adjust_value("hit_reward", 0.5, "keep") is None

    # zero stays zero: no change to report
    assert _adjust_value("hit_reward", 0.0, "increase_10") is None


def test_adjust_value_int_knob_clamped():
    # Already at the documented upper bound: an increase clamps back to the
    # current value and is reported as no change
    assert _adjust_value("n_steps", 4096, "increase_25") is None
    # Below the bound, the increase applies
    assert _adjust_value("n_steps", 3000, "increase_25") == 3750
    # Float knobs are clamped too
    assert _adjust_value("clip_range", 0.39, "increase_25") == pytest.approx(0.4)


def test_stage2_questions_only_for_chosen_knobs():
    from pong.coach.questions import build_stage2_questions

    stage1 = {
        "reward_knob": {"choice": "none"},
        "hp_knob": {"choice": "learning_rate"},
    }
    questions = build_stage2_questions(stage1)
    assert "reward_adjustment" not in questions
    assert "learning_rate" in questions["hp_adjustment"]["instructions"]

    both_none = {
        "reward_knob": {"choice": "none"},
        "hp_knob": {"choice": "none"},
    }
    assert build_stage2_questions(both_none) == {}

    # integer knobs stay integers
    new = _adjust_value("n_steps", 1024, "increase_25")
    assert isinstance(new, int) and new == 1280


def test_derive_changes_bounded():
    answers = make_answers(
        reward_knob="pressure_scale",
        reward_adjustment="increase_10",
        hp_knob="learning_rate",
        hp_adjustment="decrease_25",
    )
    derived = derive_changes(answers, 2, make_metrics()["config"][2])

    assert derived["overrides"]["pressure_scale"] == pytest.approx(1.1)
    # 2.5e-4 * 0.75 = 1.9e-4 after two-significant-digit rounding
    assert derived["overrides"]["learning_rate"] == pytest.approx(1.9e-4)
    kinds = {c["kind"] for c in derived["changes"]}
    assert kinds == {"reward", "hyperparameter"}


def test_derive_changes_obs_feature_warning():
    answers = make_answers(obs_feature="intercept_y")
    derived = derive_changes(answers, 2, make_metrics()["config"][2])

    assert derived["overrides"]["obs_features"] == ["intercept_y"]
    assert any("fresh model" in w for w in derived["warnings"])
    change = [c for c in derived["changes"] if c["kind"] == "observation"]
    assert len(change) == 1


def test_derive_changes_keep_means_no_overrides():
    derived = derive_changes(make_answers(), 2, make_metrics()["config"][2])
    assert derived["overrides"] == {}
    assert derived["changes"] == []


# --- proposal generation ---------------------------------------------------


def test_mock_proposal_schema(tmp_path):
    proposal = run_proposal(
        backend_name="mock",
        metrics=make_metrics(),
        phase=2,
        out_dir=tmp_path,
    )

    for section in (
        "state",
        "metrics_snapshot",
        "questions",
        "answers",
        "derived_changes",
        "overrides",
    ):
        assert section in proposal
    assert proposal["overrides"] == {"phases": {"2": {}}}
    # The proposal file was written where the caller asked
    assert json.loads(Path(proposal["path"]).read_text())["phase"] == 2


def test_full_change_loop_forced_backend(tmp_path, monkeypatch):
    """A backend that actually picks a change must produce an override that the
    trainer accepts via coach/overrides.json."""

    class ForceChangeBackend(MockBackend):
        name = "force"

        def predict(self, state, questions):
            answers = super().predict(state, questions)
            if "reward_knob" in answers:
                answers["reward_knob"]["choice"] = "pressure_scale"
            if "hp_knob" in answers:
                answers["hp_knob"]["choice"] = "none"
            if "reward_adjustment" in answers:
                answers["reward_adjustment"]["choice"] = "increase_25"
            return answers

    monkeypatch.setattr(
        "pong.coach.propose.get_backend",
        lambda name, model_dir=None: ForceChangeBackend(),
    )

    proposal = run_proposal(
        backend_name="mock",
        metrics=make_metrics(),
        phase=2,
        out_dir=tmp_path,
    )
    overrides = proposal["overrides"]["phases"]["2"]
    assert "pressure_scale" in overrides
    assert overrides["pressure_scale"] != 1.0

    # proposal -> overrides file -> trainer merge
    overrides_path = tmp_path / "overrides.json"
    apply_proposal(Path(proposal["path"]), overrides_path)
    stored = json.loads(overrides_path.read_text())
    assert stored["phases"]["2"]["pressure_scale"] == overrides["pressure_scale"]

    from scripts.train_ppo_curriculum import CURRICULUM_PHASES

    original = CURRICULUM_PHASES[2]["pressure_scale"]
    try:
        changes = apply_overrides(stored)
        assert changes and "pressure_scale" in changes[0]
        assert CURRICULUM_PHASES[2]["pressure_scale"] == overrides["pressure_scale"]
    finally:
        CURRICULUM_PHASES[2]["pressure_scale"] = original


def test_state_text_handles_missing_metrics():
    from pong.coach.metrics import build_state_text

    # No tensorboard phases and no evaluation: only the config section
    text = build_state_text({"config": {1: {"hit_reward": 0.5}}}, 1)
    assert "hit_reward = 0.5" in text
    assert "Recent training metrics" not in text
    assert "Fresh evaluation" not in text


# --- overrides plumbing in the trainer -------------------------------------


def test_apply_overrides_merges_and_rejects():
    from scripts.train_ppo_curriculum import CURRICULUM_PHASES

    original = CURRICULUM_PHASES[3]["hit_reward"]
    try:
        changes = apply_overrides({"phases": {"3": {"hit_reward": 0.12}}})
        assert changes == ["phase 3: hit_reward 0.1 -> 0.12"]

        with pytest.raises(ValueError, match="Unknown override keys"):
            apply_overrides({"phases": {"3": {"not_a_key": 1}}})

        with pytest.raises(ValueError, match="unknown phase"):
            apply_overrides({"phases": {"9": {"hit_reward": 1}}})

        with pytest.raises(ValueError, match="Invalid phase key"):
            apply_overrides({"phases": {"banana": {"hit_reward": 1}}})
    finally:
        CURRICULUM_PHASES[3]["hit_reward"] = original


# --- observation features --------------------------------------------------


@pytest.mark.parametrize("feature", SUPPORTED_OBS_FEATURES)
def test_obs_features_step_and_shape(feature):
    env = PongHeadlessEnv(obs_features=[feature])
    obs, _ = env.reset(seed=42)
    assert obs.shape == (10,)
    assert env.observation_space.shape == (10,)

    for _ in range(100):
        obs, _, terminated, truncated, _ = env.step(2)
        if terminated or truncated:
            obs, _ = env.reset()
        assert obs.shape == (10,)
        assert obs.min() >= -1.0 and obs.max() <= 1.0


def test_obs_features_all_and_unknown():
    env = PongHeadlessEnv(obs_features=list(SUPPORTED_OBS_FEATURES))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (9 + len(SUPPORTED_OBS_FEATURES),)

    with pytest.raises(ValueError, match="Unknown obs_features"):
        PongHeadlessEnv(obs_features=["teleport_ball"])

    # Default stays at the original 9 dimensions (backward compatible)
    env = PongHeadlessEnv()
    obs, _ = env.reset()
    assert obs.shape == (9,)


# --- apply round-trip ------------------------------------------------------


def test_config_snapshot_includes_ppo_defaults():
    from pong.coach.metrics import current_config_snapshot

    snapshot = current_config_snapshot()
    # These live as trainer defaults, not in CURRICULUM_PHASES, but the coach
    # must see them to be able to adjust them
    assert snapshot[1]["n_steps"] == 1024
    assert snapshot[1]["batch_size"] == 128
    assert snapshot[1]["clip_range"] == pytest.approx(0.2)


def test_collect_evaluation_rejects_obs_dim_mismatch():
    from pong.coach.metrics import collect_evaluation

    project_root = Path(__file__).parent.parent
    model = project_root / "models" / "ppo_final.zip"
    if not model.exists():
        pytest.skip("models/ppo_final.zip not present")

    # The shipped model was trained on 9 dims; evaluating it as if it had an
    # extra feature must fail loudly, not with a cryptic shape error
    with pytest.raises(ValueError, match="obs_features"):
        collect_evaluation(model, n_episodes=1, obs_features=["intercept_y"])


def test_apply_round_trip(tmp_path):
    proposal = {
        "created_at": "now",
        "overrides": {
            "phases": {"2": {"hit_reward": 0.4, "obs_features": ["intercept_y"]}}
        },
    }
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal))
    overrides_path = tmp_path / "overrides.json"

    merged = apply_proposal(proposal_path, overrides_path)
    assert merged["phases"]["2"]["hit_reward"] == 0.4

    # A second proposal on the same key overwrites; untouched keys survive
    proposal["overrides"] = {"phases": {"2": {"hit_reward": 0.5}}}
    proposal_path.write_text(json.dumps(proposal))
    merged = apply_proposal(proposal_path, overrides_path)
    assert merged["phases"]["2"]["hit_reward"] == 0.5
    assert merged["phases"]["2"]["obs_features"] == ["intercept_y"]


def test_apply_rejects_unhonored_keys(tmp_path):
    from pong.coach.apply import validate_overrides

    with pytest.raises(ValueError, match="cannot honor"):
        validate_overrides({"phases": {"2": {"gamma": 0.9}}})
