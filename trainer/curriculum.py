"""
Curriculum Manager for 6-Phase Progressive Mastery Curriculum (PMC).

Implements win-rate focused phase transitions with variance checking
and adaptive thresholds to train a master-level Pong AI.

Phases:
1. Ball Tracking (static opponent)
2. Basic Returns (slow AI)
3. First Wins (beginner AI) - CRITICAL PHASE
4. Competitive Play (normal AI)
5. Advanced Control (reactive AI)
6. Mastery (self-play)

Key Features:
- Pure win-rate metrics from Phase 3 onward
- Variance checking for stable advancement
- Adaptive thresholds to prevent stalling
- Sub-phase progression in Phase 3
- Stability requirements per phase
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from collections import deque
import numpy as np


@dataclass
class Phase:
    """Configuration for a curriculum phase."""
    name: str
    opponent_type: str
    ball_speed: float
    advance_metric: Optional[str]      # "alignment", "hit_rate", "win_rate"
    advance_threshold: Optional[float]
    min_episodes: int = 100
    stability_episodes: int = 20       # NEW: Episodes above threshold before advance
    variance_threshold: float = 0.12   # NEW: Max variance allowed


class CurriculumManager:
    """
    Manages 6-phase Progressive Mastery Curriculum with:
    - Win-rate focused metrics
    - Variance checking for stable advancement
    - Adaptive thresholds to prevent stalling
    - Sub-phase progression support
    """
    
    # 6-Phase PMC - Progressive Mastery Curriculum
    PHASES = [
        Phase(
            name="Ball Tracking",
            opponent_type="slow_ai",  # Changed from static - gives more varied ball trajectories
            ball_speed=0.6,
            advance_metric="alignment",
            advance_threshold=0.55,
            min_episodes=75,
            stability_episodes=15,  # Reduced from 20 for faster advancement
            variance_threshold=0.15,
        ),
        Phase(
            name="Basic Returns",
            opponent_type="slow_ai",
            ball_speed=0.75,
            advance_metric="hit_rate",
            advance_threshold=0.65,
            min_episodes=100,
            stability_episodes=30,
            variance_threshold=0.12,
        ),
        Phase(
            name="First Wins",
            opponent_type="beginner_ai",
            ball_speed=0.85,
            advance_metric="win_rate",
            advance_threshold=0.30,
            min_episodes=200,
            stability_episodes=50,
            variance_threshold=0.12,
        ),
        Phase(
            name="Competitive Play",
            opponent_type="normal_ai",
            ball_speed=0.95,
            advance_metric="win_rate",
            advance_threshold=0.40,
            min_episodes=250,
            stability_episodes=50,
            variance_threshold=0.12,
        ),
        Phase(
            name="Advanced Control",
            opponent_type="reactive_ai",
            ball_speed=1.0,
            advance_metric="win_rate",
            advance_threshold=0.48,
            min_episodes=400,
            stability_episodes=75,
            variance_threshold=0.12,
        ),
        Phase(
            name="Mastery",
            opponent_type="agent",
            ball_speed=1.0,
            advance_metric="win_rate",
            advance_threshold=0.50,
            min_episodes=500,
            stability_episodes=100,
            variance_threshold=0.10,
        ),
    ]
    
    # Minimum hit rate floor to prevent skill regression
    MIN_HIT_RATE_FLOOR = 0.70
    
    def __init__(self, start_phase: int = 0, metrics_window: int = 150):
        """
        Initialize curriculum manager.
        
        Args:
            start_phase: Starting phase index (0-5)
            metrics_window: Number of episodes to average for advancement decisions
        """
        self.current_phase = max(0, min(start_phase, len(self.PHASES) - 1))
        self.episodes_in_phase = 0
        self.total_episodes = 0
        self.metrics_window = metrics_window
        
        # Metrics history for current phase
        self._metrics_history: deque = deque(maxlen=metrics_window)
        
        # Track phase transitions
        self.phase_history: List[Dict[str, Any]] = []
        
        # Track consecutive episodes above threshold
        self._consecutive_above_threshold = 0
    
    @property
    def phase(self) -> Phase:
        """Get current phase configuration."""
        return self.PHASES[self.current_phase]
    
    @property
    def is_final_phase(self) -> bool:
        """Check if we're in the final phase."""
        return self.current_phase >= len(self.PHASES) - 1
    
    def get_phase_config(self) -> Dict[str, Any]:
        """Get configuration dict for current phase."""
        phase = self.phase
        return {
            "phase_index": self.current_phase,
            "phase_name": phase.name,
            "opponent_type": phase.opponent_type,
            "ball_speed_multiplier": phase.ball_speed,
            "advance_metric": phase.advance_metric,
            "advance_threshold": self.get_effective_threshold(),
            "episodes_in_phase": self.episodes_in_phase,
            "min_episodes": phase.min_episodes,
            "stability_required": phase.stability_episodes,
            "consecutive_above": self._consecutive_above_threshold,
        }
    
    def get_effective_threshold(self) -> float:
        """
        Get effective threshold with adaptive reduction for stuck phases.
        
        Returns:
            Effective threshold (may be reduced if stuck)
        """
        phase = self.phase
        
        if phase.advance_threshold is None:
            return 0.0
        
        base_threshold = phase.advance_threshold
        
        # Adaptive threshold reduction if stuck
        if phase.min_episodes:
            # If stuck for 2.5x minimum episodes, reduce by 10%
            if self.episodes_in_phase > phase.min_episodes * 2.5:
                return base_threshold * 0.90
            
            # If stuck for 4x minimum episodes, reduce by 20%
            if self.episodes_in_phase > phase.min_episodes * 4:
                return base_threshold * 0.80
        
        return base_threshold
    
    def record_episode(self, info: Dict[str, Any]) -> bool:
        """
        Record episode metrics and check for phase advancement.
        
        Args:
            info: Episode info dict containing metrics from environment
            
        Returns:
            True if advanced to next phase, False otherwise
        """
        self._metrics_history.append(info)
        self.episodes_in_phase += 1
        self.total_episodes += 1
        
        # Update consecutive counter
        self._update_consecutive_counter(info)
        
        # Check for phase advancement
        if self.should_advance():
            self._advance_phase()
            return True
        
        return False
    
    def _update_consecutive_counter(self, info: Dict[str, Any]):
        """Update counter of consecutive episodes above threshold."""
        phase = self.phase
        
        if phase.advance_metric is None:
            return
        
        # Get current episode metric value
        metric_value = self._get_metric_from_info(info, phase.advance_metric)
        threshold = self.get_effective_threshold()
        
        if metric_value >= threshold:
            self._consecutive_above_threshold += 1
        else:
            self._consecutive_above_threshold = 0
    
    def _get_metric_from_info(self, info: Dict[str, Any], metric: str) -> float:
        """Extract specific metric from info dict."""
        if metric == "alignment":
            return info.get("alignment", 0.0)
        elif metric == "hit_rate":
            return info.get("hit_rate", 0.0)
        elif metric == "win_rate":
            return info.get("win_rate", 0.0)
        elif metric == "rally":
            return info.get("avg_rally", 0.0)
        elif metric == "score":
            return info.get("player_score", 0.0) - info.get("opponent_score", 0.0)
        return 0.0
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate metrics from recent episodes."""
        if not self._metrics_history:
            return {
                "alignment": 0.0,
                "hit_rate": 0.0,
                "rally": 0.0,
                "score": 0.0,
                "win_rate": 0.0,
                "variance": 1.0,
            }
        
        metrics = list(self._metrics_history)
        
        # Calculate primary metrics
        alignment = np.mean([m.get("alignment", 0.0) for m in metrics])
        hit_rate = np.mean([m.get("hit_rate", 0.0) for m in metrics])
        rally = np.mean([m.get("avg_rally", 0.0) for m in metrics])
        score = np.mean([m.get("player_score", 0.0) - m.get("opponent_score", 0.0) for m in metrics])
        win_rate = np.mean([m.get("win_rate", 0.0) for m in metrics])
        
        # Calculate variance of current advance metric
        phase = self.phase
        if phase.advance_metric:
            metric_values = [self._get_metric_from_info(m, phase.advance_metric) for m in metrics]
            variance = np.std(metric_values) if len(metric_values) > 1 else 0.0
        else:
            variance = 0.0
        
        return {
            "alignment": alignment,
            "hit_rate": hit_rate,
            "rally": rally,
            "score": score,
            "win_rate": win_rate,
            "variance": variance,
        }
    
    def should_advance(self) -> bool:
        """
        Check if agent should advance to next phase.
        
        Criteria:
        1. Not final phase
        2. Minimum episodes met
        3. Metric threshold met
        4. Variance check passed (stability)
        5. Stability requirement met (consecutive episodes)
        6. Minimum hit rate floor (Phase 3+)
        """
        phase = self.phase
        
        # Can't advance from final phase
        if self.is_final_phase:
            return False
        
        # Check minimum episodes
        if phase.min_episodes and self.episodes_in_phase < phase.min_episodes:
            return False
        
        # No metric required (shouldn't happen)
        if phase.advance_metric is None:
            return False
        
        # Get aggregated metrics
        metrics = self.aggregate_metrics()
        current_value = metrics.get(phase.advance_metric, 0.0)
        threshold = self.get_effective_threshold()
        
        # Check metric threshold
        if current_value < threshold:
            return False
        
        # Check variance (stability requirement)
        variance = metrics.get("variance", 1.0)
        if variance > phase.variance_threshold:
            return False
        
        # Check consecutive episodes above threshold
        if self._consecutive_above_threshold < phase.stability_episodes:
            return False
        
        # Check minimum hit rate floor (Phase 3+)
        if self.current_phase >= 2:  # Phase 3 onward
            hit_rate = metrics.get("hit_rate", 0.0)
            if hit_rate < self.MIN_HIT_RATE_FLOOR:
                return False
        
        return True
    
    def _advance_phase(self):
        """Advance to next phase."""
        # Record transition
        self.phase_history.append({
            "from_phase": self.current_phase,
            "to_phase": self.current_phase + 1,
            "episodes_in_phase": self.episodes_in_phase,
            "total_episodes": self.total_episodes,
            "final_metrics": self.aggregate_metrics(),
        })
        
        # Move to next phase
        self.current_phase += 1
        self.episodes_in_phase = 0
        self._consecutive_above_threshold = 0
        self._metrics_history.clear()
    
    def force_advance(self):
        """Force advancement to next phase (for testing/debugging)."""
        if not self.is_final_phase:
            self._advance_phase()
    
    def get_status_string(self) -> str:
        """Get a formatted status string for logging."""
        phase = self.phase
        metrics = self.aggregate_metrics()
        
        status = f"Phase {self.current_phase + 1}/6: {phase.name}"
        status += f" | Episodes: {self.episodes_in_phase}"
        
        if phase.advance_metric and phase.advance_threshold:
            current = metrics.get(phase.advance_metric, 0.0)
            effective_threshold = self.get_effective_threshold()
            status += f" | {phase.advance_metric}: {current:.2f}/{effective_threshold:.2f}"
        
        if phase.min_episodes:
            status += f" | Min: {phase.min_episodes}"
        
        # Show stability progress
        status += f" | Stable: {self._consecutive_above_threshold}/{phase.stability_episodes}"
        
        # Show variance
        variance = metrics.get("variance", 0.0)
        status += f" | Var: {variance:.3f}"
        
        return status
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "current_phase": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "total_episodes": self.total_episodes,
            "phase_history": self.phase_history,
            "metrics_history": list(self._metrics_history),
            "consecutive_above_threshold": self._consecutive_above_threshold,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.current_phase = state.get("current_phase", 0)
        self.episodes_in_phase = state.get("episodes_in_phase", 0)
        self.total_episodes = state.get("total_episodes", 0)
        self.phase_history = state.get("phase_history", [])
        self._consecutive_above_threshold = state.get("consecutive_above_threshold", 0)
        
        # Restore metrics history
        metrics = state.get("metrics_history", [])
        self._metrics_history = deque(metrics, maxlen=self.metrics_window)


def create_env_for_phase(
    phase_config: Dict[str, Any],
    agent=None,
    render_mode: Optional[str] = None,
) -> "PongHeadlessEnv":
    """
    Create environment configured for a specific curriculum phase.
    
    Args:
        phase_config: Phase configuration from CurriculumManager
        agent: Agent for self-play phases (optional)
        render_mode: Rendering mode
        
    Returns:
        Configured PongHeadlessEnv
    """
    from pong.env.pong_headless import PongHeadlessEnv, OpponentType
    
    opponent_type = phase_config["opponent_type"]
    ball_speed_mult = phase_config["ball_speed_multiplier"]
    phase_index = phase_config["phase_index"]
    
    # Map opponent type string to OpponentType
    opponent_mapping = {
        "static": OpponentType.STATIC,
        "slow_ai": OpponentType.SLOW_AI,
        "beginner_ai": OpponentType.BEGINNER_AI,
        "normal_ai": OpponentType.NORMAL_AI,
        "reactive_ai": OpponentType.REACTIVE_AI,
        "agent": OpponentType.AGENT,
    }
    
    env = PongHeadlessEnv(
        render_mode=render_mode,
        ball_speed_multiplier=ball_speed_mult,
        opponent_type=opponent_mapping.get(opponent_type, OpponentType.NORMAL_AI),
        agent_controlled_opponent=(opponent_type == "agent"),
        curriculum_phase=phase_index,
    )
    
    return env


# Convenience function for quick testing
if __name__ == "__main__":
    print("Testing 6-Phase Progressive Mastery Curriculum...")
    
    cm = CurriculumManager()
    
    print(f"\n📚 6-Phase PMC Configuration:")
    for i, phase in enumerate(cm.PHASES):
        print(f"\n  Phase {i+1}: {phase.name}")
        print(f"    Opponent: {phase.opponent_type}, Ball: {phase.ball_speed}x")
        if phase.advance_metric:
            print(f"    Advance: {phase.advance_metric} >= {phase.advance_threshold}")
            print(f"    Min episodes: {phase.min_episodes}")
            print(f"    Stability: {phase.stability_episodes} consecutive episodes")
            print(f"    Max variance: {phase.variance_threshold}")
    
    print(f"\nCurrent phase: {cm.get_status_string()}")
    print(f"Config: {cm.get_phase_config()}")
    
    # Simulate some episodes with good performance
    print("\n🎮 Simulating episodes with good performance...")
    for i in range(200):
        fake_info = {
            "player_score": 5,
            "opponent_score": 2,
            "alignment": 0.75,
            "hit_rate": 0.85,
            "avg_rally": 3.0,
            "win_rate": 0.65,
        }
        advanced = cm.record_episode(fake_info)
        if advanced:
            print(f"  Episode {i+1}: 🏆 Advanced to Phase {cm.current_phase + 1}!")
        elif (i + 1) % 50 == 0:
            print(f"  Episode {i+1}: {cm.get_status_string()}")
    
    print(f"\n✓ CurriculumManager test completed!")
    print(f"  Final phase: {cm.current_phase + 1}")
    print(f"  Phase history: {len(cm.phase_history)} transitions")
