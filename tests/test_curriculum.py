#!/usr/bin/env python3
"""
Tests for curriculum manager fixes.

These tests verify the bug fixes made to prevent training from stalling:
1. Adaptive threshold reduction ordering (was using if/if instead of if/elif)
2. Rolling average for consecutive counter (was using per-episode value)
3. Soft reset for noisy metrics (was hard resetting to 0)
"""
import numpy as np
import pytest
from trainer.curriculum import CurriculumManager, Phase


class TestAdaptiveThresholdReduction:
    """Test that adaptive threshold reduction works correctly."""
    
    def test_threshold_reduces_at_3x_episodes(self):
        """Threshold should reduce by 10% when stuck for 3x min_episodes."""
        cm = CurriculumManager()
        # Phase 1 has min_episodes=75, threshold=0.55
        cm.episodes_in_phase = 226  # > 75 * 3 = 225
        
        threshold = cm.get_effective_threshold()
        expected = 0.55 * 0.90  # 10% reduction
        
        assert abs(threshold - expected) < 0.001, f"Expected {expected}, got {threshold}"
    
    def test_threshold_reduces_at_5x_episodes(self):
        """Threshold should reduce by 15% when stuck for 5x min_episodes."""
        cm = CurriculumManager()
        # Phase 1 has min_episodes=75
        cm.episodes_in_phase = 376  # > 75 * 5 = 375
        
        threshold = cm.get_effective_threshold()
        expected = 0.55 * 0.85  # 15% reduction
        
        assert abs(threshold - expected) < 0.001, f"Expected {expected}, got {threshold}"
    
    def test_threshold_reduces_at_8x_episodes(self):
        """Threshold should reduce by 20% when stuck for 8x min_episodes."""
        cm = CurriculumManager()
        # Phase 1 has min_episodes=75
        cm.episodes_in_phase = 601  # > 75 * 8 = 600
        
        threshold = cm.get_effective_threshold()
        expected = 0.55 * 0.80  # 20% reduction
        
        assert abs(threshold - expected) < 0.001, f"Expected {expected}, got {threshold}"
    
    def test_threshold_ordering_is_correct(self):
        """Larger reductions should take precedence over smaller ones."""
        cm = CurriculumManager()
        
        # At 8x episodes, should get 20% reduction
        cm.episodes_in_phase = 650
        threshold_8x = cm.get_effective_threshold()
        
        cm.episodes_in_phase = 400
        threshold_5x = cm.get_effective_threshold()
        
        cm.episodes_in_phase = 250
        threshold_3x = cm.get_effective_threshold()
        
        # 8x should be lower than 5x, which should be lower than 3x
        assert threshold_8x < threshold_5x < threshold_3x, \
            f"Threshold ordering wrong: 8x={threshold_8x}, 5x={threshold_5x}, 3x={threshold_3x}"


class TestRollingAverageConsecutiveCounter:
    """Test that consecutive counter uses rolling average instead of per-episode values."""
    
    def test_noisy_metrics_dont_reset_counter(self):
        """Noisy per-episode metrics shouldn't constantly reset the counter."""
        cm = CurriculumManager()
        cm._consecutive_above_threshold = 0
        
        np.random.seed(42)  # For reproducibility
        
        # Simulate 50 episodes with noisy hit_rate (0.4-0.8, avg ~0.6)
        for i in range(50):
            fake_info = {
                'hit_rate': np.random.uniform(0.4, 0.8),
                'alignment': 0.6,
                'win_rate': 0.5,
            }
            cm.record_episode(fake_info)
        
        # With old per-episode logic, counter would be ~0 due to constant resets
        # With rolling average, counter should be positive
        assert cm._consecutive_above_threshold > 10, \
            f"Counter should be >10 with rolling avg, got {cm._consecutive_above_threshold}"
    
    def test_soft_reset_instead_of_hard_reset(self):
        """Counter should decay by 1, not reset to 0."""
        cm = CurriculumManager()
        
        # Build up history first
        for i in range(25):
            cm.record_episode({
                'hit_rate': 0.7,
                'alignment': 0.6,
                'win_rate': 0.5,
            })
        
        counter_before = cm._consecutive_above_threshold
        
        # Add one "bad" episode (rolling avg still above threshold)
        cm.record_episode({
            'hit_rate': 0.3,
            'alignment': 0.4,
            'win_rate': 0.3,
        })
        
        counter_after = cm._consecutive_above_threshold
        
        # Counter should have decayed by 1 at most, not reset to 0
        # (unless rolling avg dropped below threshold)
        assert counter_after >= counter_before - 1 or counter_after > 0, \
            f"Counter dropped too much: {counter_before} -> {counter_after}"


class TestPhaseConfiguration:
    """Test that phase configurations are reasonable."""
    
    def test_phase2_variance_threshold_increased(self):
        """Phase 2 variance threshold should be >= 0.30 for noisy hit_rate."""
        cm = CurriculumManager()
        phase2 = cm.PHASES[1]  # Basic Returns
        
        assert phase2.variance_threshold >= 0.30, \
            f"Phase 2 variance threshold too strict: {phase2.variance_threshold}"
    
    def test_phase2_advance_threshold_achievable(self):
        """Phase 2 advance threshold should be <= 0.65."""
        cm = CurriculumManager()
        phase2 = cm.PHASES[1]
        
        assert phase2.advance_threshold <= 0.65, \
            f"Phase 2 threshold too high: {phase2.advance_threshold}"
    
    def test_hit_rate_floor_not_too_strict(self):
        """MIN_HIT_RATE_FLOOR should be achievable (< 0.70)."""
        assert CurriculumManager.MIN_HIT_RATE_FLOOR <= 0.60, \
            f"Hit rate floor too strict: {CurriculumManager.MIN_HIT_RATE_FLOOR}"


class TestPhase2HitRateFloor:
    """Test that Phase 2 requires minimum hit rate to advance."""
    
    def test_low_hit_rate_blocks_phase2_advancement(self):
        """Agent with <55% hit rate should NOT advance from Phase 2."""
        cm = CurriculumManager()
        
        # Advance to Phase 2 first
        for i in range(100):
            cm.record_episode({
                'hit_rate': 0.75,
                'alignment': 0.65,
                'win_rate': 0.6,
            })
        
        assert cm.current_phase == 1, "Should be in Phase 2"
        
        # Now try to advance with low hit_rate (45%) - should NOT advance
        for i in range(300):
            cm.record_episode({
                'hit_rate': 0.45,  # Below 55% floor
                'alignment': 0.80,
                'win_rate': 0.60,
            })
        
        # Should still be in Phase 2 despite good other metrics
        assert cm.current_phase == 1, \
            f"Should NOT advance with 45% hit_rate, but at Phase {cm.current_phase + 1}"
    
    def test_good_hit_rate_allows_phase2_advancement(self):
        """Agent with >=55% hit rate CAN advance from Phase 2."""
        cm = CurriculumManager()
        
        # Advance to Phase 2 first
        for i in range(100):
            cm.record_episode({
                'hit_rate': 0.75,
                'alignment': 0.65,
                'win_rate': 0.6,
            })
        
        assert cm.current_phase == 1, "Should be in Phase 2"
        
        # Now with good hit_rate (60%) - should advance
        for i in range(200):
            cm.record_episode({
                'hit_rate': 0.60,  # Above 55% floor
                'alignment': 0.75,
                'win_rate': 0.50,
            })
        
        # Should have advanced to Phase 3
        assert cm.current_phase >= 2, \
            f"Should advance with 60% hit_rate, but at Phase {cm.current_phase + 1}"


class TestPhaseAdvancement:
    """Test that phase advancement works correctly."""
    
    def test_advancement_with_good_metrics(self):
        """Agent should advance with consistently good metrics."""
        cm = CurriculumManager()
        
        # Simulate good performance for Phase 1
        advanced_count = 0
        for i in range(200):
            result = cm.record_episode({
                'hit_rate': 0.75,
                'alignment': 0.65,
                'win_rate': 0.6,
                'avg_rally': 3.0,
            })
            if result:
                advanced_count += 1
        
        # Should have advanced at least once (from Phase 1 to Phase 2)
        assert advanced_count >= 1, \
            f"Should have advanced with good metrics, advanced {advanced_count} times"
        assert cm.current_phase >= 1, \
            f"Should be at least Phase 2, but at Phase {cm.current_phase + 1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

