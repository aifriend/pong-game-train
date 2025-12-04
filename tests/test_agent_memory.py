"""
Unit tests for agent memory (replay buffer) transition alignment.

These tests verify that sampled (state, action, reward, next_state) transitions
correctly match the intended environment flow for both Memory and PrioritizedMemory.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trainer.agent_memory import Memory, PrioritizedMemory


class TestMemoryTransitionAlignment:
    """Test that Memory.sample_batch returns correctly aligned transitions."""

    def test_basic_transition_alignment(self):
        """
        Test that sampled transitions have correct (state, action, reward, next_state) alignment.
        
        Simulates the environment flow:
        1. initialize_new_game: add_experience(s0, 0.0, 0, False)  # initial dummy
        2. take_step: add_experience(s1, r0, a0, False)  # transition s0 -> s1
        3. take_step: add_experience(s2, r1, a1, False)  # transition s1 -> s2
        
        Expected transitions:
        - (s0, a0, r0, s1)
        - (s1, a1, r1, s2)
        """
        np.random.seed(42)
        
        # Create memory with small buffer
        mem = Memory(max_len=100, observation_dim=4)
        
        # Create distinct states for easy identification
        s0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        s2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        
        # Simulate environment flow (matching environment.py behavior)
        # initialize_new_game: adds initial state with dummy action/reward
        mem.add_experience(s0, 0.0, 0, False)
        
        # take_step 1: action a0=10 taken from s0, received r0=1.0, arrived at s1
        a0, r0 = 10, 1.0
        mem.add_experience(s1, r0, a0, False)
        
        # take_step 2: action a1=20 taken from s1, received r1=2.0, arrived at s2
        a1, r1 = 20, 2.0
        mem.add_experience(s2, r1, a1, False)
        
        # Sample all valid transitions (should be 2)
        states, actions, rewards, next_states, dones = mem.sample_batch(batch_size=2)
        
        # Verify each sampled transition is valid
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            
            # Check transition (s0, a0, r0, s1)
            if np.allclose(state, s0):
                assert action == a0, f"Expected action {a0} for s0, got {action}"
                assert reward == r0, f"Expected reward {r0} for s0->s1, got {reward}"
                assert np.allclose(next_state, s1), f"Expected next_state s1, got {next_state}"
            # Check transition (s1, a1, r1, s2)
            elif np.allclose(state, s1):
                assert action == a1, f"Expected action {a1} for s1, got {action}"
                assert reward == r1, f"Expected reward {r1} for s1->s2, got {reward}"
                assert np.allclose(next_state, s2), f"Expected next_state s2, got {next_state}"
            else:
                pytest.fail(f"Unexpected state in sampled batch: {state}")

    def test_done_flag_boundary(self):
        """
        Test that transitions ending at terminal states are correctly sampled,
        and that terminal states are not used as starting states.
        
        Flow:
        1. add_experience(s0, 0.0, 0, False)  # initial
        2. add_experience(s1, r0, a0, True)   # terminal transition s0 -> s1
        3. add_experience(s2, 0.0, 0, False)  # new episode start
        4. add_experience(s3, r2, a2, False)  # transition s2 -> s3
        
        Valid transitions:
        - (s0, a0, r0, s1, done=True)
        - (s2, a2, r2, s3, done=False)
        
        Invalid (should not be sampled):
        - Any transition starting from s1 (terminal state)
        """
        np.random.seed(42)
        
        mem = Memory(max_len=100, observation_dim=4)
        
        s0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # terminal
        s2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)  # new episode
        s3 = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
        
        # Episode 1
        mem.add_experience(s0, 0.0, 0, False)
        mem.add_experience(s1, 1.0, 10, True)  # Terminal!
        
        # Episode 2
        mem.add_experience(s2, 0.0, 0, False)
        mem.add_experience(s3, 2.0, 20, False)
        
        # Sample multiple times to ensure we never get invalid transitions
        for _ in range(20):
            states, actions, rewards, next_states, dones = mem.sample_batch(batch_size=2)
            
            for i in range(len(states)):
                state = states[i]
                
                # s1 is terminal, should never be a starting state
                assert not np.allclose(state, s1), \
                    "Terminal state s1 should not be sampled as starting state"
                
                # Verify valid transitions
                if np.allclose(state, s0):
                    assert actions[i] == 10
                    assert rewards[i] == 1.0
                    assert np.allclose(next_states[i], s1)
                    assert dones[i] == True
                elif np.allclose(state, s2):
                    assert actions[i] == 20
                    assert rewards[i] == 2.0
                    assert np.allclose(next_states[i], s3)
                    assert dones[i] == False


class TestPrioritizedMemoryTransitionAlignment:
    """Test that PrioritizedMemory.sample_batch returns correctly aligned transitions."""

    def test_basic_transition_alignment(self):
        """
        Test prioritized replay has correct transition alignment.
        Same logic as Memory test but with PrioritizedMemory.
        """
        np.random.seed(42)
        
        mem = PrioritizedMemory(max_len=100, observation_dim=4)
        
        s0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        s2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        
        a0, r0 = 10, 1.0
        a1, r1 = 20, 2.0
        
        mem.add_experience(s0, 0.0, 0, False)
        mem.add_experience(s1, r0, a0, False)
        mem.add_experience(s2, r1, a1, False)
        
        # PrioritizedMemory returns extra values (indices, weights)
        result = mem.sample_batch(batch_size=2)
        assert result is not None, "sample_batch should return data"
        
        states, actions, rewards, next_states, dones, indices, weights = result
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            
            if np.allclose(state, s0):
                assert action == a0, f"Expected action {a0} for s0, got {action}"
                assert reward == r0, f"Expected reward {r0} for s0->s1, got {reward}"
                assert np.allclose(next_state, s1)
            elif np.allclose(state, s1):
                assert action == a1, f"Expected action {a1} for s1, got {action}"
                assert reward == r1, f"Expected reward {r1} for s1->s2, got {reward}"
                assert np.allclose(next_state, s2)
            else:
                pytest.fail(f"Unexpected state: {state}")

    def test_done_flag_boundary(self):
        """Test prioritized replay respects episode boundaries."""
        np.random.seed(42)
        
        mem = PrioritizedMemory(max_len=100, observation_dim=4)
        
        s0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        s2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        s3 = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
        
        mem.add_experience(s0, 0.0, 0, False)
        mem.add_experience(s1, 1.0, 10, True)  # Terminal
        mem.add_experience(s2, 0.0, 0, False)
        mem.add_experience(s3, 2.0, 20, False)
        
        for _ in range(20):
            result = mem.sample_batch(batch_size=2)
            if result is None:
                continue
                
            states, actions, rewards, next_states, dones, indices, weights = result
            
            for i in range(len(states)):
                state = states[i]
                
                assert not np.allclose(state, s1), \
                    "Terminal state s1 should not be sampled as starting state"
                
                if np.allclose(state, s0):
                    assert actions[i] == 10
                    assert rewards[i] == 1.0
                    assert dones[i] == True
                elif np.allclose(state, s2):
                    assert actions[i] == 20
                    assert rewards[i] == 2.0
                    assert dones[i] == False

    def test_priority_update(self):
        """Test that priority updates work correctly after sampling."""
        np.random.seed(42)
        
        mem = PrioritizedMemory(max_len=100, observation_dim=4)
        
        # Add some experiences
        for i in range(5):
            state = np.full(4, float(i), dtype=np.float32)
            mem.add_experience(state, float(i) * 0.1, i, False)
        
        result = mem.sample_batch(batch_size=2)
        assert result is not None
        
        states, actions, rewards, next_states, dones, indices, weights = result
        
        # Update priorities with fake TD errors
        td_errors = np.array([0.5, 1.5])
        mem.update_priorities(indices, td_errors)
        
        # Check that priorities were updated
        for idx, td_error in zip(indices, td_errors):
            expected_priority = abs(td_error) + mem.epsilon
            assert np.isclose(mem.priorities[idx], expected_priority), \
                f"Priority at {idx} should be {expected_priority}, got {mem.priorities[idx]}"


class TestMemoryEdgeCases:
    """Test edge cases for memory buffer."""

    def test_circular_buffer_wraparound(self):
        """Test that circular buffer wraparound works correctly."""
        np.random.seed(42)
        
        mem = Memory(max_len=5, observation_dim=2)
        
        # Fill buffer completely and then some
        for i in range(8):
            state = np.array([float(i), float(i)], dtype=np.float32)
            mem.add_experience(state, float(i) * 0.1, i, False)
        
        # Buffer should contain states 3, 4, 5, 6, 7 (oldest 0, 1, 2 overwritten)
        assert mem.size == 5
        assert mem.position == 3  # Next write position
        
        # Sample and verify we only get valid states (3-7)
        states, actions, rewards, next_states, dones = mem.sample_batch(batch_size=3)
        
        for state in states:
            val = state[0]
            assert 3 <= val <= 7, f"State {val} should be in range [3, 7]"

    def test_empty_buffer_handling(self):
        """Test that empty buffer returns appropriate response."""
        mem = Memory(max_len=100, observation_dim=4)
        
        # No experiences added
        valid_indices = mem._get_valid_indices()
        assert len(valid_indices) == 0

    def test_single_experience(self):
        """Test buffer with single experience."""
        mem = Memory(max_len=100, observation_dim=4)
        
        s0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mem.add_experience(s0, 0.0, 0, False)
        
        # Single experience means no valid transitions
        valid_indices = mem._get_valid_indices()
        assert len(valid_indices) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

