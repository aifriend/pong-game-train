"""
Agent memory for storing experiences in DQN training.
Efficient numpy-based implementation with proper batch sampling.
"""

import numpy as np


class Memory:
    """
    Efficient experience replay buffer for DQN agent.

    Uses pre-allocated numpy arrays for memory efficiency and fast sampling.
    Implements circular buffer pattern for O(1) insertions.
    """

    def __init__(self, max_len, observation_dim=9):
        """
        Initialize memory buffer with pre-allocated numpy arrays.

        Args:
            max_len: Maximum number of experiences to store
            observation_dim: Dimension of observation space (default: 9 for Pong)
        """
        self.max_len = max_len
        self.observation_dim = observation_dim
        self.position = 0  # Current write position
        self.size = 0  # Current number of stored experiences

        # Pre-allocate numpy arrays for efficiency
        self.frames = np.zeros((max_len, observation_dim), dtype=np.float32)
        self.actions = np.zeros(max_len, dtype=np.int32)
        self.rewards = np.zeros(max_len, dtype=np.float32)
        self.done_flags = np.zeros(max_len, dtype=np.bool_)

    def add_experience(self, next_frame, next_frames_reward, next_action, next_frame_terminal):
        """
        Add an experience to the memory buffer.

        Args:
            next_frame: Next state observation (numpy array)
            next_frames_reward: Reward received
            next_action: Action taken
            next_frame_terminal: Whether episode terminated
        """
        self.frames[self.position] = next_frame
        self.actions[self.position] = next_action
        self.rewards[self.position] = next_frames_reward
        self.done_flags[self.position] = next_frame_terminal

        # Update position (circular buffer)
        self.position = (self.position + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)

    def sample_batch(self, batch_size):
        """
        Sample a batch of valid transitions efficiently.

        A valid transition is (state, action, reward, next_state, done) where
        neither state nor next_state crosses an episode boundary.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) numpy arrays
        """
        # Get valid indices (not at episode boundaries)
        valid_indices = self._get_valid_indices()

        if len(valid_indices) < batch_size:
            # If not enough valid indices, sample with replacement
            sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        else:
            sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=False)

        # Gather batch data
        states = self.frames[sampled_indices]
        next_indices = (sampled_indices + 1) % self.max_len
        next_states = self.frames[next_indices]
        actions = self.actions[sampled_indices]
        rewards = self.rewards[next_indices]  # Reward received after taking action
        dones = self.done_flags[next_indices]

        return states, actions, rewards, next_states, dones

    def _get_valid_indices(self):
        """
        Get indices that are valid for sampling (not crossing episode boundaries).

        Returns:
            Array of valid indices
        """
        if self.size < 2:
            return np.array([], dtype=np.int32)

        # Valid indices are those where current state and next state are both valid
        # and current state is not terminal (done)
        all_indices = np.arange(self.size - 1)  # Exclude last position

        # Remove indices where done_flag is True (episode ended)
        valid_mask = ~self.done_flags[all_indices]

        # Also remove indices too close to write position in circular buffer
        if self.size == self.max_len:
            # Buffer is full, avoid position-1 to position+1
            invalid_pos = [
                (self.position - 1) % self.max_len,
                self.position,
                (self.position + 1) % self.max_len,
            ]
            for pos in invalid_pos:
                if pos < len(valid_mask):
                    valid_mask[pos] = False

        return all_indices[valid_mask]

    def __len__(self):
        """Return current size of buffer."""
        return self.size

    @property
    def is_ready(self):
        """Check if buffer has enough samples for training."""
        return self.size > 1000  # Minimum samples before training


class PrioritizedMemory(Memory):
    """
    Prioritized Experience Replay buffer.

    Samples experiences based on their TD error priority,
    giving more weight to surprising/important transitions.
    """

    def __init__(self, max_len, observation_dim=9, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize prioritized memory buffer.

        Args:
            max_len: Maximum number of experiences
            observation_dim: Dimension of observation space
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed from beta to 1)
            beta_increment: How much to increase beta per sample
        """
        super().__init__(max_len, observation_dim)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.ones(max_len, dtype=np.float32)
        self.max_priority = 1.0
        self.epsilon = 1e-6  # Small constant for numerical stability

    def add_experience(self, next_frame, next_frames_reward, next_action, next_frame_terminal):
        """Add experience with max priority."""
        # New experiences get max priority to ensure they're sampled at least once
        self.priorities[self.position] = self.max_priority
        super().add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    def sample_batch(self, batch_size):
        """
        Sample batch with prioritized sampling.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        valid_indices = self._get_valid_indices()
        if len(valid_indices) == 0:
            return None

        # Calculate sampling probabilities
        priorities = self.priorities[valid_indices] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices based on priority
        sampled_idx = np.random.choice(
            len(valid_indices),
            size=min(batch_size, len(valid_indices)),
            replace=False,
            p=probabilities,
        )
        sampled_indices = valid_indices[sampled_idx]

        # Calculate importance sampling weights
        N = len(valid_indices)
        weights = (N * probabilities[sampled_idx]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights

        # Anneal beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Gather batch data
        states = self.frames[sampled_indices]
        next_indices = (sampled_indices + 1) % self.max_len
        next_states = self.frames[next_indices]
        actions = self.actions[sampled_indices]
        rewards = self.rewards[next_indices]
        dones = self.done_flags[next_indices]

        return states, actions, rewards, next_states, dones, sampled_indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of experiences to update
            td_errors: Corresponding TD errors
        """
        priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
