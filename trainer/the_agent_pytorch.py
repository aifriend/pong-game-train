"""
DQN Agent implementation using PyTorch.
Adapted for the custom Pong environment with vector observations.
Implements Double DQN for reduced overestimation bias.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from torch.utils.tensorboard import SummaryWriter
from .agent_memory import Memory, PrioritizedMemory


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Noisy Networks.

    Adds learnable noise to weights and biases for state-dependent exploration.
    This replaces epsilon-greedy exploration with learned exploration strategies.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters, don't get gradients)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Generate factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Generate new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DQNNetwork(nn.Module):
    """
    Enhanced Dueling Deep Q-Network for Pong game.

    Features:
    - Doubled network capacity (~80K params)
    - Residual connections for better gradient flow
    - Noisy layers throughout for exploration
    - Separate Value and Advantage streams (Dueling architecture)

    Q(s,a) = V(s) + A(s,a) - mean(A(s))
    """

    def __init__(self, observation_dim, num_actions):
        """
        Initialize Enhanced Dueling DQN network.

        Args:
            observation_dim: Dimension of observation space (9 for Pong-v0)
            num_actions: Number of possible actions (3 for Pong)
        """
        super(DQNNetwork, self).__init__()

        # Expanded feature extractor with residual connections
        # First block: 9 -> 256
        self.fc1 = nn.Linear(observation_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        # Second block: 256 -> 256 (with residual)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)

        # Third block: 256 -> 128
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        # Value stream: estimates V(s) with noisy exploration
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, 1)  # Single value for the state
        )

        # Advantage stream: estimates A(s,a) with noisy exploration
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, num_actions),  # One advantage per action
        )

        self.num_actions = num_actions

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with improved initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Enhanced Dueling DQN.

        Args:
            x: Input observations

        Returns:
            Q-values combining value and advantage streams
        """
        # Feature extraction with residual connections
        # Block 1
        h1 = F.relu(self.ln1(self.fc1(x)))

        # Block 2 with residual connection
        h2 = F.relu(self.ln2(self.fc2(h1)))
        h2 = h2 + h1  # Residual connection

        # Block 3
        features = F.relu(self.ln3(self.fc3(h2)))

        # Separate value and advantage computations
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class Agent:
    """
    DQN Agent for Pong game.
    Uses PyTorch for neural network implementation.
    """

    def __init__(
        self,
        possible_actions,
        starting_mem_len,
        max_mem_len,
        learn_rate,
        observation_dim=9,
        starting_lives=5,
        debug=False,
        tensorboard_log=None,
        learn_every=8,
        batch_size=128,
        target_update_freq=5000,
    ):
        """
        Initialize Double DQN agent.

        Args:
            possible_actions: List of possible action values
            starting_mem_len: Minimum memory size before learning starts
            max_mem_len: Maximum memory buffer size
            learn_rate: Learning rate for optimizer
            observation_dim: Dimension of observation space (default: 9 for Pong-v0)
            starting_lives: Not used for Pong, kept for compatibility
            debug: Debug mode flag
            tensorboard_log: Path for TensorBoard logs (None to disable)
            learn_every: Learn every N steps (default: 8, for diverse experience)
            batch_size: Training batch size (default: 128, for stable gradients)
            target_update_freq: Update target network every N learns (default: 5000)
        """
        # Use prioritized experience replay for better sample efficiency
        self.memory = PrioritizedMemory(max_mem_len, observation_dim)
        self.possible_actions = possible_actions
        self.gamma = 0.99  # Increased discount factor for longer-term planning
        self.learn_rate = learn_rate
        self.observation_dim = observation_dim
        # Use Metal GPU on Mac, CUDA on NVIDIA, fallback to CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Metal Performance Shaders
        else:
            self.device = torch.device("cpu")

        print(f"🔧 Training device: {self.device}")

        # Build main and target networks
        self.model = DQNNetwork(observation_dim, len(possible_actions)).to(self.device)
        self.model_target = DQNNetwork(observation_dim, len(possible_actions)).to(self.device)

        # Copy weights from main to target network
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()  # Target network is always in eval mode

        # Frozen opponent network for stable self-play
        # This provides a more consistent training signal
        self.model_opponent = DQNNetwork(observation_dim, len(possible_actions)).to(self.device)
        self.model_opponent.load_state_dict(self.model.state_dict())
        self.model_opponent.eval()
        self.opponent_update_freq = 500  # Update opponent every N episodes
        self.episodes_since_opponent_update = 0

        # Optimizer and loss function (no reduction for prioritized replay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        # Learning rate scheduler: reduce LR when learning plateaus
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5000
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Element-wise Huber loss

        self.total_timesteps = 0
        self.lives = starting_lives  # Not used for Pong
        self.starting_mem_len = starting_mem_len
        self.learns = 0
        self._last_loss = 0.0
        self.learn_every = learn_every
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # TensorBoard logging
        self.writer = None
        if tensorboard_log:
            self.writer = SummaryWriter(tensorboard_log)

        # Print model summary
        self._print_model_summary()
        print("\nDouble DQN Agent Initialized (PyTorch)\n")

    def _print_model_summary(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nDQN Network Architecture:")
        print(f"  Input: {self.observation_dim} dimensions")
        print(f"  Output: {len(self.possible_actions)} actions")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {self.device}")

    def get_action(self, state, use_noise=True):
        """
        Get action using noisy networks (when use_noise=True) or greedy policy.

        Args:
            state: Current observation state (numpy array)
            use_noise: Whether to use noisy exploration (training) or greedy (testing)

        Returns:
            Selected action
        """
        if not use_noise:
            # Greedy evaluation (no noise)
            self.model.eval()
        else:
            # Training mode with noise for exploration
            self.model.train()
            # Reset noise for each action selection
            self.model.reset_noise()

        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            a_index = q_values.argmax().item()

        return self.possible_actions[a_index]

    def get_opponent_action(self, state):
        """
        Get opponent action using frozen opponent network (greedy, no exploration).

        Args:
            state: Current observation state (numpy array)

        Returns:
            Selected action for opponent
        """
        self.model_opponent.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model_opponent(state_tensor)
            a_index = q_values.argmax().item()

        return self.possible_actions[a_index]

    def update_opponent(self, episode):
        """
        Update frozen opponent network periodically for stable self-play.

        Args:
            episode: Current episode number
        """
        self.episodes_since_opponent_update += 1
        if self.episodes_since_opponent_update >= self.opponent_update_freq:
            self.model_opponent.load_state_dict(self.model.state_dict())
            self.episodes_since_opponent_update = 0
            if self.writer:
                self.writer.add_scalar("Training/Opponent_Updated", 1, episode)

    def log_episode(self, episode, score, episode_length):
        """
        Log episode statistics to TensorBoard.

        Args:
            episode: Episode number
            score: Episode total reward
            episode_length: Number of steps in episode
        """
        # Update opponent periodically for stable self-play
        self.update_opponent(episode)

        if self.writer:
            self.writer.add_scalar("Episode/Score", score, episode)
            self.writer.add_scalar("Episode/Length", episode_length, episode)

    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

    def learn(self, debug=False):
        """
        Train the agent using Double DQN with experience replay.

        Double DQN improvement: Use main network to SELECT best action,
        but use target network to EVALUATE that action's Q-value.
        This reduces overestimation bias in Q-learning.

        Args:
            debug: Debug mode flag
        """
        # Use prioritized batch sampling from memory
        batch = self.memory.sample_batch(self.batch_size)
        if batch is None:
            return

        states, actions_taken, rewards, next_states, done_flags, indices, weights = batch

        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        done_flags_tensor = torch.BoolTensor(done_flags).to(self.device)
        actions_tensor = torch.LongTensor(actions_taken).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # Get current Q-values from main network
        self.model.train()
        self.model.reset_noise()  # Reset noise for training
        current_q_values = self.model(states_tensor)

        # DOUBLE DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Main network selects best action for next state
            next_q_values_main = self.model(next_states_tensor)
            best_actions = next_q_values_main.argmax(dim=1)

            # Target network evaluates Q-value of selected action
            next_q_values_target = self.model_target(next_states_tensor)
            next_q_values = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Bellman equation
        # Q(s,a) = r + gamma * Q_target(s', argmax_a' Q_main(s', a'))
        target_q_values = rewards_tensor + self.gamma * next_q_values * (~done_flags_tensor)

        # Get Q-values for actions taken
        current_q_for_actions = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Compute TD errors for priority updates
        td_errors = target_q_values - current_q_for_actions

        # Compute weighted loss using importance sampling weights
        elementwise_loss = self.loss_fn(current_q_for_actions, target_q_values)
        loss = (elementwise_loss * weights_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm BEFORE clipping (for diagnostics)
        grad_norm_pre_clip = 0.0
        if self.writer and self.learns % 100 == 0:  # Only calculate when logging AND writer exists
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm_pre_clip += p.grad.data.norm(2).item() ** 2
            grad_norm_pre_clip = grad_norm_pre_clip**0.5

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learns += 1
        self._last_loss = loss.item()

        # Update learning rate based on loss (every 100 learns)
        if self.learns % 100 == 0:
            self.lr_scheduler.step(loss.item())

        # Update priorities in replay buffer based on TD errors
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Comprehensive TensorBoard logging for learning diagnostics
        if self.writer and self.learns % 100 == 0:
            # Loss and learning rate
            self.writer.add_scalar("Training/Loss", loss.item(), self.learns)
            self.writer.add_scalar(
                "Training/Learning_Rate", self.optimizer.param_groups[0]["lr"], self.learns
            )

            # Q-value distribution (key diagnostic for learning health)
            self.writer.add_scalar(
                "Q_Values/Mean", current_q_for_actions.mean().item(), self.learns
            )
            self.writer.add_scalar("Q_Values/Max", current_q_for_actions.max().item(), self.learns)
            self.writer.add_scalar("Q_Values/Min", current_q_for_actions.min().item(), self.learns)
            self.writer.add_scalar("Q_Values/Std", current_q_for_actions.std().item(), self.learns)

            # TD errors (measure of prediction accuracy)
            self.writer.add_scalar("TD_Error/Mean", td_errors.abs().mean().item(), self.learns)
            self.writer.add_scalar("TD_Error/Max", td_errors.abs().max().item(), self.learns)

            # Target Q-values
            self.writer.add_scalar("Target_Q/Mean", target_q_values.mean().item(), self.learns)

            # Reward statistics from batch
            self.writer.add_scalar("Batch/Reward_Mean", rewards_tensor.mean().item(), self.learns)
            self.writer.add_scalar("Batch/Reward_Max", rewards_tensor.max().item(), self.learns)

            # Gradient norm (measure of learning signal strength)
            # Calculated before clipping and optimizer step for accuracy
            self.writer.add_scalar(
                "Training/Gradient_Norm_PreClip", grad_norm_pre_clip, self.learns
            )

        # Update target network periodically
        if self.learns % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            if debug:
                print(f"\nTarget model updated (learns: {self.learns}, loss: {loss.item():.4f})")

    def save_weights(self, filepath):
        """
        Save model weights to file.

        Args:
            filepath: Path to save weights
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_target_state_dict": self.model_target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
                "learns": self.learns,
            },
            filepath,
        )

    def load_weights(self, filepath):
        """
        Load model weights from file.

        Args:
            filepath: Path to load weights from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_target.load_state_dict(checkpoint["model_target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.learns = checkpoint.get("learns", 0)
