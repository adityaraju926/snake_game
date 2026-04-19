import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snake.game import ACTIONS, SnakeGame

MODEL_PATH = 'dl_agent_weights.pth'

N_FEATURES = 11  # Size of the compact state vector (see _extract_features)


class QNetwork(nn.Module):
    '''
    Small MLP that takes 11 spatial features extracted from the board
    and outputs Q-values for each of the four actions.

    Input:  (batch, 11) feature vector
    Output: (batch, 4)  Q-values for [UP, RIGHT, DOWN, LEFT]
    '''

    def __init__(self, n_features=N_FEATURES, n_actions=4):
        super().__init__()

        # Perception + Planning: FC layers map spatial features to action Q-values
        self.fc_layers = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.fc_layers(x)


class ReplayBuffer:
    '''Fixed-size circular buffer storing (s, a, r, s', done) transitions.'''

    def __init__(self, capacity=20_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, features, action_idx, reward, next_features, done):
        self.buffer.append((features, action_idx, reward, next_features, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DLAgent:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = 0.05         # Low by default; set to 1.0 at start of training
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.batch_size = 64
        self.gamma = 0.99
        self.target_update_freq = 200
        self.steps_done = 0

        self._load_model()

    # ------------------------------------------------------------------
    # Public interface (same pattern as NonDLAgent)
    # ------------------------------------------------------------------

    def get_action(self, game):
        '''Choose the next action given the current game state.'''

        # Perception: extract 11 spatial features from the board
        features = self._extract_features(game)
        state_tensor = self._to_tensor(features)

        # Planning: Q-network scores every action
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)  # shape (1, 4)

        # Control: take the action with the highest Q-value
        best_action_idx = q_values.argmax(dim=1).item()
        return ACTIONS[best_action_idx]

    def train(self, n_episodes=1_000, callback=None):
        '''Train the DQN agent from scratch using self-play.'''

        self.epsilon = 1.0  # Start with full exploration
        game = SnakeGame()

        for episode in range(n_episodes):
            game.reset()
            done = False
            features = self._extract_features(game)

            while not done:
                # Epsilon-greedy: explore randomly or exploit the Q-network
                if random.random() < self.epsilon:
                    action = random.choice(ACTIONS)
                    action_idx = ACTIONS.index(action)
                else:
                    self.q_network.eval()
                    with torch.no_grad():
                        q_values = self.q_network(self._to_tensor(features))
                    action_idx = q_values.argmax(dim=1).item()
                    action = ACTIONS[action_idx]

                _, reward, done = game.step(action)
                next_features = self._extract_features(game)

                self.replay_buffer.push(features, action_idx, reward, next_features, done)
                features = next_features
                self.steps_done += 1

                if len(self.replay_buffer) >= self.batch_size:
                    self._update_network()

                if self.steps_done % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if callback:
                callback(episode, game.score, self.epsilon)
            elif (episode + 1) % 100 == 0:
                print(f'Episode {episode+1:>4}/{n_episodes} | Score: {game.score} | ε: {self.epsilon:.3f}')

        self._save_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, game):
        '''
        Perception: extract 11 spatial features from the board state.

        Features (all booleans cast to float):
          [0] Danger straight ahead
          [1] Danger to the right
          [2] Danger to the left
          [3-6] Current heading (up / right / down / left)
          [7-10] Food direction relative to head (up / right / down / left)
        '''
        head_x, head_y = game.snake[0]

        # Infer current heading from the last move (head vs second segment)
        if len(game.snake) > 1:
            prev_x, prev_y = game.snake[1]
            direction = (head_x - prev_x, head_y - prev_y)
        else:
            direction = (1, 0)  # Default: moving right on a fresh game

        dir_up    = direction == (0, -1)
        dir_right = direction == (1,  0)
        dir_down  = direction == (0,  1)
        dir_left  = direction == (-1, 0)

        # Straight / right / left relative to current heading
        if dir_up:
            straight, turn_right, turn_left = (0, -1), (1, 0), (-1, 0)
        elif dir_right:
            straight, turn_right, turn_left = (1, 0), (0, 1), (0, -1)
        elif dir_down:
            straight, turn_right, turn_left = (0, 1), (-1, 0), (1, 0)
        else:  # left
            straight, turn_right, turn_left = (-1, 0), (0, -1), (0, 1)

        body_cells = set(game.snake)

        def is_dangerous(dx, dy):
            nx, ny = head_x + dx, head_y + dy
            if nx < 0 or nx >= game.width or ny < 0 or ny >= game.height:
                return True
            return (nx, ny) in body_cells

        food_x, food_y = game.food

        features = np.array([
            float(is_dangerous(*straight)),
            float(is_dangerous(*turn_right)),
            float(is_dangerous(*turn_left)),
            float(dir_up),
            float(dir_right),
            float(dir_down),
            float(dir_left),
            float(food_y < head_y),   # food is up
            float(food_x > head_x),   # food is right
            float(food_y > head_y),   # food is down
            float(food_x < head_x),   # food is left
        ], dtype=np.float32)

        return features

    def _to_tensor(self, features):
        '''Convert a feature array to a (1, N_FEATURES) tensor.'''
        return torch.tensor(features).unsqueeze(0).to(self.device)

    def _update_network(self):
        '''Sample a mini-batch and perform one DQN gradient update.'''
        self.q_network.train()
        batch = self.replay_buffer.sample(self.batch_size)
        feats, action_idxs, rewards, next_feats, dones = zip(*batch)

        state_tensors      = torch.tensor(np.array(feats),      dtype=torch.float32).to(self.device)
        next_state_tensors = torch.tensor(np.array(next_feats), dtype=torch.float32).to(self.device)
        action_tensors     = torch.tensor(action_idxs,          dtype=torch.long).to(self.device)
        reward_tensors     = torch.tensor(rewards,               dtype=torch.float32).to(self.device)
        done_tensors       = torch.tensor(dones,                 dtype=torch.float32).to(self.device)

        # Q(s, a) for the actions that were actually taken
        q_values = self.q_network(state_tensors).gather(1, action_tensors.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors).max(1)[0]
            targets = reward_tensors + self.gamma * next_q_values * (1 - done_tensors)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save_model(self, path=MODEL_PATH):
        torch.save(self.q_network.state_dict(), path)
        print(f'Model saved → {path}')

    def _load_model(self, path=MODEL_PATH):
        if os.path.exists(path):
            self.q_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f'Model loaded ← {path}')
        else:
            print('No saved weights found — run training first (python dl_approach.py)')


if __name__ == '__main__':
    agent = DLAgent()
    agent.train(n_episodes=1_000)
