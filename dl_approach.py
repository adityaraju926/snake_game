import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snake.game import ACTIONS, SnakeGame

weight_path = 'dl_agent_weights.pth'
features = 11


class DLAgent:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def network():
            return nn.Sequential(
                nn.Linear(features, 256), nn.ReLU(),
                nn.Linear(256, 128),        nn.ReLU(),
                nn.Linear(128, 4),
            ).to(self.device)

        self.q_network = network()
        self.target_network = network()
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=20_000)

        self.epsilon = 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.batch_size = 64
        self.gamma = 0.99
        self.target_update_freq = 200
        self.steps_done = 0

        self.load_model()

    def get_action(self, game):
        # Next action based on current state
        features = self.extract_features(game)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(features).unsqueeze(0).to(self.device))
        return ACTIONS[q_values.argmax(dim=1).item()]

    def train(self, n_episodes=1000, callback=None):
        # Training DQN agent
        self.epsilon = 1.0
        game = SnakeGame()

        for episode in range(n_episodes):
            game.reset()
            done = False
            features = self.extract_features(game)

            while not done:
                if random.random() < self.epsilon:
                    action = random.choice(ACTIONS)
                    action_idx = ACTIONS.index(action)
                else:
                    self.q_network.eval()
                    with torch.no_grad():
                        q_values = self.q_network(torch.tensor(features).unsqueeze(0).to(self.device))
                    action_idx = q_values.argmax(dim=1).item()
                    action = ACTIONS[action_idx]

                _, reward, done = game.step(action)
                next_features = self.extract_features(game)

                self.buffer.append((features, action_idx, reward, next_features, done))
                features = next_features
                self.steps_done += 1

                if len(self.buffer) >= self.batch_size:
                    self.update_network()

                if self.steps_done % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if callback:
                callback(episode, game.score, self.epsilon)
            elif (episode + 1) % 100 == 0:
                print(f'Episode {episode+1:>4}/{n_episodes} | Score: {game.score} | ε: {self.epsilon:.3f}')

        self.save_model()

    def extract_features(self, game):
        head_x, head_y = game.snake[0]

        # Finding head based on past move
        if len(game.snake) > 1:
            prev_x, prev_y = game.snake[1]
            direction = (head_x - prev_x, head_y - prev_y)
        else:
            direction = (1, 0)

        up    = direction == (0, -1)
        right = direction == (1,  0)
        down  = direction == (0,  1)
        left  = direction == (-1, 0)

        # Movement relative to the head of snake
        if up:
            straight, turn_right, turn_left = (0, -1), (1, 0), (-1, 0)
        elif right:
            straight, turn_right, turn_left = (1, 0), (0, 1), (0, -1)
        elif down:
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
            float(up),
            float(right),
            float(down),
            float(left),
            float(food_y < head_y),   # Checks food relative to the head
            float(food_x > head_x),   
            float(food_y > head_y),   
            float(food_x < head_x),   
        ], dtype=np.float32)

        return features

    def update_network(self):
        # Sample random batch from past, compute the Bellman target, and updates gradient
        self.q_network.train()
        feats, action_idxs, rewards, next_feats, dones = zip(*random.sample(self.buffer, self.batch_size))

        state_tensors = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
        next_state_tensors = torch.tensor(np.array(next_feats), dtype=torch.float32).to(self.device)
        action_tensors = torch.tensor(action_idxs, dtype=torch.long).to(self.device)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_tensors = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q values for action taken
        q_values = self.q_network(state_tensors).gather(1, action_tensors.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors).max(1)[0]
            targets = reward_tensors + self.gamma * next_q_values * (1 - done_tensors)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path=weight_path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path=weight_path): # picks up weights from dl_agent_weights file
        if os.path.exists(path):
            self.q_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == '__main__':
    agent = DLAgent()
    agent.train(n_episodes=1000)
