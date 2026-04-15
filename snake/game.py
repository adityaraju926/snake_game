import random

import numpy as np

# (horizonal, vertical) change
UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)
ACTIONS = [UP, RIGHT, DOWN, LEFT]


class SnakeGame:
    '''Snake environment for training the agents
      0 = empty cell
      1 = snake cell
      2 = food cell
    '''

    def __init__(self, width=10, height=10, seed=None):
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        '''Start a new game and return the first observation'''
        self.snake = [(self.width // 2, self.height // 2)] # Start in the middle of the board
        self.done = False
        self.score = 0
        self.spawn_food()
        return self.get_board()

    def spawn_food(self):
        '''Place food on an empty board cell'''
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                cell = (x, y)
                if cell in self.snake: # Skip cell with snake body
                    continue
                empty_cells.append(cell)

        self.food = self.random.choice(empty_cells) # Place food on a random empty cell

    def get_board(self):
        '''Return the current board state'''
        board = np.zeros((self.height, self.width), dtype=np.uint8) # 0=empty, 1=snake, 2=food
        for x, y in self.snake:
            board[y, x] = 1 # Place snake body
        food_x, food_y = self.food
        board[food_y, food_x] = 2 # Place food
        return board

    def step(self, action):
        '''Apply an action and return observation, reward, done'''
        if self.done:
            raise ValueError('Game is already over')
        if action not in ACTIONS:
            raise ValueError('Invalid action')

        head_x, head_y = self.snake[0]
        move_x, move_y = action
        next_head = (head_x + move_x, head_y + move_y)

        if not (0 <= next_head[0] < self.width and 0 <= next_head[1] < self.height): # Wall collision, game over
            self.done = True
            return self.get_board(), -10, True # Negative points for hitting the wall

        if next_head in self.snake: # Collision with itself, game over
            self.done = True
            return self.get_board(), -10, True # Negative points for hitting itself

        self.snake.insert(0, next_head)

        if next_head == self.food: # Food, grow snake and spawn new food
            self.score += 1
            self.spawn_food()
            reward = 10
        else:
            self.snake.pop() # Move forward without growing
            reward = 0

        return self.get_board(), reward, self.done

