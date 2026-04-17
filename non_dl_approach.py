from collections import deque
from snake.game import ACTIONS


class NonDLAgent:

    def __init__(self):
        self.last_planned_path = None

    def get_action(self, game):
        '''Choose the next action given the current game state.'''
        # Perception: get game state
        head_position = game.snake[0]
        snake_body_cells = set(game.snake)
        food_position = game.food
        board_width = game.width
        board_height = game.height

        # Planning: shortest path to food using BFS
        path_to_food = self._bfs_shortest_path(
            start_cell=head_position,
            goal_cell=food_position,
            blocked_cells=snake_body_cells,
            board_width=board_width,
            board_height=board_height)
        self.last_planned_path = path_to_food

        # Control: move one step along the planned path
        if path_to_food is not None:
            next_cell = path_to_food[1]
            head_x, head_y = head_position
            next_x, next_y = next_cell
            return (next_x - head_x, next_y - head_y)

        
        return self._find_any_safe_move( # If no path to food, pick any safe move
            head_position=head_position,
            snake_body_cells=snake_body_cells,
            board_width=board_width,
            board_height=board_height,
        )

    def _bfs_shortest_path(self, start_cell, goal_cell, blocked_cells, board_width, board_height):
        '''Breadth-first search for shortest path'''
        cells_to_explore = deque()
        cells_to_explore.append(start_cell)

        came_from = {start_cell: None} # Maps each visited cell to the cell we arrived from

        while cells_to_explore:
            current_cell = cells_to_explore.popleft() # Get the next cell to explore

            if current_cell == goal_cell:
                path_in_reverse = []
                trace_cell = goal_cell
                while trace_cell is not None: # Walk backwards from goal to start
                    path_in_reverse.append(trace_cell)
                    trace_cell = came_from[trace_cell]
                path_in_reverse.reverse()
                return path_in_reverse

            current_x, current_y = current_cell
            for move_x, move_y in ACTIONS:
                neighbor_cell = (current_x + move_x, current_y + move_y) # Check each adjacent cell

                if neighbor_cell in came_from: #Already visited
                    continue
                if not self._is_in_bounds(neighbor_cell, board_width, board_height): # Wall
                    continue
                if neighbor_cell in blocked_cells: # Snake body
                    continue

                came_from[neighbor_cell] = current_cell
                cells_to_explore.append(neighbor_cell)

        return None

    def _is_in_bounds(self, cell, board_width, board_height):
        '''Function to define the boarders of the game board'''
        cell_x, cell_y = cell
        if cell_x < 0 or cell_x >= board_width:
            return False
        if cell_y < 0 or cell_y >= board_height:
            return False
        return True

    def _find_any_safe_move(self, head_position, snake_body_cells, board_width, board_height):
        '''Pick any in bounds and non body move'''
        head_x, head_y = head_position
        for candidate_action in ACTIONS:
            move_x, move_y = candidate_action
            next_cell = (head_x + move_x, head_y + move_y)

            if not self._is_in_bounds(next_cell, board_width, board_height):
                continue
            if next_cell in snake_body_cells:
                continue

            return candidate_action
        
        return ACTIONS[0] # End game if every direction is a dead end
