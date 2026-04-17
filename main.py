import time
import streamlit as st
from snake.game import SnakeGame
from non_dl_approach import NonDLAgent
# from dl_approach import DLAgent

CELL_SIZE = 40  # Pixels per grid cell
COLORS = {
    'empty': "#113246",
    'snake_head': '#1a9c54',
    'snake_body': '#27ae60',
    'food': '#ff4444',
    'grid_line': "#0C232E"}
AGENTS = {
    'BFS Search Agent': NonDLAgent,
    # 'Deep Learning Agent': DLAgent
}

def create_board(game):
    '''Create the game board as an SVG string'''
    board_width_px = game.width * CELL_SIZE
    board_height_px = game.height * CELL_SIZE

    svg_parts = []
    svg_parts.append(f'<svg width="{board_width_px}" height="{board_height_px}" xmlns="http://www.w3.org/2000/svg">') # Header
    svg_parts.append(f'<rect width="{board_width_px}" height="{board_height_px}" fill="{COLORS["empty"]}"/>') # Background

    for col in range(game.width + 1):
        x = col * CELL_SIZE
        svg_parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{board_height_px}" stroke="{COLORS["grid_line"]}" stroke-width="1"/>') # Horizontal grid lines
    for row in range(game.height + 1):
        y = row * CELL_SIZE
        svg_parts.append(f'<line x1="0" y1="{y}" x2="{board_width_px}" y2="{y}" stroke="{COLORS["grid_line"]}" stroke-width="1"/>') # Vertical grid lines

    head_x, head_y = game.snake[0]
    for seg_x, seg_y in game.snake[1:]:
        svg_parts.append(f'<rect x="{seg_x * CELL_SIZE + 1}" y="{seg_y * CELL_SIZE + 1}" width="{CELL_SIZE - 2}" height="{CELL_SIZE - 2}" rx="4" fill="{COLORS["snake_body"]}"/>') # Snake body

    svg_parts.append(f'<rect x="{head_x * CELL_SIZE + 1}" y="{head_y * CELL_SIZE + 1}" width="{CELL_SIZE - 2}" height="{CELL_SIZE - 2}" rx="6" fill="{COLORS["snake_head"]}"/>') # Snake head

    food_x, food_y = game.food
    food_center_x = food_x * CELL_SIZE + CELL_SIZE // 2
    food_center_y = food_y * CELL_SIZE + CELL_SIZE // 2
    svg_parts.append(f'<circle cx="{food_center_x}" cy="{food_center_y}" r="{CELL_SIZE // 3}" fill="{COLORS["food"]}"/>') # Food

    svg_parts.append('</svg>') # Footer
    return '\n'.join(svg_parts) # Create full SVG string

def session_state():
    '''Set up session state on first load'''
    if 'game' not in st.session_state:
        st.session_state.game = SnakeGame()
        st.session_state.agent = NonDLAgent()
        st.session_state.running = False
        st.session_state.speed = 10
        st.session_state.games_played = 0
        st.session_state.high_score = 0

def reset_game():
    '''Start a new game'''
    st.session_state.game = SnakeGame()
    st.session_state.running = False
    st.session_state.games_played += 1

def main():
    st.set_page_config(page_title='Snake Agent', layout='wide')
    st.title('Snake Game Agent')
    session_state()

    game = st.session_state.game
    agent = st.session_state.agent

    with st.sidebar:
        st.header('Controls')

        # Agent selection
        agent_name = st.selectbox('Agent', list(AGENTS.keys()))
        selected_agent_class = AGENTS[agent_name]
        if not isinstance(agent, selected_agent_class): # Only create a new agent instance if the selection changed
            st.session_state.agent = selected_agent_class()
            agent = st.session_state.agent

        st.session_state.speed = st.slider('Steps per second', 1, 30, st.session_state.speed)

        col_start, col_reset = st.columns(2)
        with col_start:
            if st.button('Start' if not st.session_state.running else 'Pause', use_container_width=True): # Start/pause toggle button
                st.session_state.running = not st.session_state.running
        with col_reset:
            if st.button('Reset', use_container_width=True): # Reset button
                reset_game()
                st.rerun()

        # Stats
        st.header('Stats')
        st.metric('Score', game.score)
        st.metric('Snake Length', len(game.snake))
        st.metric('High Score', st.session_state.high_score)
        st.metric('Games Played', st.session_state.games_played)

    # Game board
    board_placeholder = st.empty()
    board_placeholder.markdown(create_board(game), unsafe_allow_html=True)

    # Game loop: swaps the board in place so the page never re-renders
    while st.session_state.running and not game.done:
        action = agent.get_action(game)
        game.step(action)

        if game.score > st.session_state.high_score:
            st.session_state.high_score = game.score

        board_placeholder.markdown(create_board(game), unsafe_allow_html=True)
        time.sleep(1.0 / st.session_state.speed)

    if game.done:
        st.session_state.running = False
        st.warning(f'Game over! Final score: {game.score}')

if __name__ == '__main__':
    main()
