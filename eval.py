import numpy as np
from snake.game import SnakeGame
from non_dl_approach import NonDLAgent
from dl_approach import DLAgent

N_GAMES = 100


MAX_STEPS = 2000  # Cap per game to prevent infinite loops on long-running agents


def evaluate(agent, n_games=N_GAMES):
    '''Run n_games and return scores and step counts.'''
    scores = []
    steps = []

    for _ in range(n_games):
        game = SnakeGame()
        game.reset()
        step_count = 0

        while not game.done and step_count < MAX_STEPS:
            action = agent.get_action(game)
            game.step(action)
            step_count += 1

        scores.append(game.score)
        steps.append(step_count)

    return np.array(scores), np.array(steps)


def print_table(results):
    col_w = 14
    metric_w = 18

    header = f"{'Metric':<{metric_w}} {'Non-DL Agent':>{col_w}} {'DL Agent':>{col_w}}"
    divider = '-' * len(header)

    print()
    print(divider)
    print(header)
    print(divider)
    for metric, non_dl_val, dl_val in results:
        print(f"{metric:<{metric_w}} {non_dl_val:>{col_w}} {dl_val:>{col_w}}")
    print(divider)
    print()


if __name__ == '__main__':
    print(f'Evaluating agents over {N_GAMES} games each...\n')

    print('Running Non-DL Agent...')
    non_dl_scores, non_dl_steps = evaluate(NonDLAgent())

    print('Running DL Agent...')
    dl_scores, dl_steps = evaluate(DLAgent())

    results = [
        ('Avg Score',     f'{non_dl_scores.mean():.2f}', f'{dl_scores.mean():.2f}'),
        ('Avg Steps',     f'{non_dl_steps.mean():.1f}',  f'{dl_steps.mean():.1f}'),
        ('Score Std Dev', f'{non_dl_scores.std():.2f}',  f'{dl_scores.std():.2f}'),
        ('Max Score',     f'{non_dl_scores.max()}',       f'{dl_scores.max()}'),
    ]

    print_table(results)
