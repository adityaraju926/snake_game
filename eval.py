import numpy as np
from snake.game import SnakeGame
from non_dl_approach import NonDLAgent
from dl_approach import DLAgent

games = 100


max_steps = 2000

def evaluate(agent, n_games=games):
    # Running 100 games and recording the scores and steps from each agent
    scores = []
    steps = []

    for _ in range(n_games):
        game = SnakeGame()
        game.reset()
        step_count = 0

        while not game.done and step_count < max_steps:
            action = agent.get_action(game)
            game.step(action)
            step_count += 1

        scores.append(game.score)
        steps.append(step_count)

    return np.array(scores), np.array(steps)


def print_table(results):
    col_width = 14
    metric_width = 18

    header = f"{'Metric':<{metric_width}} {'Non-DL Agent':>{col_width}} {'DL Agent':>{col_width}}"
    divider = '-' * len(header)

    print()
    print(divider)
    print(header)
    print(divider)
    for metric, non_dl_val, dl_val in results:
        print(f"{metric:<{metric_width}} {non_dl_val:>{col_width}} {dl_val:>{col_width}}")
    print(divider)
    print()


if __name__ == '__main__':
    non_dl_scores, non_dl_steps = evaluate(NonDLAgent())
    dl_scores, dl_steps = evaluate(DLAgent())

    results = [
        ('Avg Score',f'{non_dl_scores.mean():.2f}',f'{dl_scores.mean():.2f}'),
        ('Avg Steps',f'{non_dl_steps.mean():.1f}',f'{dl_steps.mean():.1f}'),
        ('Score Std Dev',f'{non_dl_scores.std():.2f}',f'{dl_scores.std():.2f}'),
        ('Max Score',f'{non_dl_scores.max()}',f'{dl_scores.max()}'),
    ]

    print_table(results)
