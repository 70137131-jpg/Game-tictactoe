import argparse
import os
import pickle
import random
from typing import Tuple

from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.game import getStateKey, best_move_minimax
from tictactoe.teacher import Teacher


def _legal_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                moves.append((i, j))
    return moves


def _winner(board):
    # returns 'X', 'O', 'D' for draw, or None if ongoing
    lines = [
        # rows
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        # cols
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        # diagonals
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    for line in lines:
        a, b, c = line
        v1, v2, v3 = board[a[0]][a[1]], board[b[0]][b[1]], board[c[0]][c[1]]
        if v1 != '-' and v1 == v2 == v3:
            return v1
    if any(board[i][j] == '-' for i in range(3) for j in range(3)):
        return None
    return 'D'


def _agent_greedy_move(agent, board) -> Tuple[int, int]:
    """Return the agent's greedy (eps=0) move without learning side-effects."""
    state = getStateKey(board)
    possible = _legal_moves(board)
    # Choose action that maximizes Q(a, s). Break ties randomly.
    values = [agent.Q[a][state] for a in possible]
    max_val = max(values)
    best_idxs = [i for i, v in enumerate(values) if v == max_val]
    choice_idx = random.choice(best_idxs)
    return possible[choice_idx]


def evaluate_agent(
    agent_path: str = 'q_agent.pkl',
    agent_type: str = 'q',
    games: int = 1000,
    opponent: str = 'minimax',  # 'minimax' | 'random' | 'teacher'
    teacher_ability: float = 0.9,
    verbose: bool = False,
):
    """
    Evaluate a trained agent by playing games without learning updates.

    - Agent plays as 'O'. Opponent plays as 'X' and moves first.
    - Greedy policy is used (no exploration, no Q updates).

    Returns (wins, draws, losses).
    """
    # Load agent
    if not os.path.isfile(agent_path):
        raise FileNotFoundError(f"Agent not found at {agent_path}")
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)

    # Build opponent policy
    if opponent == 'random':
        def opp_move(board):
            return random.choice(_legal_moves(board))
    elif opponent == 'teacher':
        teacher = Teacher(level=teacher_ability)
        def opp_move(board):
            return teacher.makeMove(board)
    else:  # minimax (default)
        def opp_move(board):
            return best_move_minimax(board, key='X')

    wins = draws = losses = 0

    for g in range(games):
        board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
        turn = 'X'  # opponent starts as X
        if verbose:
            print(f"\nGame {g+1}")
        while True:
            if turn == 'X':
                i, j = opp_move(board)
                board[i][j] = 'X'
                term = _winner(board)
                if term is not None:
                    if term == 'X':
                        losses += 1
                    elif term == 'O':
                        wins += 1
                    else:
                        draws += 1
                    break
                turn = 'O'
            else:  # agent 'O'
                i, j = _agent_greedy_move(agent, board)
                board[i][j] = 'O'
                term = _winner(board)
                if term is not None:
                    if term == 'X':
                        losses += 1
                    elif term == 'O':
                        wins += 1
                    else:
                        draws += 1
                    break
                turn = 'X'

    return wins, draws, losses


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained Tic-Tac-Toe RL agent.')
    parser.add_argument('--path', type=str, default='q_agent.pkl', help='Path to agent pickle')
    parser.add_argument('--agent', type=str, default='q', choices=['q', 's'], help='Agent type (for info only)')
    parser.add_argument('--games', type=int, default=1000, help='Number of evaluation games')
    parser.add_argument('--opponent', type=str, default='minimax', choices=['minimax', 'random', 'teacher'], help='Opponent policy')
    parser.add_argument('--teacher-ability', type=float, default=0.9, help='Teacher ability if opponent=teacher')
    parser.add_argument('--verbose', action='store_true', help='Print per-game progress')
    args = parser.parse_args()

    w, d, l = evaluate_agent(
        agent_path=args.path,
        agent_type=args.agent,
        games=args.games,
        opponent=args.opponent,
        teacher_ability=args.teacher_ability,
        verbose=args.verbose,
    )
    total = max(1, w + d + l)
    print('\nEvaluation results:')
    print(f'Games:   {total}')
    print(f'Wins:    {w} ({w/total*100:.1f}%)')
    print(f'Draws:   {d} ({d/total*100:.1f}%)')
    print(f'Losses:  {l} ({l/total*100:.1f}%)')


if __name__ == '__main__':
    main()

