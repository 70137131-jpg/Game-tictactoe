from flask import Flask, request, jsonify, render_template
import os
import pickle
from typing import List, Tuple, Optional

from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.game import Game, getStateKey
from tictactoe.teacher import Teacher


# Configuration
AGENT_TYPE = os.environ.get('AGENT_TYPE', 'q')  # 'q' or 's'
AGENT_PATH = os.environ.get('AGENT_PATH', 'q_agent.pkl' if AGENT_TYPE == 'q' else 'sarsa_agent.pkl')
EPSILON = float(os.environ.get('EPSILON', '0.05'))
ALPHA = float(os.environ.get('ALPHA', '0.5'))
GAMMA = float(os.environ.get('GAMMA', '0.9'))


app = Flask(__name__)


def load_or_init_agent():
	if os.path.isfile(AGENT_PATH):
		with open(AGENT_PATH, 'rb') as f:
			agent = pickle.load(f)
		# allow overriding epsilon for play mode via env
		agent.eps = EPSILON  # ⚠️ Resets exploration rate
		return agent
	# init new
	if AGENT_TYPE == 'q':
		return Qlearner(alpha=ALPHA, gamma=GAMMA, eps=EPSILON)
	return SARSAlearner(alpha=ALPHA, gamma=GAMMA, eps=EPSILON)


def save_agent(agent) -> None:
	if os.path.isfile(AGENT_PATH):
		os.remove(AGENT_PATH)
	with open(AGENT_PATH, 'wb') as f:
		pickle.dump(agent, f)


def validate_board(board: List[List[str]]) -> Optional[str]:
	if not isinstance(board, list) or len(board) != 3:
		return 'Board must be a 3x3 list.'
	for row in board:
		if not isinstance(row, list) or len(row) != 3:
			return 'Board must be a 3x3 list.'
		for v in row:
			if v not in ['X', 'O', '-']:
				return "Board cells must be one of 'X', 'O', '-'."
	return None


def apply_player_move(board: List[List[str]], move: Tuple[int, int]) -> bool:
	i, j = move
	if i not in range(3) or j not in range(3):
		return False
	if board[i][j] != '-':
		return False
	board[i][j] = 'X'
	return True


@app.get('/')
def index():
	return render_template('index.html')


@app.post('/api/move')
def api_move():
	data = request.get_json(silent=True)
	if data is None:
		return jsonify({'error': "Invalid JSON or Content-Type header is not 'application/json'"}), 400
	board = data.get('board')
	player_move = data.get('playerMove')  # [i, j]
	if board is None or player_move is None:
		return jsonify({'error': "Required: 'board' and 'playerMove'"}), 400

	err = validate_board(board)
	if err:
		return jsonify({'error': err}), 400
	if not isinstance(player_move, list) or len(player_move) != 2:
		return jsonify({'error': "'playerMove' must be [i, j]"}), 400

	agent = load_or_init_agent()
	game = Game(agent)
	game.board = [row[:] for row in board]

	# Apply player's move (X)
	if not apply_player_move(game.board, (int(player_move[0]), int(player_move[1]))):
		return jsonify({'error': 'Invalid move'}), 400

	# Check if player ended the game
	player_end = game.checkForEnd('X')
	if player_end != -1:
		return jsonify({
			'board': game.board,
			'status': 'ended',
			'result': 'player_win' if player_end == 1 else 'draw'
		})

	# Agent move (O) — act greedily (no exploration)
	state = getStateKey(game.board)
	original_eps = getattr(agent, 'eps', 0.0)
	try:
		agent.eps = 0.0
		action = agent.get_action(state)
	finally:
		agent.eps = original_eps
	game.agentMove(action)

	# Determine outcome after agent move
	agent_end = game.checkForEnd('O')
	if agent_end != -1:
		return jsonify({
			'board': game.board,
			'status': 'ended',
			'result': 'agent_win' if agent_end == 1 else 'draw',
			'agentMove': list(action)
		})

	# No learning during play; only return board
	return jsonify({
		'board': game.board,
		'status': 'ongoing',
		'agentMove': list(action)
	})


@app.post('/api/reset')
def api_reset():
	# Just returns a clean board; agent persisted separately
	board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
	return jsonify({'board': board})


@app.post('/api/train')
def api_train():
	data = request.get_json(silent=True)
	if data is None:
		return jsonify({'error': "Invalid JSON or Content-Type header is not 'application/json'"}), 400
	episodes = int(data.get('episodes', 50000))  # Changed from 1000
	ability = float(data.get('teacherAbility', 0.9))

	agent = load_or_init_agent()
	teacher = Teacher(level=ability)
	games_played = 0

	# Add progress tracking
	print(f"\nStarting training for {episodes} episodes...")
	import time
	start_time = time.time()

	while games_played < episodes:
		game = Game(agent, teacher=teacher)
		game.start()
		games_played += 1

		# Progress updates every 10000 episodes
		if games_played % 10000 == 0:
			elapsed = time.time() - start_time
			print(f"Episode {games_played}/{episodes} | Time: {elapsed:.1f}s")

	save_agent(agent)
	total_time = time.time() - start_time
	print(f"\n✅ Training complete in {total_time:.1f} seconds!")
	print(f"Agent saved to: {AGENT_PATH}")

	return jsonify({'trainedEpisodes': episodes, 'rewards': agent.rewards[-episodes:]})


if __name__ == '__main__':
	# For local development
	app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=True)
