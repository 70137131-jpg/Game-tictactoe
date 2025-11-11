import random
from functools import lru_cache


class Game:
    """ The game class. New instance created for each new game. """
    def __init__(self, agent, teacher=None):
        self.agent = agent
        self.teacher = teacher
        # initialize the game board
        self.board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

    def playerMove(self):
        """
        Querry player for a move and update the board accordingly.
        """
        if self.teacher is not None:
            action = self.teacher.makeMove(self.board)
            self.board[action[0]][action[1]] = 'X'
        else:
            printBoard(self.board)
            while True:
                move = input("Your move! Please select a row and column from 0-2 "
                             "in the format row,col: ")
                print('\n')
                try:
                    row, col = int(move[0]), int(move[2])
                except ValueError:
                    print("INVALID INPUT! Please use the correct format.")
                    continue
                if row not in range(3) or col not in range(3) or not self.board[row][col] == '-':
                    print("INVALID MOVE! Choose again.")
                    continue
                self.board[row][col] = 'X'
                break

    def agentMove(self, action):
        """
        Update board according to agent's move.
        """
        self.board[action[0]][action[1]] = 'O'

    def checkForWin(self, key):
        """
        Check to see whether the player/agent with token 'key' has won.
        Returns a boolean holding truth value.

        Parameters
        ----------
        key : string
            token of most recent player. Either 'O' or 'X'
        """
        # check for player win on diagonals
        a = [self.board[0][0], self.board[1][1], self.board[2][2]]
        b = [self.board[0][2], self.board[1][1], self.board[2][0]]
        if a.count(key) == 3 or b.count(key) == 3:
            return True
        # check for player win on rows/columns
        for i in range(3):
            col = [self.board[0][i], self.board[1][i], self.board[2][i]]
            row = [self.board[i][0], self.board[i][1], self.board[i][2]]
            if col.count(key) == 3 or row.count(key) == 3:
                return True
        return False

    def checkForDraw(self):
        """
        Check to see whether the game has ended in a draw. Returns a
        boolean holding truth value.
        """
        draw = True
        for row in self.board:
            for elt in row:
                if elt == '-':
                    draw = False
        return draw

    def checkForEnd(self, key):
        """
        Checks if player/agent with token 'key' has ended the game. Returns -1
        if the game is still going, 0 if it is a draw, and 1 if the player/agent
        has won.

        Parameters
        ----------
        key : string
            token of most recent player. Either 'O' or 'X'
        """
        if self.checkForWin(key):
            if self.teacher is None:
                printBoard(self.board)
                if key == 'X':
                    print("Player wins!")
                else:
                    print("RL agent wins!")
            return 1
        elif self.checkForDraw():
            if self.teacher is None:
                printBoard(self.board)
                print("It's a draw!")
            return 0
        return -1

    def playGame(self, player_first):
        """ 
        Begin the tic-tac-toe game loop. 

        Parameters
        ----------
        player_first : boolean
            Whether or not the player will move first. If False, the
            agent goes first.

        """
        # Initialize the agent's state and action
        if player_first:
            self.playerMove()
        prev_state = getStateKey(self.board)
        prev_action = self.agent.get_action(prev_state)

        # iterate until game is over
        while True:
            # execute oldAction, observe reward and state
            self.agentMove(prev_action)
            check = self.checkForEnd('O')
            if not check == -1:
                # game is over. +1 reward if win, 0 if draw
                reward = check
                break
            self.playerMove()
            check = self.checkForEnd('X')
            if not check == -1:
                # game is over. -1 reward if lose, 0 if draw
                reward = -1*check
                break
            else:
                # game continues. 0 reward
                reward = 0
            new_state = getStateKey(self.board)

            # determine new action (epsilon-greedy)
            new_action = self.agent.get_action(new_state)
            # update Q-values
            self.agent.update(prev_state, new_state, prev_action, new_action, reward)
            # reset "previous" values
            prev_state = new_state
            prev_action = new_action
            # append reward

        # Game over. Perform final update
        self.agent.update(prev_state, None, prev_action, None, reward)

    def start(self):
        """
        Function to determine who moves first, and subsequently, start the game.
        If a teacher is employed, first mover is selected at random.
        If a human is playing, the human is asked whether he/she would
        like to move fist. 
        """
        if self.teacher is not None:
            # During teaching, chose who goes first randomly with equal probability
            if random.random() < 0.5:
                self.playGame(player_first=False)
            else:
                self.playGame(player_first=True)
        else:
            while True:
                response = input("Would you like to go first? [y/n]: ")
                print('')
                if response == 'n' or response == 'no':
                    self.playGame(player_first=False)
                    break
                elif response == 'y' or response == 'yes':
                    self.playGame(player_first=True)
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

def printBoard(board):
    """
    Prints the game board as text output to the terminal.

    Parameters
    ----------
    board : list of lists
        the current game board
    """
    print('    0   1   2\n')
    for i, row in enumerate(board):
        print('%i   ' % i, end='')
        for elt in row:
            print('%s   ' % elt, end='')
        print('\n')

def getStateKey(board):
    """
    Converts 2D list representing the board state into a string key
    for that state. Keys are used for Q-value hashing.

    Parameters
    ----------
    board : list of lists
        the current game board
    """
    key = ''
    for row in board:
        for elt in row:
            key += elt
    return key






# ---------- Minimax baseline with transposition cache ----------

def _flatten(board):
    return ''.join(board[i][j] for i in range(3) for j in range(3))


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


@lru_cache(maxsize=None)
def _minimax_cached(flat_board, player):
    # player: 'X' or 'O' — returns (score, best_move)
    board = [[flat_board[r * 3 + c] for c in range(3)] for r in range(3)]
    term = _winner(board)
    if term == 'X':
        return 1, None
    if term == 'O':
        return -1, None
    if term == 'D':
        return 0, None

    moves = _legal_moves(board)
    if player == 'X':
        best_score = -2
        best_mv = None
        for (i, j) in moves:
            board[i][j] = 'X'
            sc, _ = _minimax_cached(_flatten(board), 'O')
            board[i][j] = '-'
            if sc > best_score:
                best_score, best_mv = sc, (i, j)
                if best_score == 1:
                    break
        return best_score, best_mv
    else:
        best_score = 2
        best_mv = None
        for (i, j) in moves:
            board[i][j] = 'O'
            sc, _ = _minimax_cached(_flatten(board), 'X')
            board[i][j] = '-'
            if sc < best_score:
                best_score, best_mv = sc, (i, j)
                if best_score == -1:
                    break
        return best_score, best_mv


def best_move_minimax(board, key='X'):
    """Return optimal move (i, j) for given key using cached minimax."""
    _, mv = _minimax_cached(_flatten(board), key)
    # fallback to first legal if somehow None
    return mv if mv is not None else (_legal_moves(board)[0])
