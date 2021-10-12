""" Final code implements minimax with alpha-beta pruning for board game Othello."""

import copy
import sys
import numpy as np

NUM_COLS = 8
# With these constant values for players, flipping ownership is just a sign change
WHITE = 1
NOBODY = 0
BLACK = -1

TIE = 2

WIN_VAL = 100
WHITE_TO_PLAY = True
DEMO_SEARCH_DEPTH = 5

# We'll sometimes iterate over this to look in all 8 directions from a particular square.
# The values are the "delta" differences in row, col from the original square.
# (Hence no (0,0), which would be the same square.)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def read_boardstring(boardstring):
    """Converts string representation of board to 2D numpy int array"""
    board = np.zeros((NUM_COLS, NUM_COLS))
    board_chars = {
        'W': WHITE,
        'B': BLACK,
        '-': NOBODY
    }
    row = 0
    for line in boardstring.splitlines():
        for col in range(NUM_COLS):
            board[row][col] = board_chars.get(line[col], NOBODY) # quietly ignore bad chars
        row += 1
    return board

def find_winner(board):
    """Return identity of winner, assuming game is over.

    Args:
        board (numpy 2D int array):  The othello board, with WHITE/BLACK/NOBODY in spaces

    Returns:
        int constant:  WHITE, BLACK, or TIE.
    """
    # Slick counting of values:  np.count_nonzero counts vals > 0, so pass in
    # board == WHITE to get 1 or 0 in the right spots
    white_count = np.count_nonzero(board == WHITE)
    black_count = np.count_nonzero(board == BLACK)
    if white_count > black_count:
        return WHITE
    if white_count < black_count:
        return BLACK
    return TIE

def generate_legal_moves(board, white_turn):
    """Returns a list of (row, col) tuples representing places to move.

    Args:
        board (numpy 2D int array):  The othello board
        white_turn (bool):  True if it's white's turn to play
    """

    legal_moves = []
    for row in range(NUM_COLS):
        for col in range(NUM_COLS):
            if board[row][col] != NOBODY:
                continue   # Occupied, so not legal for a move
            # Legal moves must capture something
            if can_capture(board, row, col, white_turn):
                legal_moves.append((row, col))
    return legal_moves

def can_capture(board, row, col, white_turn):
    """ Helper that checks capture in each of 8 directions.

    Args:
        board (numpy 2D int array) - othello board
        row (int) - row of move
        col (int) - col of move
        white_turn (bool) - True if it's white's turn
    Returns:
        True if capture is possible in any direction
    """
    for r_delta, c_delta in DIRECTIONS:
        if captures_in_dir(board, row, r_delta, col, c_delta, white_turn):
            return True
    return False

def captures_in_dir(board, row, row_delta, col, col_delta, white_turn):
    """Returns True iff capture possible in direction described by delta parameters

    Args:
        board (numpy 2D int array) - othello board
        row (int) - row of original move
        row_delta (int) - modification needed to row to move in direction of capture
        col (int) - col of original move
        col_delta (int) - modification needed to col to move in direction of capture
        white_turn (bool) - True iff it's white's turn
    """

    # Can't capture if headed off the board
    if (row+row_delta < 0) or (row+row_delta >= NUM_COLS):
        return False
    if (col+col_delta < 0) or (col+col_delta >= NUM_COLS):
        return False

    # Can't capture if piece in that direction is not of appropriate color or missing
    enemy_color = BLACK if white_turn else WHITE
    if board[row+row_delta][col+col_delta] != enemy_color:
        return False

    # At least one enemy piece in this direction, so just need to scan until we
    # find a friendly piece (return True) or hit an empty spot or edge of board
    # (return False)
    friendly_color = WHITE if white_turn else BLACK
    scan_row = row + 2*row_delta # row of first scan position
    scan_col = col + 2*col_delta # col of first scan position
    while 0 <= scan_row < NUM_COLS and 0 <= scan_col < NUM_COLS:
        if board[scan_row][scan_col] == NOBODY:
            return False
        if board[scan_row][scan_col] == friendly_color:
            return True
        scan_row += row_delta
        scan_col += col_delta
    return False

def capture(board, row, col, white_turn):
    """Destructively change a board to represent capturing a piece with a move at (row,col).

    The board's already a copy made specifically for the purpose of representing this move,
    so there's no point in copying it again.  We'll return the board anyway.

    Args:
        board (numpy 2D int array) - The Othello board - will be destructively modified
        row (int) - row of move
        col (int) - col of move
        white_turn (bool) - True iff it's white's turn
    Returns:
        The board, though this isn't necessary since it's destructively modified
    """

    # Check in each direction as to whether flips can happen -- if they can, start flipping
    enemy_color = BLACK if white_turn else WHITE
    for row_delta, col_delta in DIRECTIONS:
        if captures_in_dir(board, row, row_delta, col, col_delta, white_turn):
            flip_row = row + row_delta
            flip_col = col + col_delta
            while board[flip_row][flip_col] == enemy_color:
                board[flip_row][flip_col] = -enemy_color
                flip_row += row_delta
                flip_col += col_delta
    return board

def play_move(board, move, white_turn):
    """Handles the logic of putting down a new piece and flipping captured pieces.

    The board that is returned is a copy, so this is appropriate to use for search.

    Args:
        board (numpy 2D int array):  The othello board
        move ((int,int)):  A (row, col) pair for the move
        white_turn:  True iff it's white's turn
    Returns:
        board (numpy 2D int array)
    """
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = WHITE if white_turn else BLACK
    new_board = capture(new_board, move[0], move[1], white_turn)
    return new_board

def evaluation_function(board):
    """Returns the difference in piece count - an easy evaluation function for minimax"""

    # We could count with loops, but we're feeling fancy
    return np.count_nonzero(board == WHITE) - np.count_nonzero(board == BLACK)

def check_game_over(board):
    """Returns the current winner of the board - WHITE, BLACK, TIE, NOBODY"""

    # It's not over if either player still has legal moves
    white_legal_moves = generate_legal_moves(board, True)
    if white_legal_moves:  # Python idiom for checking for empty list
        return NOBODY
    black_legal_moves = generate_legal_moves(board, False)
    if black_legal_moves:
        return NOBODY
    # I guess the game's over
    return find_winner(board)

def find_max_score(maxScore, board, white_turn, search_depth, alpha, beta):
    legal_moves = generate_legal_moves(board, True)
    if len(legal_moves) == 0:
        maxScore = minimax_value(board, False, search_depth, alpha, beta)
    else:
        for m in legal_moves:
            updated_board = play_move(board, m, True)
            score = minimax_value(updated_board, False, search_depth - 1, alpha, beta)
            if score > maxScore:
                maxScore = score
            if score > alpha:
                alpha = score
            if beta <= alpha:
                break
    return maxScore

def find_min_score(minScore, board, white_turn, search_depth, alpha, beta):
    legal_moves = generate_legal_moves(board, False)
    if len(legal_moves) == 0:
        minScore = minimax_value(board, True, search_depth, alpha, beta)
        
    else:
        for m in legal_moves:
            new_board = play_move(board, m, False) # Get the new board state
            score = minimax_value(new_board, True, search_depth - 1, alpha, beta)
            minScore = min(minScore, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
    return minScore


def minimax_value(board, white_turn, search_depth, alpha, beta):
    """Return the value of the board, up to the maximum search depth.

    Assumes white is MAX and black is MIN (even if black uses this function).

    Args:
        board (numpy 2D int array) - The othello board
        white_turn (bool) - True iff white would get to play next on the given board
        search_depth (int) - the search depth remaining, decremented for recursive calls
        alpha (int or float) - Lower bound on the value:  MAX ancestor forbids lower results
        beta (int or float) - Upper bound on the value:  MIN ancestor forbids larger results
    """
    # Decide the current game condition.
    if search_depth == 0:
        return np.count_nonzero(board == WHITE) - np.count_nonzero(board == BLACK)
    game_condition = check_game_over(board)
    if game_condition == TIE:
        return 0
    elif game_condition == WHITE:
        return WIN_VAL
    elif game_condition == BLACK:
        return (WIN_VAL * -1)
    
    # Otherwise, if there is no discernable game state, find the difference based on whose turn it is.
    # White's turn difference.
    if white_turn:
        max = (sys.maxsize * -1)
        maxScore = find_max_score(max, board, white_turn, search_depth, alpha, beta)
        return maxScore
    # Black's turn difference.
    else:
        min = sys.maxsize
        minScore = find_min_score(min, board, white_turn, search_depth, alpha, beta)
        return minScore

def print_board(board):
    """ Print board (and return None), for interactive mode"""
    printable = {
        -1: "B",
        0: "-",
        1: "W"
    }
    for row in range(NUM_COLS):
        line = ""
        for col in range(NUM_COLS):
            line += printable[board[row][col]]
        print(line)

def play():
    """Interactive play, for demo purposes.  Assume AI is white and goes first."""
    board = starting_board()
    while check_game_over(board) == NOBODY:
        # White turn (AI)
        legal_moves = generate_legal_moves(board, True)
        if legal_moves:  # (list is non-empty)
            print("Thinking...")
            best_val = float("-inf")
            best_move = None
            for move in legal_moves:
                new_board = play_move(board, move, True)
                move_val = minimax_value(new_board, True, DEMO_SEARCH_DEPTH, \
                                         float("-inf"), float("inf"))
                if move_val > best_val:
                    best_move = move
                    best_val = move_val
            board = play_move(board, best_move, True)
            print_board(board)
            print("")
        else:
            print("White has no legal moves; skipping turn...")

        legal_moves = generate_legal_moves(board, False)
        if legal_moves:
            player_move = get_player_move(board, legal_moves)
            board = play_move(board, player_move, False)
            print_board(board)
        else:
            print("Black has no legal moves; skipping turn...")
    winner = find_winner(board)
    if winner == WHITE:
        print("White won!")
    elif winner == BLACK:
        print("Black won!")
    else:
        print("Tie!")

def starting_board():
    """Returns a board with the traditional starting positions in Othello."""
    board = np.zeros((NUM_COLS, NUM_COLS))
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board

def get_player_move(board, legal_moves):
    """Print board with numbers for the legal move spaces, then get player choice of move

    Args:
        board (numpy 2D int array):  The Othello board.
        legal_moves (list of (int,int)):  List of legal (row,col) moves for human player
    Returns:
        (int, int) representation of the human player's choice
    """
    for row in range(NUM_COLS):
        line = ""
        for col in range(NUM_COLS):
            if board[row][col] == WHITE:
                line += "W"
            elif board[row][col] == BLACK:
                line += "B"
            else:
                if (row, col) in legal_moves:
                    line += str(legal_moves.index((row, col)))
                else:
                    line += "-"
        print(line)
    while True:
        # Bounce around this loop until a valid integer is received
        choice = input("Which move do you want to play? [0-" + str(len(legal_moves)-1) + "]")
        try:
            move_num = int(choice)
            if 0 <= move_num < len(legal_moves):
                return legal_moves[move_num]
            print("That wasn't one of the options.")
        except ValueError:
            print("Please enter an integer as your move choice.")

def eval_at_depth(boardstring, depth):
    """Returns the value of the board up to the given search depth.

    Args:
        boardstring -- String representation of the board to evaluate.
        depth --- Search depth limit.
    """
    board = read_boardstring(boardstring)
    return minimax_value(board, WHITE_TO_PLAY, depth, float("-inf"), float("inf"))
