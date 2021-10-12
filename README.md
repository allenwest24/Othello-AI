# Othello-AI

This AI contains a minimax module for the board game Othello, also known as Reversi. The main module calculates the value for a given board position. 

The rules of Othello are as follows:
- The two player colors are white and black. The white player goes first.
- You capture an opponent’s pieces when they lie in a straight line between a piece you already had on the board and a piece you just played. (A straight line is left-right, up-down, or a 45 degree diagonal.)
- You can only play a piece that would capture at least one piece. If you have no legal moves, the turn is passed.
- The game is over when neither player has any legal moves left. Whoever controls the most pieces on the board at that point wins.

Something that is slightly unusual about Othello for minimax is the fact that a turn might be skipped if a player has no legal plays. The minimax calculations take this into account. 

The AI is always presumed to be white for this implementation; if you try the demo mode, you as the human will be playing black.
