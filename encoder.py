import numpy as np
from board import Move, Player, Point

class Encoder:
    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(7):
            for c in range(7):
                p = Point(row = r + 1, col = c + 1)
                color = game_state.get_color(p)
                if color == next_player:
                    board_tensor[0, r, c] = 1
                else:
                    board_tensor[0, r, c] = -1
        return board_tensor
    
    def encode_move(self, move):
        if move.is_play:
            return ((7 * (move.point.row - 1) + (move.point.col - 1)) * 24 + move.direction)
        elif move.is_pass:
            return 1176
        
    def decode_move_index(self, index):
        if index == 1176:
            return Move.pass_turn()
        direction = index % 24
        index = (index - direction) / 24
        row = index // 7
        col = index % 7
        return Move.play(Point(row = row + 1, col = col + 1), direction = direction)

    def num_moves(self):
        return 1177

    def shape(self):
        return 1, 7, 7
