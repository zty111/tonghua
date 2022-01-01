import numpy as np
from board import Move, Player, Point

dr = [-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
dc = [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
d_8 = [6, 7, 8, 11, 12, 15, 16, 17]

class Encoder:
    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        if game_state.last_move is None or game_state.last_move.is_pass:
            pass
        else:
            p = game_state.last_move.point
            d = game_state.last_move.direction
            if d in d_8:
                board_tensor[4, int(p.row - 1), int(p.col - 1)] = 1
            board_tensor[3, int(p.row + dr[d] - 1), int(p.col + dc[d] - 1)] = 1
        for r in range(7):
            for c in range(7):
                p = Point(row = r + 1, col = c + 1)
                color = game_state.get_color(p)
                if color == next_player:
                    board_tensor[0, r, c] = 1
                elif color is None:
                    board_tensor[1, r, c] = 1
                elif color == next_player.other:
                    board_tensor[2, r, c] = 1
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

    def reverse_encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        if game_state.last_move is None or game_state.last_move.is_pass:
            pass
        else:
            p = game_state.last_move.point
            d = 23 - game_state.last_move.direction
            if d in d_8:
                board_tensor[4, int(7 - p.row), int(7 - p.col)] = 1
            board_tensor[3, int(7 - p.row + dr[d]), int(7 - p.col + dc[d])] = 1
        for r in range(7):
            for c in range(7):
                p = Point(row = 7 - r, col = 7 - c)
                color = game_state.get_color(p)
                if color == next_player:
                    board_tensor[0, r, c] = 1
                elif color is None:
                    board_tensor[1, r, c] = 1
                elif color == next_player.other:
                    board_tensor[2, r, c] = 1
        return board_tensor

    def decode_reverse_move_index(self, index):
        if index == 1176:
            return Move.pass_turn()
        direction = index % 24
        index = (index - direction) / 24
        row = index // 7
        col = index % 7
        return Move.play(Point(row = 7 - row, col = 7 - col), direction = 23 - direction)

    def num_moves(self):
        return 1177

    def shape(self):
        return 5, 7, 7
