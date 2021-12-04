from collections import namedtuple

from board import Player, Point

def compute_game_result(game_state):
    num_b, num_w = 0, 0
    for r in range(1, 8):
        for c in range(1, 8):
            player = game_state.board._grid.get(Point(r, c))
            if player is None: 
                if game_state.add_player is None: continue
                elif game_state.add_player == Player.black: num_b += 1
                elif game_state.add_player == Player.white: num_w += 1
            elif player == Player.black: num_b += 1
            elif player == Player.white: num_w += 1
    if num_b > num_w: return Player.black
    else: return Player.white

def get_black_more_num(game_state):
    num_b, num_w = 0, 0
    for r in range(1, 8):
        for c in range(1, 8):
            player = game_state.board._grid.get(Point(r, c))
            if player is None: continue
            elif player == Player.black: num_b += 1
            elif player == Player.white: num_w += 1
    return num_b - num_w