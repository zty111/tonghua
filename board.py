class Move():
    def __init__(self, point = None, direction = 0, is_pass = False):
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.direction = direction

    @classmethod
    def play(cls, point, direction):
        return Move(point = point, direction = direction)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass = True)

    def __str__(self):
        if self.is_pass:
            return 'pass'
        return '(r %d, c %d, d %d)' % (self.point.row, self.point.col, self.direction)

    def __hash__(self):
        return hash((
            self.is_play,
            self.is_pass,
            self.point,
            self.direction
        ))

    def __eq__(self, other):
        return(
            self.is_play, 
            self.is_pass,
            self.point,
            self.direction
        ) == (
            other.is_play,
            other.is_pass,
            other.point,
            other.direction
        )

import enum
from collections import namedtuple

class Player(enum.Enum):
    black = 1
    white = 2
    
    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

class Point(namedtuple('Point', 'row, col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col - 1),
            Point(self.row - 1, self.col),
            Point(self.row - 1, self.col + 1),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
            Point(self.row + 1, self.col - 1),
            Point(self.row + 1, self.col),
            Point(self.row + 1, self.col + 1)
        ]
    
    def __deepcopy__(self, memodict = {}):
        return self

import copy
dr = [-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
dc = [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
d_8 = [6, 7, 8, 11, 12, 15, 16, 17]
class GameState:
    def __init__(self, board, next_player, move):
        self.board = board
        self.next_player = next_player
        self.last_move = move
        self.add_player = None

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            new_point = Point(move.point.row + dr[move.direction], move.point.col + dc[move.direction])
            next_board.place_stone(self.next_player, new_point)
            if move.direction not in d_8:
                next_board.remove_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, move)

    @classmethod
    def new_game(cls):
        board = Board()
        return GameState(board, Player.black, None)

    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_pass:
            self.add_player = self.next_player
            return True
        return False

    def get_color(self, point):
        player = self.board._grid.get(point)
        if player is None:
            return None
        return player

    def is_valid_move(self, move):
        #if self.is_over(): return False
        if move.is_pass: return True
        r = move.point.row + dr[move.direction]
        c = move.point.col + dc[move.direction]
        if r >= 1 and r <= 7 and c >= 1 and c <= 7:
            if self.board._grid.get(move.point) == self.next_player and self.board._grid.get(Point(r, c)) is None:
                return True
        return False

    def print(self):
        print('')
        for r in range(1, 8):
            for c in range(1, 8):
                bot = self.board._grid.get(Point(r, c))
                if bot is None: print(0, end = '\t')
                else: print(bot.value, end = '\t')
            print('')

class Board:
    def __init__(self):
        self._grid = {}
        self._grid[Point(1, 1)], self._grid[Point(7, 7)] = Player.black, Player.black
        self._grid[Point(1, 7)], self._grid[Point(7, 1)] = Player.white, Player.white
    
    def remove_stone(self, player, point):
        self._grid[point] = None

    def place_stone(self, player, point):
        self._grid[point] = player
        for neighbor in point.neighbors():
            bot = self._grid.get(neighbor)
            if bot is None: continue
            self._grid[neighbor] = player
