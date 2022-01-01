from keras.saving.save import load_model
from board import GameState, Player
from encoder import Encoder
from agent import Agent
import scoring
from board import Move, Point
from tiaocan import bot_name

class My():
    def select_move(self, game_state):
        print("请输入点坐标和方向（或弃权）：")
        x, y, d = input().split(' ')
        x, y, d = int(x), int(y), int(d)
        move = Move(Point(x, y), d)
        if game_state.is_valid_move(move): return move
        else: return Move.pass_turn()
        

def simulate_game(black_agent, white_agent):
    print('Starting the game!')
    game = GameState.new_game()
    agents = {
        Player.black: black_agent,
        Player.white: white_agent
    }

    while not game.is_over():
        game.print() 
        if game.next_player == Player.black: next_move = agents[game.next_player].greedy_move(game)
        else: next_move = agents[game.next_player].select_move(game, False)
        if next_move.is_pass: print("Pass!")
        else: print(next_move.point, next_move.direction)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    if game_result == Player.black:
        print("You win!")
    else:
        print("Bot Zero win!")

encoder = Encoder()

model = load_model(bot_name)

black_agent = Agent(model, encoder, rounds_per_move = 160, c = 2.0)
white_agent = Agent(model, encoder, rounds_per_move = 160, c = 2.0)

print()
print("欢迎对局!")
print("输入为3个以空格隔开的数字")
print("前2个为点坐标(1~7)")
print("第3个为方向(0~23)，具体如下")
print("0\t1\t2\t3\t4")
print("5\t6\t7\t8\t9")
print("10\t11\t棋子\t12\t13")
print("14\t15\t16\t17\t18")
print("19\t20\t21\t22\t23")
print("不要输错哦！")
simulate_game(black_agent, white_agent)