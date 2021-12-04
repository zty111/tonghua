from keras.saving.save import load_model
from board import GameState, Player
from encoder import Encoder
from agent import Agent
import scoring
from experience import ExperienceCollector, combine_experience
import numpy as np
from tensorflow.keras.optimizers import SGD
from tiaocan import num, rand_num, mcts_num, show, bot_name

def simulate_game(black_agent, black_collector, white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game()
    agents = {
        Player.black: black_agent,
        Player.white: white_agent
    }

    black_collector.begin_episode()
    white_collector.begin_episode()

    tun = 0
    while not game.is_over():
        if show:
            game.print()
        tun += 1
        next_move = agents[game.next_player].select_move(game, tun <= 30)
        if show:
            if next_move.is_pass: print("Pass!")
            else: print(next_move.point, next_move.direction)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    if game_result == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)

def get_exp(id):
    model = load_model(bot_name)
    encoder = Encoder()

    black_agent = Agent(model, encoder, rounds_per_move = mcts_num, c = 2.0)
    white_agent = Agent(model, encoder, rounds_per_move = mcts_num, c = 2.0)
    c1 = ExperienceCollector()
    c2 = ExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)
    
    for i in range(num):
        print("process  {id} round {i} : ", end = '')
        simulate_game(black_agent, c1, white_agent, c2)

    exp = combine_experience([c1, c2])
    exp.serialize(f'exp{id}.h5')


def train(experience, learning_rate, batch_size):
    model = load_model(bot_name)

    num_examples = experience.states.shape[0]

    model_input = experience.states

    visit_sums = np.sum(experience.visit_counts, axis = 1).reshape((num_examples, 1))
    action_target = experience.visit_counts / visit_sums

    value_target = experience.rewards

    model.compile(
        SGD(learning_rate = learning_rate),
        loss=['categorical_crossentropy', 'mse']
    )
    model.fit(
        model_input, [action_target, value_target],
        batch_size = batch_size
    )

    model.save(bot_name)