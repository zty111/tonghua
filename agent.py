from functools import total_ordering
import numpy as np
import time
from multiprocessing.dummy import Pool
from tiaocan import thread_num
from board import Move
from scoring import get_more_num


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class TreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {}

    def moves(self):
        return self.branches.keys()
    
    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

class Agent():
    def __init__(self, model, encoder, rounds_per_move = 1600, c = 2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = rounds_per_move
        self.c = c

    def greedy_move(self, game_state):
        root = self.create_node(game_state)
        move_list = list(root.moves())
        score = get_more_num(game_state)
        mx = -100
        for move in move_list:
            s = game_state.apply_move(move)
            d = get_more_num(s) - score
            if mx < d: mx, cmove = d, move
        return cmove

    def random_move(self, game_state):
        root = self.create_node(game_state)
        move_list = list(root.moves())
        l = len(move_list)
        if l == 1:
            return move_list[0]
        return move_list[np.random.randint(0, l - 1)]

    def select_move(self, game_state, temperature):
        root = self.create_node(game_state)
        pool = Pool(thread_num)

        def select_one(i):
            node = root

            next_move = self.select_branch_root(node)
            root.branches[next_move].prior -= 1
            first_move = next_move
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, move = next_move, parent = node)

            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

            root.branches[first_move].prior += 1

        #st = time.time()
        pool.map(select_one, range(self.num_rounds))
        #print(time.time() - st)

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(root_state_tensor, visit_counts)
            reverse_state_tensor = self.encoder.reverse_encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_reverse_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(reverse_state_tensor, visit_counts)

        if temperature:
            p = []
            all = 0
            for move in root.moves():
                p.append(root.visit_count(move))
                all += root.visit_count(move)
            for i in range(len(p)): p[i] /= all
            return np.random.choice(list(root.moves()), p = p)
        else:
            return max(root.moves(), key = root.visit_count)

    def set_collector(self, collector):
        self.collector = collector  

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return (q + self.c * p * np.sqrt(total_n) / (n + 1)).any()

        return max(node.moves(), key = score_branch)

    def select_branch_root(self, node):
        total_n = node.total_visit_count
        move_list = list(node.moves())

        def score_branch(id):
            move = move_list[id]
            q = node.expected_value(move)
            p = node.prior(move) * 0.7 + dir[id] * 0.3
            n = node.visit_count(move)
            return (q + self.c * p * np.sqrt(total_n) / (n + 1))
        tmp = np.full(len(move_list), 0.03)
        dir = np.random.dirichlet(tmp)
        id = range(len(move_list))

        return move_list[max(id, key = score_branch)]

    def create_node(self, game_state, move = None, parent = None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        #st = time.time()
        priors, values = self.model(model_input, training = False)
        priors = np.array(priors)
        values = np.array(values)

        #print(time.time() - st)
        priors = priors[0]
        value = values[0][0]
        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(priors)
        }
        move_priors[Move.pass_turn()] = -10
        new_node = TreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node