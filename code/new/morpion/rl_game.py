
import tqdm
import numpy as np
import random
from collections import deque
import pickle5 as pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Supervisor :
    def __init__(self, game) :
        self.game = game

    def simulate(self, p1, p2, nb_round, verbose=False) :
        for battle_count in tqdm.tqdm(range(nb_round)) :
            current_game = self.game(p1, p2, verbose)
            p1_mem = []
            p2_mem = []
            turn_p1 = []
            turn_p2 = []
            for i in range(9) :
                if current_game.p1_turn :
                    turn_p1.append(current_game.board.board.copy())
                    move = current_game.next_moves()
                    turn_p1.append(move)
                    if not current_game.is_legal_move(move) :
                        turn_p1.append(current_game.board.board.copy())
                        turn_p1.append(True)
                        turn_p1.append(-1)
                        p1_mem.append(turn_p1)
                        p1.memorize(p1_mem)
                        break
                    else :
                        current_game.apply_move(move)
                        current_game.detect_end(current_game.board)
                else :
                    current_game.board.invert()
                    turn_p2.append(current_game.board.board.copy())
                    move = current_game.next_moves()
                    turn_p2.append(move)
                    if not current_game.is_legal_move(move) :
                        turn_p2.append(current_game.board.board.copy())
                        turn_p2.append(True)
                        turn_p2.append(-1)
                        p2_mem.append(turn_p2)
                        p2.memorize(p2_mem)
                        break
                    else :
                        current_game.apply_move(move)
                        current_game.detect_end(current_game.board)
                    current_game.board.invert()

                if i != 0 :
                    if current_game.p1_turn :
                        turn_p1.append(current_game.board.board.copy())
                        turn_p1.append(current_game.done)
                        if current_game.done :
                            if current_game.final_state == 1 :
                                turn_p1.append(1)
                            elif current_game.final_state == 0 :
                                turn_p1.append(0)
                            else :
                                turn_p1.append(-1)
                        else :
                            turn_p1.append(0)
                        p1_mem.append(turn_p1)
                        turn_p1 = []
                    else :
                        turn_p2.append(current_game.board.board.copy())
                        turn_p2.append(current_game.done)
                        if current_game.done :
                            if current_game.final_state == 2 :
                                turn_p2.append(1)
                            elif current_game.final_state == 0 :
                                turn_p2.append(0)
                            else :
                                turn_p2.append(-1)
                        else :
                            turn_p2.append(0)
                        p2_mem.append(turn_p2)
                        turn_p2 = []

                    if current_game.done :
                        if not current_game.p1_turn :
                            turn_p1.append(current_game.board.board.copy())
                            turn_p1.append(current_game.done)
                            if current_game.final_state == 1 :
                                turn_p1.append(1)
                            elif current_game.final_state == 0 :
                                turn_p1.append(0)
                            else :
                                turn_p1.append(-1)
                            p1_mem.append(turn_p1)
                        else :
                            turn_p2.append(current_game.board.board.copy())
                            turn_p2.append(current_game.done)
                            if current_game.final_state == 1 :
                                turn_p2.append(-1)
                            elif current_game.final_state == 0 :
                                turn_p2.append(0)
                            else :
                                turn_p2.append(1)
                            p2_mem.append(turn_p2)

                        p1.memorize(p1_mem)
                        p2.memorize(p2_mem)
                        break
                if verbose :
                    print(current_game.board)


class Morpion :
    def __init__(self, p1, p2, verbose=False) :
        self.p1 = p1
        self.p1.jeu = self
        self.p2 = p2
        self.p2.jeu = self
        self.board = Board()
        self.p1_turn = True
        self.verbose = verbose
        self.final_state = None
        self.done = False

    def possible_moves(self, board, p1_turn) :
        possibilities = []
        for i in range(9) :
            if board.board[i] == 0 :
                possibility = np.zeros((9), dtype=np.int8)
                possibility[i] = 1 if p1_turn else 2
                possibilities.append(possibility)

        print(possibilities)
        return possibilities


    def detect_end(self, board) :
        is_win = False
        for i in range(3) : # horizontal
            if board.board[0+i*3] != 0 and board.board[1+i*3] == board.board[0+i*3] and board.board[2+i*3] == board.board[0+i*3] :
                is_win = True
        for i in range(3) : # vertical
            if board.board[0+i] != 0 and board.board[3+i] == board.board[0+i] and board.board[6+i] == board.board[0+i] :
                is_win = True
        if board.board[0] != 0 and board.board[4] == board.board[0] and board.board[8] == board.board[0] :
            is_win = True
        if board.board[6] != 0 and board.board[4] == board.board[6] and board.board[2] == board.board[6] :
            is_win = True

        if is_win :
            self.final_state = 1 if not self.p1_turn else 2
            self.done = True
            if self.verbose :
                print("Le joueur {} gagne !".format(1 if not self.p1_turn else 2))
            return

        is_draw = True
        for i in range(9) :
            if board.board[i] == 0 :
                is_draw = False
        if is_draw :
            self.final_state = 0
            self.done = True
            if self.verbose :
                print("Egalitée")
            return

    def is_legal_move(self, move) :
        return self.board.board[np.argmax(move)] == 0

    def apply_move(self, move) :
        self.board.board[np.argmax(move)] = 1

    def next_moves(self) :
        if self.p1_turn :
            if self.verbose :
                print('Joueur 1')
            move = self.p1.play(self.board.copy(), self.p1_turn)
        else :
            if self.verbose :
                print('Joueur 2')
            move = self.p2.play(self.board.copy(), self.p1_turn)

        self.p1_turn = not self.p1_turn
        return move


class Board :
    def __init__(self) :
        self.board = np.zeros((9), dtype=np.int8)

    def __repr__(self) :
        delimiteur = "\n+---+---+---+\n"
        rep = delimiteur
        for i in range(9) :
            if i % 3 == 0 and i != 0 :
                rep += "|" + delimiteur
            rep += "| {} ".format(self.board[i])
        rep += "|" + delimiteur

        return rep

    def copy(self) :
        board_copy = Board()
        board_copy.board = self.board.copy()
        return board_copy

    def invert(self) :
        for i in range(9) :
            if self.board[i] == 1 : self.board[i] = 2; continue
            if self.board[i] == 2 : self.board[i] = 1
        return self


class Agent :
    def __init__(self) :
        raise NotImplementedError

    def play(self) :
        raise NotImplementedError

    def memorize(self, game_history) :
        self.memory.append(game_history)

    def save_memory(self, file) :
        with open(file, 'wb') as f :
            pickle.dump(self.memory, f)

    def load_memory(self, file) :
        with open(file, 'rb') as f :
            self.memory = pickle.load(f)


class Singe(Agent) :
    def __init__(self, jeu=None) :
        self.jeu = jeu
        self.memory = deque(maxlen=10000)

    def play(self, board, p1_turn) :
        possibilities = []
        for i in range(9) :
            if board.board[i] == 0 :
                possibility = np.zeros((9), dtype=np.int8)
                possibility[i] = 1
                possibilities.append(possibility)
        move = random.choice(possibilities)
        return move


class Debug(Agent) :
    def __init__(self, jeu=None) :
        self.jeu = jeu
        self.memory = deque(maxlen=200)

    def play(self, board, p1_turn) :
        print(board.board)
        choice = int(input("Position à jouer \n>>"))
        move = np.zeros((9), dtype=np.int8)
        move[choice] = 1
        return move


class DQNAgent(Agent) :
    def __init__(self, state_size, action_size) :
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def play(self, board, p1_turn):
        if np.random.rand() <= self.epsilon:
            move = np.zeros((9), dtype=int)
            move[random.randrange(9)] = 1
            return move
        formated_board = np.reshape(board.board, [1, 9])
        act_values = self.model.predict(formated_board)
        move = np.zeros((9), dtype=np.int8)
        move[np.argmax(act_values[0])] = 1
        return move

    def train(self, minibatch_size, batch):
        minibatch = random.sample(batch, batch_size)
        for state, action, next_state, done, reward in minibatch:
            target = reward                                                    # A implementer correctement
            if not done:                                                       #
                target = (reward + self.gamma *                                #
                          np.amax(self.model.predict(next_state)[0]))          #
            target_f = self.model.predict(state)                               #
            target_f[0][action] = target                                       #
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

class BatchMorpion :
    def __init__(self, gamma=0.8) :
        self.gamma = gamma

    def is_bad_move(self, game) :
        last = game[-1]
        return last[-1] == -1 and np.array_equal(last[0], last[2])

    def create_batch(self, file) :
        batch = []
        p1_win = 0
        p2_win = 0
        with open(file, 'rb') as f :
            games = pickle.load(f)
        for game in games :
            if self.is_bad_move(game) :
                game = [game[-1]]
            else :
                reward = game[-1][-1]
                for i, turn in enumerate(game[:-1]) :
                    turn[-1] = turn[-1] + reward * self.gamma**(len(game)-i)
            for turn in game :
                batch.append(turn)

        return batch

if __name__ == '__main__' :
    p1 = DQNAgent(9, 9)
    #p1 = Singe()
    p2 = Singe()

    mj = Supervisor(Morpion)
    mj.simulate(p1, p2, 10, verbose=False)

    p1.save_memory('p1.mem')

    batcher = BatchMorpion()
    batch = batcher.create_batch('p1.mem')

    #p1.train(32, batch)

