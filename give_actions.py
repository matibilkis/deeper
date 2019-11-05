import numpy as np


class Give_Action():
    def __init__(self, actions0, actions1, possible_phases):
        self.actions0 = actions0
        self.actions1 = actions1
        self.possible_phases = possible_phases


    def give_first_beta(self, network , epsilon):
        q1s = network
        if np.random.random() < epsilon:
            label = np.random.choice(np.arange(len(self.actions0)))
            return label, self.actions0[label]
        else:
            input = np.expand_dims(np.array([]), axis=0)
            q1s = qn_l1_prim(input)
            q1s = q1s.numpy()
            label = np.argmax(q1s)
            beta1 = self.actions0[label]
            return label, beta1

    def give_second_beta(self,network, new_state, epsilon):
        q2s = network
        if np.random.random() < epsilon:
            label = np.random.choice(np.arange(len(self.actions1)))
            return label, self.actions1[label]
        else:
            input = np.expand_dims(np.array(new_state), axis=0)
            q2s = qn_l2_prim(input)
            q2s = q2s.numpy()
            label = np.argmax(q2s)
            beta2 = self.actions1[label]
            return label, beta2


    def give_guess(network, new_state, epsilon):
        qguess = network
        if np.random.random() < epsilon:
            guess = np.random.choice(self.possible_phases,1)[0]
            return int((guess+1)/2), guess
        else:
            input = np.expand_dims(np.array(new_state), axis=0)
            qguess = qn_guess_prim(input)
            guess = qguess.numpy()
            label = np.argmax(guess)
            guess = self.possible_phases[label]
            return int((guess+1)/2), guess
