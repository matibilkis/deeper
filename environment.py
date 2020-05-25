import misc
import numpy as np
import scipy.optimize as sp
from tqdm import tqdm
import os
import shutil
import random

class Environment(misc.Basics):
    """

    Environment class.  Despite this being slower, it is a proof of principle and ideally would be implemented in an optical table controlled by this software. See give_outcome_sf method.

    ** amplitude: mean energy
    ** layers: #photodetectors
    ** number_phases (Alice's alphabet)

    """

    def __init__(self, amplitude=.4, dolinar_layers=2, number_phases=2):
        super().__init__(amplitude=amplitude,dolinar_layers=dolinar_layers,
         number_phases=number_phases)

        self.pick_phase()


    def lambda_q(self,q):
        """Auxiliary method to compute pretty good measurement bound (helstrom in this case, see Holevo book)"""
        number_states = self.number_phases
        nsig = self.amplitude**2 #in case you change...
        c=0
        for m in range(1,number_states+1):
            c+= np.exp(((1-q)*(2*np.pi*(1j)*m)/number_states) + nsig*np.exp(2*np.pi*(1j)*m/number_states))
        return c*np.exp(-nsig)


    def helstrom(self):
        """
        Returns helstrom probability sucess
        Eq (9) M-ary-state phase-shift-keying discrimination below the homodyne limit
        F. E. Becerra,1,* J. Fan,1 G. Baumgartner,2 S. V. Polyakov,1 J. Goldhar,3 J. T. Kosloski,4 and A. Migdall1
        """
        nsig=self.amplitude**2
        number_states=self.number_phases

        prob = 0
        for q in range(1,number_states+1):
            prob += np.sqrt(self.lambda_q(q))
        prob = 1 - (prob/number_states)**2

        return 1-prob

    def pick_phase(self):
        """Pick a random phase (equal priors) to send to Bob """
        self.phase = random.choices(self.possible_phases)[0]
        self.label_phase = np.where(self.possible_phases == self.phase)[0][0]

        return

    def give_outcome(self, beta,layer):
        """ Returns outcome according to current layer (needed to compute the current intensity)""" #Actually, if all intensities are equal, you don't need to keep track of the layer here...

        effective_attenuation = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])#Warning one!
        probs = [self.P(self.phase*self.amplitude, beta, effective_attenuation, n) for n in [0,1]]

        return random.choices([0.,1.],weights=probs)[0]


    def give_reward(self, guess, modality="bit_stochastic", history=[]):
        """We put label_phase to avoid problems
        with the np.round we applied to complex phases"""
        # if (self.flipped):
        #     self.label_phase = np.where(self.possible_phases == -self.phase)[0][0]
        if modality=="bit_stochastic":

            if guess == self.label_phase:
                return 1
            else:
                return 0

        else:
            prob=1
            phase_guess=self.possible_phases[guess]
            for layer in range(int(len(history)/2)):
                effective_attenuation = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])#Warning one!
                prob*=self.P(phase_guess*self.amplitude, history[layer], effective_attenuation, history[layer+1])
            return prob
            #I take a history of #(beta, outcome, beta, outcome)...
