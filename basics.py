import numpy as np
import misc
from math import erf

class Basics():

    """
    A class that defines some common things about the Environment, Agent and Training.

    Here we define the actions given a resolution and bound_displacements (max value of displacement), and the probability of error (which is our figure of merit).

    Notice also input efficiency, which takes into account dark counts probability
    """
    def __init__(self, layers=2, resolution=.1, number_phases=2, bound_displacements=1, efficiency=1):
        self.layers = layers
        self.number_phases = number_phases
        self.resolution = resolution
        self.bound_displacements = bound_displacements
        #### I define this here to implement the "planing" in the agent if the model is knonw...
        self.at = misc.make_attenuations(self.layers) #Bob uses this if she knows the model
        # print("Initializing environment with method "+ how + ".Attenuations: np.cos(  ", self.at, ")")
        self.amplitude = .56
        if self.number_phases>2:
            self.possible_phases=[]
            for k in misc.croots(self.number_phases):
                self.possible_phases.append(np.round(k,10))
        else:
            self.possible_phases = [-1,1]

        self.number_actions = (self.bound_displacements/self.resolution)**self.layers
        self.number_actions = self.number_actions*self.number_phases
        self.efficiency = efficiency
        #########   actions: first layer for real will be restricted in the binary case ######
        #########   actions: first layer for real will be restricted in the binary case ######

    def define_actions(self):
        """ We define the displacements (value of the action for the RL agent).
            For 2 phases (\pm) we only take real displacements, and complex displacements in the case of more phases.

            The first layer consists only on negative displacements.
        """
        if self.number_phases==2:
            self.all_or_not = 0
            if self.layers==1:
                self.actions = [np.arange(-self.bound_displacements, 0 + self.resolution, self.resolution)]
            else:
                self.actions = [np.arange(-self.bound_displacements, 0 + self.resolution, self.resolution), np.arange(-self.bound_displacements,self.bound_displacements +self.resolution, 2*self.resolution)]
            self.action_indexes = np.arange(0,len(self.actions[0]))

        else:
            self.number_displacements = int(2*(self.bound_displacements)/self.resolution)+1

            self.actions = np.zeros((self.number_displacements,self.number_displacements),dtype=complex)
            for index1,action1 in np.ndenumerate(np.arange(-self.bound_displacements,self.bound_displacements+self.resolution,self.resolution)):
                for index2,action2 in np.ndenumerate(1j*np.flip(np.arange(-self.bound_displacements,self.bound_displacements+self.resolution,self.resolution))):
                    self.actions[(index1[0],index2[0])] = action1+action2

            self.actions_matrix_form = self.actions
            self.actions = self.actions.flatten()

            self.action_indexes = np.arange(0,len(self.actions))

        return


    def P(self,a,b,et,n):
        """
        | < \beta | et* \alpha >|**2

        Notice that the real phase is not considered here, and is multiplied externally, when the function is called, as
        P(real_phase*a, beta, et, n)...
        """

        p0=np.exp(-abs((et*a)+b)**2)
        if n ==0:
            return p0*self.efficiency
        else:
            return 1-(p0*self.efficiency)

    def err_kennedy(self,beta):
        return (1 + np.exp(- (-beta + self.amplitude)**2)  - np.exp(- (beta + self.amplitude)**2)  )/2

    def homodyne(self):
        """ returns the probability of success by only doing homodyne measurement
         """

        a = self.amplitude.real
        return (1+erf(self.amplitude))/2

    def heterodyne(self):
        # """ returns the probability of success by only doing heterodyne measurement """
        # a = self.amplitude.real
        #
        # return (1+erf(a/2))/2
        return (1+(1-np.exp(-2*self.amplitude**2))/np.sqrt(np.pi))/2


    def probability_error(self, betas):
        """
        **** probability of error for a discrimination taks, of symmetric coherent states, with len(betas) displacements and photodetector, allowing feedback from the later photodetector to the current displacement.


        How it works:

        What matrix B represents is that up to layer k, you consider the same displacements, and then varies
        from the different outcomes you might obtain departing from layer k. Then k varies, and you end up
        with all the tree of possible displacements. It's assumed that the same bunch of displacements is performed when
         you obtain the same sub-sequence of outcomesself.

        Note that the last number s1[-1], s2[-1] is not important for the displacements, as you not
        measure after that outcome: for example, if you obtain:

        s1 = {0,1}
        s2 = {0,0}

        But IT IS important if you obtain

        s1 = {1,0}
        s2 = {0,0}
    ----------------------------------------------------------------------------------------------------------------------------------
        Example with 3 displacements & 3 photodetectors:

        We encode all possible displacements in a matrix B which will be of the form

        [bin  bin  bin  bin]
        [b0   b1   b0   b1]
        [b00  b01  b10  b11]

        It's important that the input is  params[0] = [bin, bin-0, bin-0, ...]

        The idea with this notation is as follows. At first displacement, you make displacement "bin", independent
        of the outcomes you can get, or the oucomes you go (which are none). At second displacement, you will
        make a displacement b0 or b1, depending on the oucome of the first photodetector. Still, you would have
        performed bin in the first layer (first displacement), so we take into account all possible displacements, which
        will be 2**(L-1) sequence of L displaecements.

        In this sense, we need to input all possible displacements, which are (2**(L) -1) where L is the number of photodetections
        and we substract 1 because we discplace first and then measure for the first time.


        So matrix B is a matrix of 2**L columns, and L rows.

    ----------------------------------------------------------------------------------------------------------------------------------
        #### BEAM SPLITTER NOTATION ####

             |0>
    |a> ----- \ ----- |cos(\theta) a>
              '
              '
         |sin(\theta)*a>
              '

        params[1] should have L numbers, which are the thetas, between 0 and Pi/2.


        To do: minimize with adaptative BS.
        """
        outcomes = misc.outcomes_universe(self.layers)
        if self.number_phases>2:
            B=np.zeros((self.layers,2**(self.layers-1)),dtype=complex)
        else:
            B=np.zeros((self.layers,2**(self.layers-1)))


        for k in range(self.layers):
            B[k,:] = np.repeat(betas[2**k -1:2**(k+1)-1],2**(self.layers-k-1),axis=0).flatten()
        p=0
        k=0

        for i in range(outcomes.shape[0]):
            ot = outcomes[i]
            if self.layers==1:
                ot = [ot]
            if i%2==0:
                bb = B[:,k]
                k+=1
            term =0
            compare = []
            for phase in misc.croots(self.number_phases):
                term = 1
                for j in range(self.layers):
                    r =  np.prod( np.sin(self.at[:j]))
                    t = np.cos(self.at[j])

                    term = term*self.P(phase*self.amplitude,bb[j],r*t, ot[j])
                compare.append(term)
            p+=np.max(compare)
        return 1- (p/self.number_phases)
