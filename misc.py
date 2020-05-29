import numpy as np
import os
import random
import cmath
from math import erf
import pickle
import tensorflow as tf

############################ ONLY FOR FIRST LAYER #############
############################ ONLY FOR FIRST LAYER #############
############################ ONLY FOR FIRST LAYER #############
############################ ONLY FOR FIRST LAYER #############
def Prob(alpha, beta, n):
    p0 = np.exp(-(alpha-beta)**2)
    if n == 0:
        return p0
    else:
        return 1-p0

### this is just p(R=1 | g, n; beta) = p((-1^{g} alpha | n)) = p(n|allpha) pr(alpha)(p(n))
def qval(beta, n, guess):
    #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
    alpha = 0.4
    pn = np.sum([Prob(g*alpha, beta, n) for g in [-1,1]])
    return Prob(guess*alpha, beta, n)/pn

def ps_maxlik(beta):
    #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
    alpha = 0.4
    p=0
    for n1 in [0,1]:
       p+=Prob(np.sign(beta)*(-1)**(n1)*alpha, beta, n1)
    return p/2


############################ ONLY FOR FIRST LAYER #############
############################ ONLY FOR FIRST LAYER #############
############################ ONLY FOR FIRST LAYER #############
############################ ONLY FOR FIRST LAYER #############


def P(a,b,et,n):

    p0=np.exp(-abs((et*a)+b)**2)

    if n ==0:
        return p0
    else:
        return 1-(p0)

def insert(v,M):
    """
    Takes v, M and returns an array that has, for each element of v, a matrix M

    Example:
    x = [x0,x1]
    y = [[0,0],[0,1],[1,0],[1,1]]
    insert(x,y) returns

    [x0 0 0]
    [x0 0 1]
    [x0 1 0]
    [x0 1 1]
    [x1 0 0]
    [x1 0 1]
    [x1 1 0]
    [x1 1 1]
    """
    try:
        a=M.shape
        if len(a)<2:
            a.append(1)
    except Exception:
         a = [1,len(M)]
    result=np.zeros((a[0]*len(v),a[1] +1 )).astype(int)

    f = len(v)+1
    cucu=0
    for k in v:
        result[cucu:(cucu+a[0]),0] = k
        result[cucu:(cucu+a[0]),1:] = M
        cucu+=a[0]
    return result

def outcomes_universe(L):
    """
    Takes L (# of photodetections in the experiment) and returns
    all possible outcomes in a matrix of 2**L rows by L columns,
    which are all possible sequence of outcomes you can ever get.
    """
    a = np.array([0,1])
    two_outcomes = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(int)
    if L<2:
        return np.array([[0],[1]]).astype(int)
    elif L==2:
        return two_outcomes
    else:
        x = insert(a,two_outcomes)
        for i in range(L-3):
            x = insert(a,x)
        return x.astype(int)

def make_attenuations(layers):
    if layers == 1:
        return [0]
    else:
        ats=[0]
        for i in range(layers-1):
            ats.append(np.arctan(1/np.cos(ats[i])))
        return np.flip(ats)


class Basics():

    """
    A class that defines some common things inherited by Environment and PolicyEvaluator

    amplitude: alpha, sqrt(intensity of the coherent states)
    layers: number of displacements/photodetectors
    number_phases: how many states you want to discriminate among. Notice they will be aranged as sqrt of unity

    """
    def __init__(self, amplitude = 0.4, dolinar_layers=2, number_phases=2):

        self.dolinar_layers = dolinar_layers
        self.number_phases = number_phases
        self.at = self.make_attenuations(self.dolinar_layers) #Bob uses this if she knows the model
        self.amplitude = amplitude

        self.possible_phases=[]
        for k in croots(self.number_phases):
            self.possible_phases.append(np.round(k,10))

    def make_attenuations(self,layers):
        if layers == 1:
            return [0]
        else:
            ats=[0]
            for i in range(layers-1):
                ats.append(np.arctan(1/np.cos(ats[i])))
            return np.flip(ats)

    def P(self,a,b,et,n):
        """
        | < \beta | et* \alpha >|**2

        Notice that the real phase is not considered here, and is multiplied externally, when the function is called, as
        P(real_phase*a, beta, et, n)...

        """
        p0=np.exp(-abs((et*a)+b)**2)
        if n ==0:
            return p0
        else:
            return 1-p0

    def err_kennedy(self,beta):
        return (1 + np.exp(- (-beta + self.amplitude)**2)  - np.exp(- (beta + self.amplitude)**2)  )/2


    def homodyne(self):
        """ returns the probability of success by only doing homodyne measurement
         """

        a = self.amplitude.real
        return (1+erf(self.amplitude))/2

    def heterodyne(self):
        return (1+(1-np.exp(-2*self.amplitude**2))/np.sqrt(np.pi))/2

class PolicyEvaluator(Basics):
    def __init__(self, **kwargs):
        amplitude= kwargs.get("amplitude", .4)
        dolinar_layers=kwargs.get("dolinar_layers", 2)
        number_phases=kwargs.get("number_phases", 2)
        super().__init__(amplitude=amplitude, dolinar_layers=dolinar_layers, number_phases=number_phases)

        displacement_tree = {}
        trajectory_tree = {}
        trajectory_tree_recorded = {}
        #self.at = make_attenuations(self.number_layers)
        for layer in range(self.dolinar_layers+1):
            displacement_tree[str(layer)] = {}
            trajectory_tree[str(layer)] = {}
            trajectory_tree_recorded[str(layer)] = {}

        for k in outcomes_universe(self.dolinar_layers):
            for layer in range(self.dolinar_layers+1):
                displacement_tree[str(layer)][str(k[:layer])] = 0.
                trajectory_tree[str(layer)][str(k[:layer])] = []
                trajectory_tree_recorded[str(layer)][str(k[:layer])] = []

        self.history_tree = displacement_tree
        self.recorded_trajectory_tree = trajectory_tree
        self.recorded_trajectory_tree_would_do = trajectory_tree_recorded

    def save_hisory_tree(self, directory):
        output = open(directory+"/history_tree.pkl", "wb")
        pickle.dump(self.recorded_trajectory_tree, output)
        output.close()
        output = open(directory+"/history_tree_would_do.pkl", "wb")
        pickle.dump(self.recorded_trajectory_tree_would_do, output)
        output.close()

    def load_history_tree(self, directory):
        opp = open(directory+"/history_tree.pkl", "rb")
        self.recorded_trajectory_tree_loaded = pickle.load(opp)
        opp = open(directory+"/history_tree_would_do.pkl", "rb")
        self.recorded_trajectory_tree_loaded_would_do = pickle.load(opp)
        print("recorded_trajectory_tree_loaded")
        return

    def random_tree(self):
        actions = self.history_tree.copy()
        for k in outcomes_universe(self.dolinar_layers):
            for layer in range(self.dolinar_layers+1):
                actions[str(layer)][str(k[:layer])] = np.random.random()
        return actions

    def success_probability(self, displacements_tree):
        """
        Given a tree of conditional actions (on the outcomes history), computes
        the success probability. Notice the final action is the guess
        for the phase of the state given a given branch.
        """
        p=0
        for ot in outcomes_universe(self.dolinar_layers):
            c=1
            for layer in range(self.dolinar_layers):
                eff_at = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])
                c*=P(displacements_tree[str(self.dolinar_layers)][str(ot)]*self.amplitude,
                     displacements_tree[str(layer)][str(ot[:(layer)])], eff_at, ot[self.dolinar_layers-1-layer] ) #notice i respect that the columns of the outcomes_universe
                #correspond to the layer: the last column is the first layer.
            p += c
        return p/self.number_phases


    def give_max_lik_guess(self, history):
        prob=np.ones(self.number_phases)
        for layer in range(int(len(history)/2)):
            effective_attenuation = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])#Warning one!
            prob*=np.array([self.P(phase_guess*self.amplitude, history[layer], effective_attenuation, history[layer+1]) for
                            phase_guess in self.possible_phases])
        guess_index = np.argmax(prob)
        return guess_index
        # if return_index is True:
        #     guess_index = np.argmax(prob)
        #     return guess_index
        # else:
        #     return self.possible_phases[np.argmax(prob)]

    def greedy_strategy(self, actor, critic):
        """
        Assuming actor, critic and self have the same dolinar_layers.
            self.possible_phases are the possible phases of the coherent states

        """

        rr = np.ones((2**(self.dolinar_layers-1), self.dolinar_layers, 1))*actor.pad_value
        if self.dolinar_layers != 1:
            rr[:,1:] = np.reshape(outcomes_universe(self.dolinar_layers-1),(2**(self.dolinar_layers-1), self.dolinar_layers-1,1))

        actor.lstm.stateful = False
        preds = np.squeeze(actor(rr))
        actor.lstm.stateful = True

        if self.dolinar_layers==1:
            history = preds
            self.history_tree[str(0)][str([])] = preds
            for final_outcome in [[0], [1]]:
                final_history = np.append(history, final_outcome)
                self.history_tree[str(self.dolinar_layers)][str(final_outcome)] = self.possible_phases[self.give_max_lik_guess(final_history)]

            return self.success_probability(self.history_tree)
        else:

            for ot, seqot in zip(outcomes_universe(self.dolinar_layers-1), preds):
                for layer in range(self.dolinar_layers):
                    self.history_tree[str(layer)][str(ot[:layer])] = seqot[layer]

                history = []
                index_seqot, index_ot= 0, 0
                for index_history in range(2*self.dolinar_layers-1):
                    if index_history%2==0:
                        history.append(seqot[index_seqot])
                        index_seqot+=1
                    else:
                        history.append(ot[index_ot])
                        index_ot+=1
                for final_outcome in [[0],[1]]:
                    final_history = np.append(history, final_outcome)
                    #self.history_tree[str(self.dolinar_layers)][str(np.append(ot,final_outcome))] = self.possible_phases[critic.give_favourite_guess(final_history)[0]]
                    self.history_tree[str(self.dolinar_layers)][str(np.append(ot,final_outcome))] = self.possible_phases[self.give_max_lik_guess(final_history)]


        return self.success_probability(self.history_tree)


def record():
    if not os.path.exists("results/number_rune.txt"):
        with open("results/number_rune.txt", "w+") as f:
            f.write("0")
            f.close()
        a=0
        number_run=0
    else:
        with open("results/number_rune.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        with open("results/number_rune.txt", "w") as f:
            f.truncate(0)
            f.write(str(int(a)+1))
            f.close()
        number_run = int(a)+1
    if not os.path.exists("run_"+str(number_run)):
        os.makedirs("results/run_"+str(number_run))
        os.makedirs("results/run_"+str(number_run)+"/learning_curves")
        os.makedirs("results/run_"+str(number_run)+"/action_trees")

        for net in ["actor_primary", "actor_target", "critic_primary", "critic_target"]:
            os.makedirs("results/run_"+str(number_run)+"/networks/"+net)

    return number_run

class Complex(complex):
    def __repr__(self):
        rp = '%7.5f' % self.real if not self.pureImag() else ''
        ip = '%7.5fj' % self.imag if not self.pureReal() else ''
        conj = '' if (
            self.pureImag() or self.pureReal() or self.imag < 0.0
        ) else '+'
        return '0.0' if (
            self.pureImag() and self.pureReal()
        ) else rp + conj + ip

    def pureImag(self):
        return abs(self.real) < 0.000005

    def pureReal(self):
        return abs(self.imag) < 0.000005


def croots(n):
    if n <= 0:
        return None
    return (Complex(cmath.rect(1, 2 * k * cmath.pi / n)) for k in range(n))



@tf.function
def step_critic_tf(batched_input,labels_critic, critic, optimizer_critic):
    with tf.GradientTape() as tape:
        tape.watch(critic.trainable_variables)
        preds_critic = critic(batched_input)
        loss_critic = tf.keras.losses.MSE(tf.expand_dims(labels_critic, axis=2), preds_critic)
        loss_critic = tf.reduce_mean(loss_critic)
        grads = tape.gradient(loss_critic, critic.trainable_variables)
        #tf.print(" dL_dQ", [tf.math.reduce_mean(k).numpy() for k in grads])

        optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
        return tf.squeeze(loss_critic)

@tf.function
def critic_grad_tf(critic, experiences):
    with tf.GradientTape() as tape:
        unstacked_exp = tf.unstack(tf.convert_to_tensor(experiences), axis=1)
        to_stack = []
        actions_wathed_index = []
        for index in range(0,experiences.shape[-1]-3,2): # I consider from first outcome to last one (but guess)
            actions_wathed_index.append(index)
            to_stack.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))

        actions_indexed = tf.concat(to_stack,axis=1)
        tape.watch(actions_indexed)

        index_actions=0
        watched_exps=[tf.ones((experiences.shape[0],1,1))*critic.pad_value]
        watched_actions_unstacked = tf.unstack(actions_indexed, axis=1)
        for index in range(0,experiences.shape[-1]-1):
            if index in actions_wathed_index:
                watched_exps.append(tf.expand_dims(watched_actions_unstacked[index_actions], axis=2))
                index_actions+=1
            else:
                watched_exps.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))

        qvals = critic(tf.reshape(tf.concat(watched_exps, axis=2), (experiences.shape[0],critic.dolinar_layers+1,2)))

        dq_da = tape.gradient(qvals, actions_indexed)
        #tf.print("dq_da mean", tf.math.reduce_mean(dq_da))
        return dq_da

@tf.function
def actor_grad_tf(actor, dq_da, experiences, optimizer_actor):
    with tf.GradientTape() as tape:
        tape.watch(actor.trainable_variables)
        finns = [actor(tf.ones((experiences.shape[0], 1,1))*actor.pad_value)]
        unstacked_exp = tf.unstack(experiences, axis=1)
        for index in range(1,2*actor.dolinar_layers-2,2):
            finns.append(actor(tf.reshape(unstacked_exp[index], (experiences.shape[0], 1,1))))
        final_preds = tf.concat(finns, axis=1)
        final_preds = actor(final_preds)
        da_dtheta=tape.gradient(final_preds, actor.trainable_variables, output_gradients=-dq_da/experiences.shape[0]) #- because you wanna minimize, and the Q value maximizes..
        #/experiences.shape[0] because it's 1/N (this is checked in debugging actor notebook... proof of [...])
        optimizer_actor.apply_gradients(zip(da_dtheta, actor.trainable_variables))
    return

# @tf.function
def optimization_step(experiences, critic, critic_target, actor, actor_target, optimizer_critic, optimizer_actor):
    # actor.lstm.reset_states()
    # experiences = experiences.astype(np.float32)
    targeted_experience = actor_target.process_sequence_of_experiences_tf(experiences)
    sequences, zeroed_rews = critic_target.process_sequence_tf(targeted_experience)
    labels_critic = critic_target.give_td_errors_tf( sequences, zeroed_rews)

    loss_critic = step_critic_tf(sequences ,labels_critic, critic, optimizer_critic)

    dq_da = critic_grad_tf(critic, experiences)
    actor_grad_tf(actor, dq_da, experiences, optimizer_actor)
    return loss_critic




# ### this is just p(R=1 | g, n; beta) = p((-1^{g} alpha | n)) = p(n|allpha) pr(alpha)(p(n))
# def qval(beta, n, guess):
#     #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
#     alpha = 0.4
#     pn = np.sum([Prob(g*alpha, beta, n) for g in [-1,1]])
#     return Prob(guess*alpha, beta, n)/pn
#
# def ps_maxlik(beta):
#     #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
#     alpha = 0.4
#     p=0
#     for n1 in [0,1]:
#        p+=Prob(np.sign(beta)*(-1)**(n1)*alpha, beta, n1)
#     return p/2


#
# # def make_figure(run_id,lr, Nbatch, loss_train, loss_test, predictions,
# #  epochs, betas_train, shuffle="False"):
# #     name_title = "Learning rate: "+ str(lr)+ "\nNumber of batches: "+str(Nbatch)
# #     name="lr"+str(lr)+"_"+"nb"+str(Nbatch)
# #     if shuffle == "True":
# #         name = name+"_shuffled"
# #     plt.figure(figsize=(10,10))
# #     ax1=plt.subplot2grid((2,1),(0,0))
# #     ax2=plt.subplot2grid((2,1),(1,0))
# #     plt.subplots_adjust(hspace=0.5)
# #     plt.title(name_title)
# #     ax1.plot(np.log10(range(1,epochs+1)), loss_train, color="red", label="Training loss", alpha=0.7)
# #     ax1.plot(np.log10(range(1,epochs+1)), loss_test, color="blue", label="Testing loss", alpha=0.7)
# #     ax1.set_xlabel("epoch",size=25)
# #     ax1.legend()
# #
# #     ax2.scatter(betas_test, predictions, s=20, color="red",label="Predictions",alpha=0.7)
# #     ax2.plot(betas_test, ps(betas_test), '--',color="blue",label="Testing range" ,alpha=0.7)
# #     ax2.scatter(betas_train, ps(betas_train), color="green", label="Training set",alpha=0.7 )
# #
# #     ax2.set_xlabel(r'$\beta$',size=25)
# #     ax1.set_title("Loss evolution",size=25)
# #     ax2.set_title("Success probability: prediction vs. true", size=25)
# #     ax2.legend()
# #     plt.savefig(run_id+"/"+str(name)+".png")
# #     plt.close()
# #     return
#

#
#
# def create_dataset(number_betas, reward_samples, fake_dataset=False):
#     data = []
#     betas = np.linspace(-1,1,number_betas)
#     psucs = [ps(b) for b in betas]
#     if fake_dataset==True:
#         for i in range(len(betas)):
#             for l in range(reward_samples):
#                 data.append(np.array([betas[i],psucs[i]]))
#         return np.array(data), betas
#
#     else:
#         for i in range(len(betas)):
#             p=0
#             for l in range(reward_samples):
#                 r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
#                 p+=r
#                 data.append(np.array([betas[i],r]))
#         return np.array(data), betas
#
#     for i in range(len(betas)):
#         p=0
#         for l in range(reward_samples):
#             r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
#             p+=r
#             data.append(np.array([betas[i],r]))
#     return np.array(data), betas
#
#     for i in range(len(betas)):
#         p=0
#         for l in range(reward_samples):
#             r=np.random.choice([1,0], 1, p=[psucs[i], 1-psucs[i]])[0]
#             p+=r
#             data.append(np.array([betas[i],r]))
#     return np.array(data), betas
#
#
#
#
# def Prob(alpha, beta, n1):
#     p0 = np.exp(-(alpha-beta)**2)
#     if n1 == 0:
#         return p0
#     else:
#         return 1-p0
#
#
#
# def give_outcome(alpha,beta):
#     p0=np.exp(-(alpha-beta)**2)
#     out = np.random.choice([0,1],1,p=[p0,1-p0])[0]
#     return out
#
# def ps_maxlik(beta):
#     #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
#     alpha = 0.4
#     p=0
#     for n1 in [0,1]:
#        p+=Prob(np.sign(beta)*(-1)**(n1)*alpha, beta, n1)
#     return p/2
#
#
# def ps(beta,g):
#     #g = [guess|0, guess|1]
#     #dolinar guessing rule (= max-likelihood for L=1, careful sign of \beta)
#     alpha = 0.4
#     p=0
#     for n1 in [0,1]:
#        p+=Prob(g[n1]*alpha, beta, n1)
#     return p/2
#
#
#
#
#
# def check_folder(name):
#     if not os.path.exists(name):
#         os.makedirs(name)
#     os.chdir(name)
#
#
#
# def greedy_action(q1, betas, ep=0):
#     if np.random.random()<ep:
#         l = np.random.choice(range(len(betas)), 1)[0]
#         return l, betas[l]
#     else:
#         qs= np.squeeze(q1(np.expand_dims(betas, axis=1)).numpy())
#         l=np.where(qs==max(qs))[0]
#         # print(qs)
#         # print(ps(betas))
#         # print("***")
#         if len(l)>1:
#             l=np.random.choice(l,1)[0]
#         else:
#             l=l[0]
#         return l, betas[l]
#
#
# def gredy_action(q2, inp):
#     qval = np.squeeze(q2(np.expand_dims(inp, axis=0)).numpy())
#     l=np.where(qs==max(qs))[0]
#     if len(l)>1:
#         l=np.random.choice(l,1)[0]
#     else:
#         l=l[0]
#     return l, [-1,1][l]
#
#
# class ReplayBufferv1:
#     def __init__(self, buffer_size=10**3):
#         self.buffer_size = buffer_size
#         self.count = 0
#         self.buffer = deque()
#
#     def add(self, a, r):
#         experience = (a,r)
#         if self.count < self.buffer_size:
#             self.buffer.append(experience)
#             self.count += 1
#         else:
#             self.buffer.popleft()
#             self.buffer.append(experience)
#
#     def size(self):
#         return self.count
#
#     def sample(self, batch_size):
#         batch = []
#         if self.count < batch_size:
#             batch = random.sample(self.buffer, self.count)
#         else:
#             batch = random.sample(self.buffer, batch_size)
#         a_batch, r_batch= list(map(np.array, list(zip(*batch))))
#         return a_batch, r_batch
#
#     def clear(self):
#         self.buffer.clear()
#         self.count = 0
#
#
#
# class ReplayBufferKennedy:
#     def __init__(self, buffer_size=10**3):
#         self.buffer_size = buffer_size
#         self.count = 0
#         self.buffer = deque()
#
#     def add(self, a, n, g, r):
#         experience = (a, n , g, r)
#         if self.count < self.buffer_size:
#             self.buffer.append(experience)
#             self.count += 1
#         else:
#             self.buffer.popleft()
#             self.buffer.append(experience)
#
#     def size(self):
#         return self.count
#
#     def sample(self, batch_size):
#         batch = []
#         if self.count < batch_size:
#             batch = random.sample(self.buffer, self.count)
#         else:
#             batch = random.sample(self.buffer, batch_size)
#         a_batch, outcome_batch, guess_batch, r_batch= list(map(np.array, list(zip(*batch))))
#         return a_batch, outcome_batch, guess_batch, r_batch
#
#     def clear(self):
#         self.buffer.clear()
#         self.count = 0
#
#
# def save_models(run_id, models):
#     for k in models:
#         k.save_weights(run_id+"/models/"+k.__str__())
#
# def plot_evolution_vKennedy1(rt, pt, optimal, betas, preds=None, loss=False,
# history_betas=False, history_would_have_done=False,run_id="algo"):
#
#     matplotlib.rc('font', serif='cm10')
#     plt.rcParams.update({'font.size': 50})
#
#     plt.figure(figsize=(40,40), dpi=100)
#     T=len(rt)
#     ax1=plt.subplot2grid((2,2),(0,0))
#     ax2=plt.subplot2grid((2,2),(1,0))
#     ax3=plt.subplot2grid((2,2),(0,1))
#     ax4=plt.subplot2grid((2,2),(1,1))
#
#     ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#     ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#     ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#     ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#
#     if preds!=None:
#         ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#
#     ax3.scatter(betas, ps_maxlik(betas), color="blue", s=250, label="true values", alpha=0.6)
#     ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")
#
#     for ax in [ax1, ax2, ax3,ax4]:
#         ax.legend(prop={"size":30})
#     plt.savefig(run_id+"/learning_curves.png")
#     plt.close()
#
#     if history_betas!=False:
#         optimal_beta = betas[np.where(ps_maxlik(betas) == max(ps_maxlik(betas)))[0][0]]
#         plt.figure(figsize=(40,40), dpi=100)
#
#         plt.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
#         plt.hist(history_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
#         plt.legend()
#         plt.text(optimal_beta, 0, "*", size=350)
#         plt.text(-optimal_beta, 0, "*", size=350)
#
#         plt.savefig(run_id+"/histograma_betas.png")
#         plt.close()
#
#
#
#         plt.figure(figsize=(40,40), dpi=100)
#         T=len(rt)
#         ax1=plt.subplot2grid((1,1),(0,0))
#
#         ax1.plot(np.arange(1, len(history_betas)+1),history_betas, color="red", linewidth=15, alpha=0.8, label="done")
#         ax1.plot(np.arange(1, len(history_betas)+1),history_would_have_done, color="green", linewidth=15, alpha=0.8, label="would have done")
#         ax1.plot(np.arange(1, len(history_betas)+1),np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#         ax1.plot(np.arange(1, len(history_betas)+1),-np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#
#
#         for ax in [ax1]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/evolution_actions.png")
#         plt.close()
#     return
#
# def qguess(b,n1,sign=1):
#     alpha=0.4
#     pn1 = np.sum([Prob(ph*alpha, b, n1) for ph in [-1,1]])
#     return Prob(sign*alpha, b, n1)/pn1
#
#
# def plot_evolution_vKennedy_last(rt, pt, optimal, betas, preds=None, pred_guess=None, loss=False,
# history_betas=False, history_would_have_done=False,run_id="algo"):
#
#     matplotlib.rc('font', serif='cm10')
#     plt.rcParams.update({'font.size': 50})
#
#     plt.figure(figsize=(40,40), dpi=100)
#     T=len(rt)
#     ax1=plt.subplot2grid((2,3),(0,0))
#     ax2=plt.subplot2grid((2,3),(1,0))
#     ax3=plt.subplot2grid((2,3),(0,1))
#     ax4=plt.subplot2grid((2,3),(1,1))
#     ax5=plt.subplot2grid((2,3),(0,2))
#     ax6=plt.subplot2grid((2,3),(1,2))
#
#
#     ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#     ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#     ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#     ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#
#     ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#     ax3.scatter(betas, ps_maxlik(betas), color="blue", s=250, label="true values", alpha=0.6)
#     ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")
#
#     pred_guess_0, pred_guess_1 = pred_guess
#     ax5.plot(betas, pred_guess_0[:,0], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta,0), +)$')
#     ax5.plot(betas, [qguess(b,0,sign=1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta,0), +)$')
#     ax5.plot(betas, pred_guess_1[:,0], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta, 1), +)$')
#     ax5.plot(betas, [qguess(b,1,sign=1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta, 1) +)$')
#
#     ax6.plot(betas, pred_guess_0[:,1], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta,0), -)$')
#     ax6.plot(betas, [qguess(b,0,sign=-1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta,0), -)$')
#     ax6.plot(betas, pred_guess_1[:,1], color="red", linewidth=15, alpha=0.8, label=r'$\hat{Q}((\beta, 1), -)$')
#     ax6.plot(betas, [qguess(b,1,sign=-1) for b in betas], color="blue", linewidth=15, alpha=0.8, label=r'$Q((\beta, 1) -)$')
#
#     for ax in [ax1, ax2, ax3,ax4,ax5,ax6]:
#         ax.legend(prop={"size":30})
#     plt.savefig(run_id+"/learning_curves.png")
#     plt.close()
#
#     if history_betas!=False:
#         optimal_beta = betas[np.where(ps_maxlik(betas) == max(ps_maxlik(betas)))[0][0]]
#         plt.figure(figsize=(50,50), dpi=100)
#
#         plt.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
#         plt.hist(history_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
#         plt.legend()
#         plt.text(optimal_beta, 0, "*", size=350)
#         plt.text(-optimal_beta, 0, "*", size=350)
#
#         plt.savefig(run_id+"/histograma_betas.png")
#         plt.close()
#
#
#
#         plt.figure(figsize=(40,40), dpi=100)
#         T=len(rt)
#         ax1=plt.subplot2grid((1,1),(0,0))
#
#         ax1.plot(np.arange(1, len(history_betas)+1),history_betas, color="red", linewidth=15, alpha=0.8, label="done")
#         ax1.plot(np.arange(1, len(history_betas)+1),history_would_have_done, color="green", linewidth=15, alpha=0.8, label="would have done")
#         ax1.plot(np.arange(1, len(history_betas)+1),np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#         ax1.plot(np.arange(1, len(history_betas)+1),-np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
#
#
#         for ax in [ax1]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/evolution_actions.png")
#         plt.close()
#
#
#     return
#
#
#
# def plot_evolution_v1(rt, pt, optimal, betas, preds=None, loss=False,
# history_betas=False, history_would_have_done=False,run_id="algo"):
#     matplotlib.rc('font', serif='cm10')
#     plt.rcParams.update({'font.size': 50})
#
#     if loss != "False":
#         plt.figure(figsize=(40,40), dpi=100)
#         T=len(rt)
#         ax1=plt.subplot2grid((2,2),(0,0))
#         ax2=plt.subplot2grid((2,2),(1,0))
#         ax3=plt.subplot2grid((2,2),(0,1))
#         ax4=plt.subplot2grid((2,2),(1,1))
#
#         ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#         ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         ax2.plot(np.log10(np.arange(1,T+1)),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#         ax2.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         if preds!=None:
#             ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#         ax3.scatter(betas, ps(betas), color="blue", s=250, label="true values", alpha=0.6)
#         ax4.plot(loss, color="black",  linewidth=15,alpha=0.5, label="Loss evolution")
#
#         for ax in [ax1, ax2, ax3,ax4]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/learning_curves.png")
#         plt.close()
#
#         if history_betas!=False:
#             optimal_beta = betas[np.where(ps(betas) == max(ps(betas)))[0][0]]
#             plt.figure(figsize=(40,40), dpi=100)
#
#             plt.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
#             plt.hist(history_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
#             plt.legend()
#             plt.text(optimal_beta, 0, "*", size=350)
#             plt.text(-optimal_beta, 0, "*", size=350)
#
#             plt.savefig(run_id+"/histograma_betas.png")
#             plt.close()
#         return
#     else:
#         plt.figure(figsize=(30,15))
#         T=len(rt)
#         ax1=plt.subplot2grid((2,2),(0,0))
#         ax2=plt.subplot2grid((2,2),(1,0))
#         ax3=plt.subplot2grid((2,2),(0,1), rowspan=2)
#         ax1.plot(np.arange(1,T+1),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
#         ax1.plot(optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         ax2.plot(np.arange(1,T+1),pt, color="red", linewidth=15, alpha=0.8, label=r'$P_t$')
#         ax2.plot(optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
#         if preds!=None:
#             ax3.scatter(betas, preds, color="red", s=250, label="predictions", alpha=0.6)
#         ax3.scatter(betas, ps(betas), color="blue", s=250, label="true values", alpha=0.6)
#
#         for ax in [ax1, ax2, ax3]:
#             ax.legend(prop={"size":30})
#         plt.savefig(run_id+"/learning_curves.png")
#         plt.close()
#         return
