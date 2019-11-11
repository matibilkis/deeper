import numpy as np



def prob(beta, alpha, n):
    p0 = np.exp(-(beta-alpha)**2)
    if n == 0:
        return p0
    else:
        return 1-p0

def probability_greedy(alpha, networks):
    qn_l1, qn_guess = networks
    p=0
    label1, beta1 = qn_l1.give_first_beta(epsilon=0)
    for n1 in [0,1]:
        gueslabel, guess_phase = qn_guess.give_guess([n1,beta1], epsilon=0)
        p+= prob(beta1, guess_phase*alpha,n1)
    return p/2
