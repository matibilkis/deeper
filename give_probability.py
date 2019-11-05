import numpy as np



def prob(beta, alpha, n):
    p0 = np.exp(-(beta-alpha)**2)
    if n == 0:
        return p0
    else:
        return 1-p0

def probability_greedy(ats, alpha, networks):
    qn_l1, qn_l2, qn_guess = networks
    p=0
    label1, beta1 = qn_l1.give_first_beta(epsilon=0)
    for n1 in [0,1]:
        for n2 in [0,1]:
            l2, beta2 = qn_l2.give_second_beta([n1, beta1], epsilon=0)
            gueslabel, guess_phase = qn_guess.give_guess([n1,n2,beta1,beta2], epsilon=0)
            p+= prob(beta1, guess_phase*np.cos(ats[0])*alpha,n1)*prob(beta2, guess_phase*np.sin(ats[0])*alpha, n2)
    return p/2
