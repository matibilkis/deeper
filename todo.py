import numpy as np
import basics
import misc
import tensorflow as tf
from tensorflow.keras.layers import Dense
import random
from give_actions import Give_Action

from memory import Memory
from networks import QN_l1, QN_l2, QN_guess


basic = basics.Basics(resolution=.1)
basic.define_actions()
ats = misc.make_attenuations(layers=2)


qn_l1_prim = QN_l1()
qn_l1_targ = QN_l1()
qn_l2_prim = QN_l2()
qn_l2_targ = QN_l2()
qn_guess_prim = QN_guess()
qn_guess_targ = QN_guess()

giving_actions = Give_Action(basic.actions[0], basic.actions[1], basic.possible_phases )
