from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test, set_map
import numpy as np
import random

q_table = np.load("q_table.npy")

orientation = ['N', 'E', 'S', 'W']


def grab_ice():
    state = turn_right()
    state = move_forward()
    return state


states = [[a, b] for a in range(2) for b in range(2)]
actions = [turn_left, move_forward, grab_ice]

thin_ice_blocks = [(2, 2), (3, 3), (4, 4), (5, 5), (5, 6), (4, 3), (5, 2)]

set_speed(10)
test()
(x, y), ori, sensor, done = set_map(thin_ice_blocks)

##############################################
#### Copy and paste your step 4 code here ####
##############################################

print(q_table)

while not done:
    state = sensor[1:]
    state_idx = states.index(state)
    q_table_row = q_table[state_idx]
    action_idx = np.argmax(q_table_row)
    action = actions[action_idx]
    (x, y), ori, sensor, done = action()
##############################################
