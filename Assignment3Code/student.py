import pygame
from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden

import numpy as np
import time
import random

show_animation(False)
# set_speed(10)          # This line is only meaningful if animations are enabled.

#####################################
#### Implement steps 1 to 3 here ####
#####################################
grid_size = (8, 8)             # Size of the map
goal = (6, 6)             # Coordinates of the goal
orientation = ['N', 'E', 'S', 'W']  # List of orientations


def grab_ice():
    state = turn_right()
    state = move_forward()
    return state


# Hyperparameters: Feel free to change all of these!
states = [[a, b] for a in range(2) for b in range(2)]
actions = [turn_left, move_forward, grab_ice]


num_epochs = 100
alpha = 1
gamma = 0.95
epsilon = 0.1

step_limit = 200

penalty = -100
q_table = np.zeros([len(states), len(actions)])


# Define your reward function
# state = sensor, action = move function
def reward(state, action, num_step):
    step_decay = 1 - num_step / step_limit

    if state == [1, 1] and action == turn_left:
        return 1 * step_decay

    if state == [0, 1] and action == move_forward:
        return 1 * step_decay

    if state[1] == 0 and action == grab_ice:
        return 1 * step_decay

    return penalty


def get_action(state_idx):
    q_table_row = q_table[state_idx]
    rand = random.random()

    # Epsilon-greedy algorithm
    if rand < epsilon:
        return actions[random.randint(0, len(actions)-1)]
    else:
        max_val = np.max(q_table_row)
        max_idx_list = [idx for idx, q_val in enumerate(
            q_table_row) if q_val == max_val]
        return actions[max_idx_list[random.randint(0, len(max_idx_list)-1)]]


def get_max_q_val(state):
    state_idx = states.index(state)
    q_table_row = q_table[state_idx]
    return np.max(q_table_row)


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


for i in range(num_epochs):
    num_step = 0
    (x, y), ori, sensor, done = reset_map()
    while (x, y) != goal:
        num_step += 1
        state = sensor[1:]
        state_idx = states.index(state)
        action = get_action(state_idx)
        action_idx = actions.index(action)
        (x, y), ori, sensor, done = action()

        r = reward(state, action, num_step)

        if done or num_step > step_limit:
            r = penalty

        q_table[state_idx][action_idx] = q_table[state_idx][action_idx] + alpha * \
            (r + gamma * get_max_q_val(state) -
                q_table[state_idx][action_idx])

        if done or num_step > step_limit:
            break


####################################

np.save("q_table", q_table)
set_speed(10)
test()
(x, y), ori, sensor, done = reset_map()

###############################
#### Implement step 4 here ####
###############################
while not done:
    state = sensor[1:]
    state_idx = states.index(state)
    q_table_row = q_table[state_idx]
    action_idx = np.argmax(q_table_row)
    action = actions[action_idx]
    (x, y), ori, sensor, done = action()

##############################

# If you want to try moving around the map with your keyboard, uncomment the below lines

# set_speed(5)
# show_animation(True)
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             exit("Closing...")
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 print(turn_left())
#             if event.key == pygame.K_RIGHT:
#                 print(turn_right())
#             if event.key == pygame.K_UP:
#                 print(move_forward())
#             if event.key == pygame.K_t:
#                 test()
#             if event.key == pygame.K_r:
#                 print(reset_map())
#             if event.key == pygame.K_q:
#                 exit("Closing...")
