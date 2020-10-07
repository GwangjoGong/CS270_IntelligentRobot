import pygame
from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden

import numpy as np
import time
import random

show_animation(True)
set_speed(3)          # This line is only meaningful if animations are enabled.

#####################################
#### Implement steps 1 to 3 here ####
#####################################
grid_size = (8, 8)             # Size of the map
goal = (6, 6)             # Coordinates of the goal
orientation = ['N', 'E', 'S', 'W']  # List of orientations

# Hyperparameters: Feel free to change all of these!
states = [(a, b) for a in range(8) for b in range(8)]
actions = [(a, b) for a in range(8) for b in range(8)]


num_epochs = 100
alpha = 1
gamma = 0.8
epsilon = 0.2

impossible = -100
q_table = np.zeros([64, 64])

max_step = 100
cache_threshold = 2


# return distance of two point a,b
def distance(a, b):
    dist_x = abs(a[0]-b[0])
    dist_y = abs(a[1]-b[1])
    return dist_x + dist_y


def choose_action(current_idx, ori, sensor, is_train):
    global q_table
    masked = []
    candidates = []
    action_chosen = (-1, -1)
    current_coord = states[current_idx]

    # if surrounded by ice, go backward and mask as impossible
    if sensor == [1, 1, 1]:
        for i in range(len(q_table)):
            if q_table[i][current_idx] != impossible:
                masked.append(((i, current_idx), q_table[i][current_idx]))
                q_table[i][current_idx] = impossible
        action_chosen = get_backward_coord(current_coord, ori)
        return masked, action_chosen

    q_table_row = q_table[current_idx]

    # mask ice block to q-table based on sensor and orientation
    ice_coords = get_all_ice_coord(current_coord, ori, sensor)
    #print("ices : ", ice_coords)
    for coord in ice_coords:
        coord_idx = actions.index(coord)
        for i in range(len(q_table)):
            if q_table[i][coord_idx] != impossible:
                masked.append(((i, coord_idx), q_table[i][coord_idx]))
                q_table[i][coord_idx] = impossible

    # filter impossible actions
    for idx, q_val in enumerate(q_table_row):
        if q_val != impossible:
            candidates.append((q_val, actions[idx]))

    # Epsilon-greedy algorithm
    #print("candidates : ", candidates)
    rand = random.random()

    if is_train and rand < epsilon:
        # print("Epsilon")
        random_idx = random.randint(0, len(candidates)-1)
        action_chosen = candidates[random_idx][1]
    else:
        max_q_val = np.max(list(map(lambda x: x[0], candidates)))
        #print("max : ", max_q_val)
        max_candidates = []
        for i in range(len(candidates)):
            if candidates[i][0] == max_q_val:
                max_candidates.append(candidates[i])
        max_rand_idx = random.randint(0, len(max_candidates) - 1)
        action_chosen = max_candidates[max_rand_idx][1]

    return masked, action_chosen


def get_front_coord(current, ori):
    x, y = current
    if ori == 'S':
        return x, y+1
    elif ori == 'W':
        return x-1, y
    elif ori == 'N':
        return x, y-1
    else:
        return x+1, y


def get_backward_coord(current, ori):
    x, y = current
    if ori == 'N':
        return x, y+1
    elif ori == 'E':
        return x-1, y
    elif ori == 'S':
        return x, y-1
    else:
        return x+1, y


def get_all_ice_coord(current, ori, sensor):
    ice_coords = []

    ori_idx = orientation.index(ori)
    left_idx = ori_idx - 1
    if left_idx < 0:
        left_idx += 4
    right_idx = ori_idx + 1
    if right_idx > 3:
        right_idx -= 4

    left_ori = orientation[left_idx]
    right_ori = orientation[right_idx]

    if sensor[0] == 1:
        ice_coords.append(get_front_coord(current, left_ori))
    if sensor[1] == 1:
        ice_coords.append(get_front_coord(current, ori))
    if sensor[2] == 1:
        ice_coords.append(get_front_coord(current, right_ori))

    return ice_coords


def get_max_q_val(action_idx):
    max_q_val = impossible
    q_table_row = q_table[action_idx]
    for q_val in q_table_row:
        if q_val != impossible and q_val > max_q_val:
            max_q_val = q_val

    return max_q_val


def move_to_dest(current, dest, ori):
    state = (dest, ori, [], False)  # temp value
    #print(current, dest)
    ori_idx = orientation.index(ori)
    dest_ori = get_dest_ori(current, dest)
    dest_ori_idx = orientation.index(dest_ori)

    #print(ori, dest_ori)

    # turning
    diff = dest_ori_idx - ori_idx

    if diff == 3:
        state = turn_left()
    elif diff == -3:
        state = turn_right()
    elif abs(diff) == 2:
        for _ in range(2):
            state = turn_left()
    elif diff == -1:
        state = turn_left()
    elif diff == 1:
        state = turn_right()

    # move forward
    state = move_forward()
    return state


def get_dest_ori(current, dest):
    if current[0] == dest[0]:
        # vertical move
        if current[1] - dest[1] > 0:
            return "N"
        else:
            return "S"
    else:
        # horizontal move
        if current[0] - dest[0] > 0:
            return "W"
        else:
            return "E"


# Define your reward function
# state, action are both coordinates
def reward(state, action, num_step, coord_visited):
    if distance(state, action) > 1:
        return impossible  # impossible action

    # if action == goal:
    #     return 1 - num_step / 1000
    # return 0

    r_direction = distance(state, goal) - distance(action, goal)

    r_action = distance(action, goal) / 10
    diff = abs(action[0] - action[1]) + 1
    r_diagonal = r_action / diff

    r_step = 1 - num_step/max_step

    #cv_length = len(coord_visited)
    #r_repeat = 0.5 - coord_visited[cv_length-10:].count(action) / 10

    # dist_state = distance(state, goal)
    # score_state = 10 - dist_state

    # dist_action = distance(action, goal)
    # score_action = 10 - dist_action

    # if score_action - score_state <= 0:
    #     return (-1) * score_action / 10

    # reward about action's distance 0...10
    # r_dist = 1 - dist_action / 10

    # reward about action's direction 1 / -1
    # r_direction = dist_state - dist_action

    return r_direction * r_diagonal * r_step


# Q-Table setup (marking impossible)
for i in range(len(q_table)):
    x, y = states[i]
    if x == 0 or x == 7:
        q_table[i, :] = impossible
    if y == 0 or y == 7:
        q_table[i, :] = impossible
    for j in range(len(q_table[i])):
        ax, ay = states[j]
        if ax == 0 or ax == 7:
            q_table[i][j] = impossible
        if ay == 0 or ay == 7:
            q_table[i][j] = impossible
        if distance((x, y), (ax, ay)) != 1:
            q_table[i][j] = impossible


for i in range(num_epochs):
    (x, y), ori, sensor, done = reset_map()
    masked = []
    num_step = 0
    coord_visited = [(x, y)]

    while not done:
        num_step += 1
        current_state = (x, y)
        current_idx = states.index((x, y))
        masked_item, action_chosen = choose_action(
            current_idx, ori, sensor, True)
        action_idx = actions.index(action_chosen)
        (x, y), ori, sensor, done = move_to_dest(
            current_state, action_chosen, ori)

        r = reward(current_state, action_chosen,
                   num_step, coord_visited)

        #print(action_chosen, r)
        q_table[current_idx][action_idx] = q_table[current_idx][action_idx] + alpha * \
            (r + gamma * get_max_q_val(action_idx) -
                q_table[current_idx][action_idx])
        # print(q_table[current_idx][action_idx])
        # print("---------------")

        coord_visited.append(action_chosen)
        masked += masked_item
        # time.sleep(10)

        if num_step >= max_step:
            break

    for (qi, qj), org_val in masked:
        q_table[qi][qj] = org_val

#####################################

np.save("q_table", q_table)

set_speed(3)
test()
(x, y), ori, sensor, done = reset_map()

###############################
#### Implement step 4 here ####
###############################
masked = []
while not done:
    current_idx = states.index((x, y))
    masked_item, action_chosen = choose_action(current_idx, ori, sensor, False)
    action_idx = actions.index(action_chosen)
    (x, y), ori, sensor, done = move_to_dest((x, y), action_chosen, ori)
    masked += masked_item

for (qi, qj), org_val in masked:
    q_table[qi][qj] = org_val
###############################

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
