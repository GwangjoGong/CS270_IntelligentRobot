from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math

show_animation(True)
# This line is only meaningful if animations are enabled.
set_speed(500)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grab_ice():
    state = turn_right()
    state = move_forward()
    return state


grid_size = (8, 8)
goal = (6, 6)
orientation = ['N', 'E', 'S', 'W']
actions = [move_forward, turn_left, grab_ice]
num_epochs = 2000

epsilon = 0.01
gamma = 0.95

dim_hidden = 128
step_limit = 200


BATCH_SIZE = 64


def reward(state, action, num_step):
    """
    Task 6 (optional) - design your own reward function
    """
    step_decay = 1 - num_step / step_limit
    if (state.numpy() == [1, 1]).all() and action == turn_left:
        return 100 - step_decay
    if (state.numpy() == [0, 1]).all() and action == move_forward:
        return 100 - step_decay

    if state[1].item() == 0 and action == grab_ice:
        return 100 - step_decay

    return 0


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """
        Task 3 -
        push input: "transition" into replay meory
        """

        self.memory.append(transition)
        self.position += 1
        self.position %= self.capacity

    def sample(self, batch_size):
        """
        Task 3 -
        give a batch size, pull out batch_sized samples from the memory
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        """
				Task 1 -
				generate your own deep neural network
				"""
        self.fc1 = nn.Linear(2, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, 3)

    def forward(self, x):
        """
        Task 1 -
        generate your own deep neural network
        """
        # shape of x : 2 * 1
        # print(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # list of transitions with length BATCH_SIZE
    transition = memory.sample(BATCH_SIZE)
    """
    Task 4: optimize model
    """
    state_action_values = []
    expected_state_action_values = []

    for i in range(BATCH_SIZE):
        state = transition[i][0]
        action = transition[i][1]
        reward = transition[i][2]
        next_state = transition[i][3]
        done = transition[i][4]

        # print(
        #   f"state : {state}, action: {action}, reward: {reward}, next : {next_state}")

        state_action_value = policy_net(state)[action]

        next_state_value = torch.max(target_net(next_state))
        if done:
            expected_state_action_value = reward
        else:
            expected_state_action_value = torch.Tensor([(
                gamma * next_state_value.item()) + reward.item()])

        state_action_values.append(state_action_value.view(1))
        expected_state_action_values.append(expected_state_action_value)

    state_action_values = torch.cat(
        state_action_values)
    expected_state_action_values = torch.cat(
        expected_state_action_values)

    #print(state_action_values.shape, expected_state_action_values.shape)

    loss = F.mse_loss(state_action_values,
                      expected_state_action_values)
    #print(f"loss : {loss} ")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


def select_action(state):
    """
    Task 2: select action
    """
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
    with torch.no_grad():
        print(f"policy : {policy_net(state)}")
        return torch.argmax(policy_net(state))


TARGET_UPDATE = 5
for i in range(num_epochs):
    if i % 100 == 0:
        print(f"Epochs : {i}")
    num_step = 0
    (x, y), ori, sensor, done = reset_map()
    while (x, y) != goal:
        num_step += 1
        o_i = orientation.index(ori)
        cur_state = torch.Tensor(sensor[1:])

        idx_action = select_action(cur_state)

        action = actions[idx_action]
        (new_x, new_y), new_ori, new_sensor, done = action()
        new_o_i = orientation.index(new_ori)

        new_state = torch.Tensor(new_sensor[1:])
        reward_val = torch.FloatTensor(
            [reward(cur_state, action, num_step)], device=device)

        if done or num_step > step_limit:
            reward_val = torch.FloatTensor([-100], device=device)

        # tuple ( state, action, reward, next_state, done )
        transition = (cur_state, idx_action,
                      reward_val, new_state, done)

        memory.push(transition)
        (x, y), ori, sensor = (new_x, new_y), new_ori, new_sensor
        optimize_model()

        if done or num_step > step_limit:
            break

    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

"""
Task 5 - save your policy net
"""
torch.save(policy_net.state_dict(), "./policy_net.pt")


def test_network():
    """
    Task 5: test your network
    """
    set_speed(3)
    test()
    (x, y), ori, sensor, done = reset_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN().to(device)
    policy_net.load_state_dict(torch.load("./policy_net.pt"))
    policy_net.eval()

    while not done:
        o_i = orientation.index(ori)
        """
		fill this section to test your network
		"""
        cur_state = torch.Tensor(sensor[1:])
        idx_action = torch.argmax(policy_net(cur_state))
        action = actions[idx_action]
        (x, y), ori, sensor, done = action()


test_network()

###############################

# If you want to try moving around the map with your keyboard, uncomment the below lines
# import pygame
# set_speed(5)
# show_animation(True)
# while True:
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			exit("Closing...")
# 		if event.type == pygame.KEYDOWN:
# 			if event.key == pygame.K_LEFT: print(turn_left())
# 			if event.key == pygame.K_RIGHT: print(turn_right())
# 			if event.key == pygame.K_UP: print(move_forward())
# 			if event.key == pygame.K_t: test()
# 			if event.key == pygame.K_r: print(reset_map())
# 			if event.key == pygame.K_q: exit("Closing...")
