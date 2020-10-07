import matplotlib.pyplot as plt
import datetime as datetime
import numpy as np


def log():
    # read file into string
    with open('rl_template.py', 'r') as inputfile:
        textstr = inputfile.read()
        fn = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt"

        with open("logs/"+fn, 'w') as outputfile:
            outputfile.write(textstr)


log()

N = 100  # The goal for truly winning
p = 0.25  # Probability of winning one bet
gamma = 1    # discount factor

# The states of the game when defined as a Markov Decision Process.
states = [i for i in range(1, N)]
# states[x] represents the state of currently having x chips.
# The current state value function. v[x] is the value of state x.
v = [0 for i in range(0, N+1)]
# List to represent optimal policy.
optimal_actions = [0 for i in range(0, N+1)]
# optimal_actions[x] should equal the optimal number of
# coins to bet when you currently have x chips (you're in state x).

# For both v and optimal_actions, if x == 0 or N, v[x] = optimal_actions[x] = 0.


### Implement value iteration here ###

k = 0

while True:
    old_v = v[:]
    for i in range(len(states)):
        state = states[i]
        threshold = min(state, 100 - state)
        values = []
        for bet in range(1, threshold+1):
            # 1/4 -> state + bet || 3/4 -> state - bet
            # |reward| -> bet
            value = (1/4) * (bet + gamma *
                             v[state+bet]) + (3/4) * (gamma * v[state-bet] - bet)
            values.append(value)

        v[state] = np.max(values)

    if v == old_v:
        break
    k += 1


for i in range(len(states)):
    state = states[i]
    threshold = min(state, 100 - state)
    args = []
    bets = []
    for bet in range(1, threshold+1):
        # 1/4 -> state + bet || 3/4 -> state - bet
        value = (1/4) * (bet + gamma * v[state+bet]) + \
            (3/4) * (gamma * v[state-bet] - bet)
        args.append(value)
        bets.append(bet)
    optimal_actions[state] = bets[np.argmax(args)]


######################################

# Plot state value function for every state
fig = plt.figure()
plt.plot(states, v[1:-1])
plt.show()

# Plot optimal policy for every state
plt.scatter(states, optimal_actions[1:-1])
plt.show()
