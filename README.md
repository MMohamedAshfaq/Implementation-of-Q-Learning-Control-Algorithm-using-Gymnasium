# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To develop a Q-Learning agent that learns the optimal policy for navigating the FrozenLake environment and reaching the goal while avoiding holes.


## Software Requirements
Python 3.x

Gymnasium

NumPy

Matplotlib

Jupyter Notebook / Google Colab


## Environment Description
FrozenLake-v1 is a grid-world environment with 16 states and 4 actions (Left, Down, Right, Up). The agent must reach the goal state while avoiding holes.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
Create the FrozenLake environment.

Initialize the Q-table with zeros.

Select actions using the epsilon-greedy policy.

Perform the Q-Learning update.

Reduce epsilon after each episode.

Repeat for 10,000 episodes.

Select the action with the highest Q-value as the learned policy.

Calculate the average reward and plot the learning curve.


## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of states:", n_states)
print("Number of actions:", n_actions)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

alpha = 0.8
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 10000


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state, epsilon):

    if np.random.random() < epsilon:
        # Explore
        return env.action_space.sample()

    else:
        # Exploit
        return np.argmax(Q[state])


# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        # Epsilon-greedy action selection
        action = choose_action(state, epsilon)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Q-Learning update
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] = Q[state, action] + alpha * (
            target - Q[state, action]
        )

        # Move to next state
        state = next_state

        # Store reward
        total_reward += reward

    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


print("\nTraining completed!")
print("Final epsilon:", epsilon)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

# State-value function
state_values = np.max(Q, axis=1)

# Best action for each state
learned_policy = np.argmax(Q, axis=1)

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", average_reward)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title("Q-Learning Curve - FrozenLake")

plt.grid(True)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()




```
---

## Output

```text
Final Q-table:

Final Q-table:
[[0.735 0.774 0.774 0.735]
 [0.735 0.    0.815 0.506]
 [0.772 0.857 0.759 0.804]
 [0.814 0.    0.    0.533]
 [0.774 0.815 0.    0.735]
 [0.    0.    0.    0.   ]
 [0.    0.902 0.    0.814]
 [0.    0.    0.    0.   ]
 [0.815 0.    0.857 0.774]
 [0.815 0.902 0.902 0.   ]
 [0.857 0.95  0.    0.857]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.902 0.95  0.857]
 [0.902 0.95  1.    0.902]
 [0.    0.    0.    0.   ]]





Estimated State-Value Function:
Estimated State-Value Function:
[[0.774 0.815 0.857 0.814]
 [0.815 0.    0.902 0.   ]
 [0.857 0.902 0.95  0.   ]
 [0.    0.95  1.    0.   ]]






Learned Policy:

[['D' 'R' 'D' 'L']
 ['D' 'L' 'D' 'L']
 ['R' 'D' 'D' 'L']
 ['L' 'R' 'R' 'L']]

Average reward over last 1000 episodes: 0.993


```
<img width="726" height="1076" alt="ex7" src="https://github.com/user-attachments/assets/6e47f456-3206-456f-b9e9-df3660e06f04" />
<img width="1374" height="978" alt="ex7 2" src="https://github.com/user-attachments/assets/c3c05a00-41a0-4190-9938-03cba8b474dd" />


---

## Result

```text
The Q-Learning algorithm was successfully implemented using the Gymnasium FrozenLake environment. The agent learned an effective policy with an average reward of 0.993.

```

---

## Inference

```text
The agent successfully learned to reach the goal while avoiding holes. The high reward indicates effective learning and convergence of the Q-table.


```

---

