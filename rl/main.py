import gym
import gym_maze
import numpy as np

def get_next_location(action):
    next_state, reward, done, truncated = env.step(action)
    print(next_state)
    return next_state[0], next_state[1]


def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)


discount_factor = 0.9
learning_rate = 0.9


def q_learning(num):
    epsilon = 0.9
    # counter = 0
    sum = 0
    sum_of_rewards_array = []
    iterations_array = []
    episode_steps = []

    for episode in range(num):
        current_state = env.reset()
        sum_of_rewards = 0
        row_index = int(current_state[0])
        column_index = int(current_state[1])
        limit = 0
        done = False

        steps = 0
        while not done:
            steps += 1
            epsilon -= 0.001
            limit += 1
            # if limit > 200:
            #     break

            # env.render()
            action_index = get_next_action(row_index, column_index, epsilon)

            old_row_index, old_column_index = row_index, column_index

            next_state, reward, done, truncated = env.step(action_index)
            sum_of_rewards += reward
            row_index = next_state[0]
            column_index = next_state[1]

            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (
                        discount_factor * np.max(q_values[row_index, column_index, :])) - old_q_value

            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
            # if(done):
            # counter += 1
            # print(episode,':', steps, end='   ')

        sum += steps
        sum_of_rewards_array.append(sum_of_rewards)
        episode_steps.append(steps)
        iterations_array.append(episode)
    print('Training complete!')
    avg = sum // num
    print(avg)


env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()
NUM_EPISODES = 1000

q_values = np.zeros((10, 10, 4))
q_learning(NUM_EPISODES)
#print(q_values)