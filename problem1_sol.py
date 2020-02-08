import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt
from keras.models import load_model

class DQNAgent:
    def __init__(self, env):
        self.MODEL_NAME = "DQN_MOUNTAINCAR"
        self.env = env
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.n
        self.replay_memory = deque(maxlen=20000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.00  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.009
        self.learning_rate = 0.01
        self.mse_loss_per_episode = []

        ##############################################
        #-------------- Note ------------------------#
        # Uncomment following lines to start learning #
        # the model from scrach                      #
        # ############################################
        
        # self.model = self.create_model()
        # self.target_model = self.create_model()
        
        ##############################################
        #-------------- Note ------------------------#
        # Uncomment following line to continue       #
        # learning from the previously learnt model  #
        # ############################################
        
        self.model = load_model("best_one.h5")
        self.target_model = load_model("best_one.h5") 
        # target_model = Model to predict future rewards/ target values
        # If this step is not used model does not converge

        # count the episodes to update the target model and epsilon
        self.target_model_update_count = 0
    def create_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_shape, activation='relu'))
        model.add(Dense(self.action_shape))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        return model
    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
    def train_model(self, batch_size):
        # If memory does not have enough elements to create batch
        if len(self.replay_memory) < batch_size:
            return
        # choose batch_size number of elements from memory
        mini_batch = random.sample(self.replay_memory, batch_size)
        episode_loss = []
        for sample in mini_batch:
            state, action, reward, new_state, done = sample
            # current_q is the prediction of three action values of current_q
            current_q = self.model.predict(state)
            # if you reach end of the batch, there is no future reward, so
            # current reward is the final reward for that action
            # else, use the bellman equation
            if done:
                current_q[0][action] = reward
            else:
                max_future_q = max(
                    self.target_model.predict(new_state)[0])
                current_q[0][action] = reward + max_future_q * self.gamma
            history = self.model.fit(state, current_q,epochs=1, verbose=0)
            episode_loss.append(history.history['loss'][0])
        avg_loss = np.array(episode_loss)
        self.mse_loss_per_episode.append(np.mean(avg_loss))
        print("Avg loss: ", np.mean(avg_loss))       
        self.target_model_update_count += 1
        # Update the target model every seven episodes
        if self.target_model_update_count % 7 == 0:
            self.train_target_model()
        # Update the epsilon every 35 episodes
        if self.target_model_update_count % 35 == 0:
          if(self.epsilon > self.epsilon_min):
            self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)
          self.target_model_update_count = 0
    # This function copies weights from current model to the target model
    # but this is done less frequently
    def train_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    # Based on the epsilon value, make exploration/exploitation decision
    def take_action(self, state):
        r = np.random.random()
        if r < max(self.epsilon, self.epsilon_min):
            # Go for the exploration
            return self.env.action_space.sample()
        # else, go for the exploitation
        return np.argmax(self.model.predict(state)[0])
    
    def save_model(self, fn):
        self.model.save(fn)

def get_running_avg(reward_history):
    if len(reward_history) < 100:
        return -200
    else:
        new_array = np.array(reward_history[-100:])
        return np.mean(new_array)

def main():
    env = gym.make('MountainCar-v0')
    env.seed(42);
    # Print some info about the environment
    print("State space (gym calls it observation space)")
    print(env.observation_space)
    print("\nAction space")
    print(env.action_space)
    # Parameters
    NUM_STEPS = 200
    NUM_EPISODES = 1000
    LEN_EPISODE = 200
    reward_history = []
    running_averages = []
    below_175 = []
    below_150 = []
    below_100 = []
    # create DQNAgent
    agent = DQNAgent(env)
    # Run for NUM_EPISODES
    for episode in range(NUM_EPISODES):
        episode_reward = 0
        curr_state = env.reset().reshape(1,2)
        for step in range(LEN_EPISODE):
            # Based on epsilon-greedy strategy take the action
            action = agent.take_action(curr_state)
            # Comment to stop rendering the environment
            # If you don't render, you can speed things up
            env.render()
            # Step forward and receive next state and reward
            # done flag is set when the episode ends: either goal is reached or
            #       200 steps are done
            next_state, reward, done, _ = env.step(action)
            # This is where your NN/GP code should go
            # Create target vector
            # Train the network/GP
            # Update the policy
            if next_state[0] >= 0.5:
              print("---------------------------Target Reached---------------------------------------")
            next_state = next_state.reshape(1,2)
            agent.remember(curr_state, action, reward, next_state, done)
            # Record history
            episode_reward += reward
            # Current state for next step
            curr_state = next_state
            if done:
                # Record history
                reward_history.append(episode_reward)
                running_avg = get_running_avg(reward_history)
                running_averages.append(running_avg)
                if running_avg > -175 and running_avg < -150:
                    below_175.append(episode)
                if running_avg > -150 and running_avg < -100:
                    below_150.append(episode)
                if running_avg > -100:
                    below_100.append(episode)
                fig = plt.figure(1)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(running_averages,'ro')
                plt.xlabel('Episode')
                plt.ylabel('Running Average')
                plt.title('Running avg of latest 100 episodes')
                plt.pause(0.01)
                fig.canvas.draw()
                if(episode == NUM_EPISODES - 1):
                    plt.savefig("Running Averages.png")
                # You may want to plot periodically instead of after every episode
                # Otherwise, things will slow down
                fig2 = plt.figure(2)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(reward_history,'ro')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Per Episode')
                plt.pause(0.01)
                fig2.canvas.draw()
                if(episode == NUM_EPISODES - 1):
                    plt.savefig("Episode Rewards.png")
                
                print("Episode reward: ", episode_reward, "Episode Count: ", episode, "Epsilon: ", agent.epsilon)
                break
        # Train the model with batch size equal to 64
        agent.train_model(64)
    # Save the model for further use    
    agent.save_model("latest.h5")
    print("Below 175: ", below_175)
    print("Below 150: ", below_150)
    print("Below 100: ", below_100)
if __name__ == "__main__":
    main()
