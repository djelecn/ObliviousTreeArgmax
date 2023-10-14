import numpy as np
from tqdm import tqdm
import gym
# import matplotlib.pyplot as plt
from agents.ota import ODT_agent
from utils.datalog import Datalog



def run(model, datalog, env, epochs = 10, rounds_per_epoch = 20):
    avg_rewards = []
    true_avg_rewards = []
    epsilon = (np.arange(epochs,0,-1)**2)/epochs**2
    for epoch in tqdm(range(epochs)):
        ep_actions = []
        rews = []
        true_rews = []
        observation = env.reset()
        prev_reward = 0.1

        for rnd in range(rounds_per_epoch):
            
            env.render(mode = 'human')
            actions = model.act(observation.reshape(1,-1), random = (np.random.binomial(1,p=epsilon[epoch])))
            prev_observation = observation.copy()
            observation, reward, terminated, info = env.step(actions)
            # true_rews.append(_)
            rews.append(reward)
            if epoch>0:
                state_value = model.get_state_value(observation.reshape(1,-1))
            else:
                state_value = reward

            datalog.update(prev_observation, np.array([actions]), np.array([reward]),state_value, rollout_id = np.array([epoch]))
            if terminated:
                observation = env.reset()
                
            prev_reward = reward  
    #     action_list.append(ep_actions)
        avg_rewards.append(np.mean(rews))
        # true_avg_rewards.append(np.mean(true_rews))
        print(avg_rewards[-1])

        x = datalog.X
        y = datalog.y
        

        model.fit(x,y)
        datalog.reset()
    env.close()
    print(rews)
    # plt.plot(rews)
    # plt.show()

if __name__=='__main__':

    epochs = 10
    rounds = 2000

    tree_depth = 4

    env = gym.make("LunarLander-v2")
    # env.action_space.seed(42)

    state_ids = np.arange(0,8,1)
    action_format = (1,(0,4))

    model = ODT_agent(state_ids = state_ids, action_format=action_format, depth = tree_depth)
    datalog = Datalog()

    run(model, datalog, env, epochs, rounds)