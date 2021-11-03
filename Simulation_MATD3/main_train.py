'''
The python code of the conference paper:
        AoI Aware Radio Resource Management of Autonomous Platoons Based on Multi Agent Deep
                                    Reinforcement Learning
Written by: Mohammad Parvini, Researcher at Tarbiat Modares University

'''
import numpy as np
import os
import scipy.io
import Classes.Environment_Platoon as ENV
from ddpg_torch import Agent
from Classes.buffer import ReplayBuffer
from global_critic import Global_Critic

# ################## SETTINGS ######################
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
print('------------- lanes are -------------')
print('up_lanes :', up_lanes)
print('down_lanes :', down_lanes)
print('left_lanes :', left_lanes)
print('right_lanes :', right_lanes)
print('------------------------------------')
width = 750 / 2
height = 1298 / 2
IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN
label = 'marl_model'
# ------------------------------------------------------------------------------------------------------------------ #
# simulation parameters:
# ------------------------------------------------------------------------------------------------------------------ #
size_platoon = 4
n_veh = 16  # n_platoon * size_platoon
n_platoon = int(n_veh / size_platoon)  # number of platoons
n_RB = 3  # number of resource blocks
n_S = 2  # decision parameter
Gap = 25 # meter
max_power = 30  # platoon leader maximum power in dbm ---> watt = 10^[(dbm - 30)/10]
V2I_min = 300  # minimum required data rate for V2I Communication in each time-slot
bandwidth = int(180000)
V2V_size = int((4000) * 8)
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------- characteristics related to the network -------- #
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
update_actor_interval = 2
noise = 0.2
# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 1024
A_fc2_dims = 512
# ------------------------------

tau = 0.005
env = ENV.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, size_platoon, n_RB,
                  V2I_min, bandwidth, V2V_size, Gap)
env.new_random_game()  # initialize parameters in env

n_episode = 500
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
def get_state(env, idx):
    """ Get state from the environment """

    V2I_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0

    V2V_abs = (env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60)/60.0

    V2I_fast = (env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[
        idx * size_platoon] + 10) / 35

    V2V_fast = (env.V2V_channels_with_fastfading[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1)), :]
                - env.V2V_channels_abs[idx * size_platoon, idx * size_platoon +
                                       (1 + np.arange(size_platoon - 1))].reshape(size_platoon - 1, 1) + 10) / 35

    Interference = (-env.Interference_all[idx] - 60) / 60

    AoI_levels = env.AoI[idx] / (int(env.time_slow / env.time_fast))

    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])

    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                           np.reshape(V2V_fast, -1), np.reshape(Interference, -1), np.reshape(AoI_levels, -1), V2V_load_remaining), axis=0)
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(get_state(env=env, idx=0))
n_output = 3  # channel selection, mode selection, power
# --------------------------------------------------------------
agents = []
for index_agent in range(n_platoon):
    print("Initializing agent", index_agent)
    agent = Agent(alpha, beta, n_input, tau, n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, batch_size, n_platoon, index_agent, noise)
    agents.append(agent)
memory = ReplayBuffer(memory_size, n_input, n_output, n_platoon)
print("Initializing Global critic ...")
global_agent = Global_Critic(beta, n_input, tau, n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_platoon, update_actor_interval, noise)
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
AoI_evolution = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
Demand_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2I_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2V_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
power_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
success_probability = np.zeros([n_episode_test, n_step_per_episode], dtype=np.float16)

AoI_total = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_reward_ = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_critics_loss_ = np.zeros([n_platoon+1, n_episode], dtype=np.float16)
# ------------------------------------------------------------------------------------------------------------------ #
if IS_TRAIN:
    # global_agent.load_models()
    # for i in range(n_platoon):
    #     agents[i].load_models()
    for i_episode in range(n_episode):
        done = False
        print("-------------------------------------------------------------------------------------------------------")
        record_reward = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
        record_AoI = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)

        env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
        env.individual_time_limit = env.time_slow * np.ones(n_platoon, dtype=np.float16)
        env.active_links = np.ones((int(env.n_Veh / env.size_platoon)), dtype='bool')
        if i_episode == 0:
            env.AoI = np.ones(int(n_platoon)) * 100

        if i_episode % 20 == 0: # Due to the dynamic behavior of fast-fading, we have deliberately increased the
                                # slow-fading freeze-time so as to let the RL agents conceive the fast-fading dynamics
                                # better; yet this line can be omitted, with the cost of longer convergence time
            env.renew_positions()  # update vehicle position
            env.renew_channel(n_veh, size_platoon)  # update channel slow fading
            env.renew_channels_fastfading()  # update channel fast fading

        # env.renew_positions()  # update vehicle position
        # env.renew_channel(n_veh, size_platoon)  # update channel slow fading
        # env.renew_channels_fastfading()  # update channel fast fading

        state_old_all = []
        for i in range(n_platoon):
            state = get_state(env=env, idx=i)
            state_old_all.append(state)

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_platoon, n_output], dtype=np.int)
            # receive observation
            for i in range(n_platoon):
                action = agents[i].choose_action(state_old_all[i])
                action = np.clip(action, -0.999, 0.999)
                action_all.append(action)

                action_all_training[i, 0] = ((action[0]+1)/2) * n_RB  # chosen RB
                action_all_training[i, 1] = ((action[1]+1)/2) * n_S  # Inter/Intra platoon mode
                action_all_training[i, 2] = np.round(np.clip(((action[2]+1)/2) * max_power, 1, max_power))  # power selected by PL

            # All the agents take actions simultaneously, obtain reward, and update the environment
            action_temp = action_all_training.copy()
            train_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand_R, V2V_success = \
                env.act_for_training(action_temp)
            for i in range(n_platoon):
                record_reward[i, i_step] = train_reward[i]
                record_AoI[i, i_step] = env.AoI[i]

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            # get new state
            for i in range(n_platoon):
                state_new = get_state(env, i)
                state_new_all.append(state_new)

            if i_step == n_step_per_episode - 1:
                done = True

            # taking the agents actions, states and reward
            memory.store_transition(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                                    global_reward, train_reward, np.asarray(state_new_all).flatten(), done)

            # agents take random samples and learn
            if memory.mem_cntr >= batch_size:
                states, actions, rewards_g, rewards_l, states_, dones = memory.sample_buffer(batch_size)

                global_agent.global_learn(agents, states, actions, rewards_g, rewards_l, states_, dones)

            # old observation = new_observation
            for i in range(n_platoon):
                state_old_all[i] = state_new_all[i]
            print("-----------------------------------")
            print('Episode:', i_episode)
            print('iteration:', i_step)
            print('Intra-platoon data rate is :\n', env.intraplatoon_rate)
            print('Inter-platoon data rate is :\n', env.interplatoon_rate)
            print('Age of Information :\n', env.AoI)
            print('remaining v2v payload :\n', env.V2V_demand)
            print('selected subchannels :\n', action_all_training[:,0])
            print('selected modes :\n', action_all_training[:,1])
            print('selected powers :\n', action_all_training[:,2])
            print('agents rewards :\n', train_reward)
            print('agents global rewards :\n', global_reward)

            for i in range(n_platoon):
                AoI_evolution[i, i_episode % 100, i_step] = platoon_AoI[i]
                Demand_total[i, i_episode % 100, i_step] = Demand_R[i]
                V2I_total[i, i_episode % 100, i_step] = C_rate[i]
                V2V_total[i, i_episode % 100, i_step] = V_rate[i]
                power_total[i, i_episode % 100, i_step] = action_temp[i, 2]
            success_probability[i_episode % 100, i_step] = V2V_success

        record_reward_[:, i_episode] = np.mean(record_reward, axis=1)
        AoI_total[:, i_episode] = np.mean(record_AoI, axis=1)
        record_critics_loss_[0, i_episode] = np.mean(np.asarray(global_agent.Global_Loss))
        global_agent.Global_Loss = []
        for i in range(n_platoon):
            record_critics_loss_[i+1, i_episode] = np.mean(np.asarray(agents[i].local_critic_loss))
            agents[i].local_critic_loss = []

        if i_episode % 50 == 0:
            global_agent.save_models()
            for i in range(n_platoon):
                agents[i].save_models()

    print('Training Done. Saving models...')
    current_dir = os.path.dirname(os.path.realpath(__file__))

    reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')
    AoI_path = os.path.join(current_dir, "model/" + label + '/AoI.mat')
    AoI_evolution_path = os.path.join(current_dir, "model/" + label + '/AoI_evolution.mat')
    Demand_path = os.path.join(current_dir, "model/" + label + '/demand.mat')
    V2I_path = os.path.join(current_dir, "model/" + label + '/V2I.mat')
    V2V_path = os.path.join(current_dir, "model/" + label + '/V2V.mat')
    power_path = os.path.join(current_dir, "model/" + label + '/power.mat')
    success_path = os.path.join(current_dir, "model/" + label + '/success.mat')
    critic_loss_path = os.path.join(current_dir, "model/" + label + '/critic_loss.mat')

    scipy.io.savemat(reward_path, {'reward': record_reward_})
    scipy.io.savemat(AoI_path, {'AoI': AoI_total})
    scipy.io.savemat(AoI_evolution_path, {'AoI_evolution': AoI_evolution})
    scipy.io.savemat(Demand_path, {'demand': Demand_total})
    scipy.io.savemat(V2I_path, {'V2I': V2I_total})
    scipy.io.savemat(V2V_path, {'V2V': V2V_total})
    scipy.io.savemat(power_path, {'power': power_total})
    scipy.io.savemat(success_path, {'success': success_probability})
    scipy.io.savemat(critic_loss_path, {'critic_loss': record_critics_loss_})

    global_agent.save_models()
    for i in range(n_platoon):
        agents[i].save_models()