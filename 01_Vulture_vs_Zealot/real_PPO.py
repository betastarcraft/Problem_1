#state size : 137, 39

from core.algorithm.PPO import PPOAgent
import numpy as np
import os
from datetime import datetime
from core.common.processor import Processor
from saida_gym.starcraft.vultureVsZealot import VultureVsZealot
from core.callbacks import DrawTrainMovingAvgPlotCallback
import saida_gym.envs.conn.connection_env as Config
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import math

import argparse
from core.common.util import OPS

parser = argparse.ArgumentParser(description='PPO Configuration for Avoid_Reaver')

parser.add_argument(OPS.NO_GUI.value, help='gui', type=bool, default=False)
parser.add_argument(OPS.MOVE_ANG.value, help='move angle', default=10, type=int)
parser.add_argument(OPS.MOVE_DIST.value, help='move dist', default=2, type=int)
parser.add_argument(OPS.GAMMA.value, help='gamma', default=0.99, type=float)
parser.add_argument(OPS.EPOCHS.value, help='Epochs', default=10, type=int)

args = parser.parse_args()

dict_args = vars(args)
post_fix = ''
for k in dict_args.keys():
    if k == OPS.NO_GUI():
        continue
    post_fix += '_' + k + '_' + str(dict_args[k])

# Hyper param
NO_GUI = dict_args[OPS.NO_GUI()]
NB_STEPS = 1000000
STATE_SIZE = 20
LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = dict_args[OPS.EPOCHS()]
NOISE = 0.1  # Exploration noise
GAMMA = dict_args[OPS.GAMMA()]
BUFFER_SIZE = 256
BATCH_SIZE = 64
HIDDEN_SIZE = 80
NUM_LAYERS = 3
ENTROPY_LOSS = 1e-3
LR = 1e-4  # Lower lr stabilises training greatly


def scale_velocity(v):
    return v / 6.4

def scale_coordinate(pos):
    if pos > 0:
        return 1 if pos > 320 else int(pos / 16) / 20
    else:
        return -1 if pos < -320 else int(pos / 16) / 20

def scale_angle(angle):
    return (angle - math.pi) / math.pi

def scale_cooldown(cooldown):
    return (cooldown + 1) / 15

def scale_vul_hp(hp):
    return hp / 80

def scale_zeal_hp(hp):
    return hp / 160

def scale_bool(boolean):
    return 1 if boolean else 0

def scale_pos(pos):
    return pos / 16

def scale_pos2(pos):
    return pos / 8


def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


# Reshape the reward in a way you want
def reward_reshape(reward):
    """ Reshape the reward
        Starcraft Env returns the reward according to following conditions.
        1. Invalid action : -0.1
        2. get hit : -1
        3. goal : 1
        4. others : 0

    # Argument
        reward (float): The observed reward after executing the action

    # Returns
        reshaped reward
    """

    if math.fabs(reward + 0.1) < 0.01:
        reward = -1
    elif reward == -1:
        reward = -10
    elif reward == 1:
        reward = 15
    elif reward == 0:
        reward = -0.1

    return reward

class ObsProcessor(Processor):
    def __init__(self, env):
        self.last_action = None
        self.success_cnt = 0
        self.cumulate_reward = 0

    def process_action(self, action):
        self.last_action = action
        return action

    def process_step(self, observation, reward, done, info):
        state_array = self.process_observation(observation)
        reward = reward_reshape(reward)
        self.cumulate_reward += reward

        if reward == 10:
            if self.cumulate_reward > 0:
                self.success_cnt += 1

            self.cumulate_reward = 0
            print("success_cnt = ", self.success_cnt)

        return state_array, reward, done, info

    def process_observation(self, observation,  **kwargs):
        state_array = np.zeros(STATE_SIZE) #state_size

        # 64 x 64
        # unwalkable : -1 walkable : 0 enemy pos : 1
        #

        tmp_idx = 0
        my_x = 0
        my_y = 0
        for idx, me in enumerate(observation.my_unit):
            my_x = me.pos_x
            my_y = me.pos_y
            state_array[tmp_idx + 0] = math.atan2(me.velocity_y, me.velocity_x) / math.pi
            state_array[tmp_idx + 1] = scale_velocity(math.sqrt((me.velocity_x) ** 2 + (me.velocity_y) ** 2))
            state_array[tmp_idx + 2] = scale_cooldown(me.cooldown)
            state_array[tmp_idx + 3] = scale_vul_hp(me.hp)
            state_array[tmp_idx + 4] = scale_angle(me.angle)
            state_array[tmp_idx + 5] = scale_bool(me.accelerating)
            state_array[tmp_idx + 6] = scale_bool(me.braking)
            state_array[tmp_idx + 7] = scale_bool(me.attacking)
            state_array[tmp_idx + 8] = scale_bool(me.is_attack_frame)
            #state_array[tmp_idx + 9] = self.last_action[0] / (env.action_space.n - 1)
            tmp_idx += 9
          #  for i, terrain in enumerate(me.pos_info):
          #      state_array[tmp_idx + i] = terrain.nearest_obstacle_dist / 320
         #   tmp_idx += len(me.pos_info)

        #tmp_idx = 9 + env.action_space.n - 1

        for idx, enemy in enumerate(observation.en_unit):
            state_array[tmp_idx + 0] = math.atan2(enemy.pos_y - my_y, enemy.pos_x - my_x) / math.pi
            state_array[tmp_idx + 1] = scale_coordinate(math.sqrt((enemy.pos_x - my_x) ** 2 + (enemy.pos_y - my_y) ** 2))
            state_array[tmp_idx + 2] = math.atan2(enemy.velocity_y, enemy.velocity_x) / math.pi
            state_array[tmp_idx + 3] = scale_velocity(math.sqrt((enemy.velocity_x) ** 2 + (enemy.velocity_y) ** 2))
            state_array[tmp_idx + 4] = scale_cooldown(enemy.cooldown)
            state_array[tmp_idx + 5] = scale_zeal_hp(enemy.hp + enemy.shield)
            state_array[tmp_idx + 6] = scale_angle(enemy.angle)
            state_array[tmp_idx + 7] = scale_bool(enemy.accelerating)
            state_array[tmp_idx + 8] = scale_bool(enemy.braking)
            state_array[tmp_idx + 9] = scale_bool(enemy.attacking)
            state_array[tmp_idx + 10] = scale_bool(enemy.is_attack_frame)
            tmp_idx += 11

        #self.accumulated_observation.append(state_array)

        return state_array

def build_actor(state_size, action_size, advantage, old_prediction):
    state_input = Input(shape=(state_size,))

    x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
    for _ in range(NUM_LAYERS - 1):
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)

    out_actions = Dense(action_size, activation='softmax', name='output')(x)

    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])

    return model


def build_actor_continuous(state_size, action_size, advantage, old_prediction):

    state_input = Input(shape=(state_size,))
    x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)

    for _ in range(NUM_LAYERS - 1):
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)

    out_actions = Dense(action_size, name='output', activation='tanh')(x)

    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])

    return model


def build_critic(state_size):

    state_input = Input(shape=(state_size,))
    x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
    for _ in range(NUM_LAYERS - 1):
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)

    out_value = Dense(1)(x)

    model = Model(inputs=[state_input], outputs=[out_value])

    return model


if __name__ == '__main__':
    training_mode = True
    load_model = False
    FILE_NAME = os.path.basename(__file__).split('.')[0] + "-" + datetime.now().strftime("%m%d%H%M%S")
    action_type = 0

    env = VultureVsZealot(version = 0,move_angle=dict_args[OPS.MOVE_ANG()], move_dist=dict_args[OPS.MOVE_DIST()], frames_per_step=16
                       , verbose=0, action_type=action_type, no_gui=NO_GUI)

    ACTION_SIZE = env.action_space.n
    print("액션크기:"+str(ACTION_SIZE))

    continuous = False if action_type == 0 else True

    # Build models
    actor = None
    ADVANTAGE = Input(shape=(1,))
    OLD_PREDICTION = Input(shape=(ACTION_SIZE,))

    if continuous:
        actor = build_actor_continuous(STATE_SIZE, ACTION_SIZE, ADVANTAGE, OLD_PREDICTION)
    else:
        actor = build_actor(STATE_SIZE, ACTION_SIZE, ADVANTAGE, OLD_PREDICTION)

    critic = build_critic(STATE_SIZE)

    agent = PPOAgent(STATE_SIZE, ACTION_SIZE, continuous, actor, critic, GAMMA, LOSS_CLIPPING, EPOCHS, NOISE, ENTROPY_LOSS,
                     BUFFER_SIZE,BATCH_SIZE, processor=ObsProcessor(env=env))

    agent.compile(optimizer=[Adam(lr=LR), Adam(lr=LR)], metrics=[ADVANTAGE, OLD_PREDICTION])

    cb_plot = DrawTrainMovingAvgPlotCallback(os.path.realpath('../../save_graph/' + FILE_NAME + '_'+post_fix + '.png'), 10, 5, l_label=['episode_reward'])

    agent.run(env, NB_STEPS, train_mode=training_mode, verbose=2, callbacks=[cb_plot], action_repetition=1, nb_episodes=1000)

    if training_mode:
        agent.save_weights(os.path.realpath("../../save_model"),"vulture_zealot_PPO"+post_fix)

    env.close()
