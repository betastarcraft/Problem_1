from saida_gym.starcraft.vultureVsZealot import VultureVsZealot
## gym 환경 import VultureVsZealot

from collections import deque
import numpy as np
import random
import os
import math
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical ## 분포 관련
from tensorboardX import SummaryWriter

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size,128) ## input state
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,action_size) ## output each action

    def forward(self, x, soft_dim):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        prob_each_actions = F.softmax(self.fc4(x),dim=soft_dim) ## NN에서 각 action에 대한 확률을 추정한다.

        return prob_each_actions

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

def rearrange_State(observation, state_size, env):
    state_arr = deque(maxlen=state_size)

    my_x = 0
    my_y = 0
    if observation.my_unit:
        for idx, me in enumerate(observation.my_unit): ## 9
            my_x = me.pos_x
            my_y = me.pos_y
            state_arr.append(math.atan2(me.velocity_y, me.velocity_x) / math.pi)
            state_arr.append(scale_velocity(math.sqrt((me.velocity_x) ** 2 + (me.velocity_y) ** 2)))
            state_arr.append(scale_cooldown(me.cooldown))
            state_arr.append(scale_vul_hp(me.hp))
            state_arr.append(scale_angle(me.angle))
            state_arr.append(scale_bool(me.accelerating))
            state_arr.append(scale_bool(me.braking))
            state_arr.append(scale_bool(me.attacking))
            state_arr.append(scale_bool(me.is_attack_frame))
            for i, terrain in enumerate(me.pos_info): ##12
                state_arr.append(terrain.nearest_obstacle_dist / 320)
    else:
        for _ in range(state_size - 11):
            state_arr.append(0)

    if observation.en_unit:
        for idx, enemy in enumerate(observation.en_unit): ## 11
            state_arr.append(math.atan2(enemy.pos_y - my_y, enemy.pos_x - my_x) / math.pi)
            state_arr.append(scale_coordinate(math.sqrt((enemy.pos_x - my_x) ** 2 + (enemy.pos_y - my_y) ** 2)))
            state_arr.append(math.atan2(enemy.velocity_y, enemy.velocity_x) / math.pi)
            state_arr.append(scale_velocity(math.sqrt((enemy.velocity_x) ** 2 + (enemy.velocity_y) ** 2)))
            state_arr.append(scale_cooldown(enemy.cooldown))
            state_arr.append(scale_zeal_hp(enemy.hp + enemy.shield))
            state_arr.append(scale_angle(enemy.angle))
            state_arr.append(scale_bool(enemy.accelerating))
            state_arr.append(scale_bool(enemy.braking))
            state_arr.append(scale_bool(enemy.attacking))
            state_arr.append(scale_bool(enemy.is_attack_frame))
    else:
        for _ in range(11):
            state_arr.append(0)
  

    return state_arr

def reward_reshape(state, next_state, reward, done):

    KILL_REWARD = 10
    DEAD_REWARD = -10
    DAMAGED_REWARD = -4
    HIT_REWARD = 2

    if done:
        if reward > 0: ## env에서 반환된 reward가 1 이면, 질럿을 잡음.
            reward = KILL_REWARD
            if next_state[3] == 1.0 and next_state[-6] == 0:
                reward+=5
                
            return reward
            # 잡은  경우
        else: ## 게임이 종료되고 -1 값을 받게 된다면, 
            reward = DEAD_REWARD
            return reward
    else: ## 게임이 종료되지 않았다면,
        my_pre_hp = state[3]
        my_cur_hp = next_state[3]
        
        en_pre_hp = state[-6]
        en_cur_hp = next_state[-6]
        
        if my_pre_hp - my_cur_hp > 0: ## 벌쳐가 맞아 버렸네 ㅠㅠ
            reward += DAMAGED_REWARD
        if en_pre_hp - en_cur_hp > 0: ## 질럿을 때려 버렸네 ㅠㅠ
            reward += HIT_REWARD
        
        ## 벌쳐가 맞고, 질럿도 때리는 2가지 동시 case가 있을 거 같아. reward를 +=을 했고 각각 if문으로 처리했습니다.
    
    return reward

def main():
    
    load = True
    episode = 45670
    
    env = VultureVsZealot(version=0, frames_per_step=12, action_type=0, move_angle=20, move_dist=3, verbose=0, no_gui=True
                          ,auto_kill=False) ## clear frame = 12 move = 45 move_dist = 6
    print_interval = 10
    
    learning_rate=0.00003
    torch.manual_seed(500)
    
    state_size = 38
    
    action_size= 19
    

    actor = Actor(state_size, action_size)
    
    
    if load: ## 경로를 model 파일 경로+ model 파일 이름으로 변경해주세요. 저는 원래 episode 번호로 구분했습니다.
        actor.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_ppo3_clear/','clear_ppo_actor_'+str(episode)+'.pkl')))
        
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    
    episode = 0
    clear_cnt=0
    for n_iter in range(1000):
        step = 0
        
        state = env.reset()
            
        state = rearrange_State(state, state_size, env)
        episode+=1
        temp_score = 0.0
        while True:
                
            prob_each_actions = actor(torch.Tensor(state), soft_dim=0)
   
            distribution = Categorical(prob_each_actions)
                
            action = distribution.sample().item() 
                
            next_state, reward, done, info = env.step([action])
            next_state = rearrange_State(next_state, state_size, env)
                
            reward = reward_reshape(state, next_state, reward, done) 
                
            mask = 0 if done else 1 
       
            state = next_state
               
            temp_score += reward 
                
            if next_state[3] == 1.0 and next_state[-6] == 0:
                clear_cnt+=1
                print("clear: ",next_state[3],next_state[-6],"clear_score: ",temp_score, "clear_cnt: ", clear_cnt," / ", n_iter+1)
                
            if done: 
                print("step: ", step, "per_episode_score: ",temp_score)
                
                break

    print("clear count: ",clear_cnt," percent: ",(clear_cnt/n_iter))  
    env.close()
    
if __name__ == '__main__':
    main()
