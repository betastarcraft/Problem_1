from saida_gym.starcraft.avoidReavers import AvoidReavers
## gym 환경 import VultureVsZealot * Saida RL library

from collections import deque
import numpy as np
import random
import os
import math
import pickle
import time
## 파이썬 내부 함수들.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical ## 분포 관련
from tensorboardX import SummaryWriter
## pytorch 함수들, 마지막 tensorboardX만 tensorboard 사용을 원한다면. pip install tensorboardX로 설치하면 끝.

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size,512) ## input state
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,action_size) ## output each action

    def forward(self, x, soft_dim):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        prob_each_actions = F.softmax(self.fc4(x),dim=soft_dim) ## NN에서 각 action에 대한 확률을 추정한다.

        return prob_each_actions

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size,512) ## input state
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,1) ## output value

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.fc4(x)

        return value

## -- 여기부터 -- ##
def scale_velocity(v):
    return v


def scale_angle(angle):
    return (angle - math.pi) / math.pi


def scale_pos(pos):
    return pos / 16


def scale_pos2(pos):
    return pos / 8

def rearrange_State(observation, state_size):
    
    if len(observation.my_unit) > 0:
            s = np.zeros(state_size)
            me = observation.my_unit[0]
            # Observation for Dropship
            s[0] = scale_pos2(me.pos_x)  # X of coordinates
            s[1] = scale_pos2(me.pos_y)  # Y of coordinates
            s[2] = scale_pos2(me.pos_x - 320)  # relative X of coordinates from goal
            s[3] = scale_pos2(me.pos_y - 320)  # relative Y of coordinates from goal
            s[4] = scale_velocity(me.velocity_x)  # X of velocity
            s[5] = scale_velocity(me.velocity_y)  # y of coordinates
            s[6] = scale_angle(me.angle)  # Angle of head of dropship
            s[7] = 1 if me.accelerating else 0  # True if Dropship is accelerating

            # Observation for Reavers
            for ind, ob in enumerate(observation.en_unit):
                s[ind * 8 + 8] = scale_pos2(ob.pos_x - me.pos_x)  # X of relative coordinates
                s[ind * 8 + 9] = scale_pos2(ob.pos_y - me.pos_y)  # Y of relative coordinates
                s[ind * 8 + 10] = scale_pos2(ob.pos_x - 320)  # X of relative coordinates
                s[ind * 8 + 11] = scale_pos2(ob.pos_y - 320)  # Y of relative coordinates
                s[ind * 8 + 12] = scale_velocity(ob.velocity_x)  # X of velocity
                s[ind * 8 + 13] = scale_velocity(ob.velocity_y)  # Y of velocity
                s[ind * 8 + 14] = scale_angle(ob.angle)  # Angle of head of Reavers
                s[ind * 8 + 15] = 1 if ob.accelerating else 0  # True if Reaver is accelerating

    return s

## -- 여기까지는 튜토리얼 코드의 state 전처리와 scale부분 똑같습니다. -- ##

def reward_reshape(next_state, reward, damage_count): ## reward 재정의 

    """ Reshape the reward
        Starcraft Env returns the reward according to following conditions.
        1. Invalid action : -0.1
        2. get hit : -1
        3. goal : 100
        4. others : 0

    # Argument
        reward (float): The observed reward after executing the action

    # Returns
        reshaped reward
    elif reward == 0.0: ## 아무것도 안함.
        reward = -0.01

    """
    if reward == -1.0: ##  드랍쉽이 맞음.
        reward = -1
        damage_count += 1
    elif reward == 100.0: ## goal에 도착
        reward = 2
    elif reward == 0.0: ## 아무것도 안함.
        reward = 0
    else: ## 맵 밖으로 나갔을때
        reward = -1
        damage_count += 1

    return reward, damage_count


def GAE(critic, states, rewards, masks):## GAE Generalized Advantage Estimator 논문의 수식을 참고해주세요.

    rewards = torch.Tensor(rewards) 
    masks = torch.Tensor(masks) ## 여기서 masks가 게임이 진행중인지, 종료 됬는지 구별하는 값이므로, 죽었을때의 trajectory 기록이면 0으로 아무것도 계산하지 않습니다.
    states = torch.Tensor(states)

    V_Target_G = torch.zeros_like(rewards) ## L^VF를 구할때 사용되는 V^target 값입니다. G를 붙인 이유는 reinforce 논문에서 G로 표현되서 같이 표기해 봤습니다.
    Advantages = torch.zeros_like(rewards) ## 구해진 advantage 값을 담기 위한 배열.

    gamma = 0.99 ## 감마
    lmbda = 0.95 ## 람다
    
    next_value = 0
    Return = 0
    advantage = 0

    values = critic(torch.Tensor(states)) ## delta 계산에 사용되는 V(St)
    
    for t in reversed(range(0,len(rewards))): ## 뒤에서부터 계산하는 이유는 알고리즘상 효율성 때문입니다. 
        
        ##  L^VF를 구할때 사용되는 V^target계산
        Return = rewards[t] + gamma * Return * masks[t] ## Return이 논문 수식의 R을 나타냄. 여기서 mask를 잘 생각해보면, 여러 게임의 trajactory지만, 순서대로 계산을 해도 문제 없음을 알 수 있습니다.
        V_Target_G[t] = Return ## return이 critic loss에 쓰이는 V^target에 해당하는 값이다.

        ## advantage 계산
        delta = rewards[t] + gamma * next_value * masks[t] - values.data[t] ## 여기서 mask를 잘 생각해보면, 여러 게임의 trajactory지만, 순서대로 계산을 해도 문제 없음을 알 수 있습니다.
        next_value = values.data[t] ## 뒤에서 부터 계산하니 당연히 마지막 데이터는 next_value가 0입니다.
        
        advantage = delta + gamma* lmbda * advantage * masks[t]
        Advantages[t] = advantage    
        
    return V_Target_G, Advantages
    
def surrogate_loss(actor, old_policy, Advantages, states, actions):
    policy = torch.zeros_like(old_policy) ## policy를 담기위한 0 배열
    
    prob_each_actions = actor(torch.Tensor(states), soft_dim=1) ##actor로 부터 softmax화 시킨 각 action의 확률 값을 받습니다.
    distribution = Categorical(prob_each_actions) ## Categorical 함수를 이용해 하나의 분포도로 만들어줍니다.
    entropies = distribution.entropy() ## 논문에 pi_seta에서 St에 따른 entropy bonus여서 action들의 분포에 대한 entropy로 구했습니다.
    
    for t, act in enumerate(actions): ## enumerate는 자동으로 array에 index와 원소를 매칭해 반환해 줍니다. 0번부터 시작함.
        policy[t] = prob_each_actions[t][act].item() ## 실제 action에 해당하는 new policy를 담아 줍니다.
    
    ratio = torch.exp(torch.log(policy) - torch.log(old_policy)) ## 원래는 policy_pi/policy_old_pi 식인데 = exp(log(policy_pi)-log(policy_old_pi)) 로 변경한것. 정확한 이유는 모르지만, 더 효율적이라 이렇게 쓴다고함.
    
    ratio_A = ratio * Advantages

    return ratio, ratio_A, entropies

def train(writer, n_iter, actor, critic, trajectories, actor_optimizer, critic_optimizer, T_horizon, batch_size, epoch):
    
    c_1 = 1 ## coefficient c1
    c_2 = 0.01 ## coefficient c2
    eps = 0.2 ## clipped에 사용되는 epsilon
    
    trajectories = np.array(trajectories) ##deque인 type을 np.array로 바꿔 줍니다.
    states = np.vstack(trajectories[:, 0]) ## vstack인 이유는 state는 한 data당 4개의 정보가 들어있기 때문입니다.
    actions = list(trajectories[:, 1]) ## 나머지 action, reward, mask는 1개의 상수 값이기때문에 그냥 list화 해줍니다.
    rewards = list(trajectories[:, 2])  
    masks = list(trajectories[:, 3]) 
    old_policies = list(trajectories[:, 4])

    
    V_Target_G, Advantages = GAE(critic, states, rewards, masks) ## random mini batch 전에 미리 구해놔야함. 아무래도 t 순서대로 계산되다보니 미리 구하는 게 편함.
    

    r_batch = np.arange(len(states)) ## random mini batch를 위해 저장된 states 만큼 숫자순서대로 배열을 만듬. 1~
    
    for i in range(epoch): ## 전체 데이터를 몇번 반복해 학습할지. 10으로 해둠. 즉 trajactories에 2048개 데이터가 저장되어있다면, 2048*10번 함.
        np.random.shuffle(r_batch) ## 1~ 들어있는 숫자들을 무작위로 섞음.
        
        for j in range(T_horizon//batch_size): ##2048/64  0 ~ 31 
            mini_batch = r_batch[batch_size * j: batch_size * (j+1)] ## batch 크기 간격으로 앞에서부터 자름.
            
            mini_batch = torch.LongTensor(mini_batch) ## LongTensor가 찾아보니 정수형이였음.
            
            states_b = torch.Tensor(states)[mini_batch] ## 선택된 mini_batch index에 맞게 batch 크기 만큼 데이터를 선별함.
            actions_b = torch.LongTensor(actions)[mini_batch]
                 
            
            V_Target_G_b = torch.Tensor(V_Target_G)[mini_batch].detach() ## Target은 변하면 안되기 때문에 detach()를 해준다.
            
            ## critic loss
            mse_loss = torch.nn.MSELoss() ## mse_loss 정의
            
            values = critic(states_b).squeeze(1) ## V_Target_G_b = [64] 인데 values = [64,1] 이여서 dimension을 맞춰주기위해 values=[64]로 만듬.
            critic_loss = mse_loss(values,V_Target_G_b) ## mean square loss라서 이 값 자체가 mean임.

            ## critic loss end.
            
            ## actor loss
                        
            old_policies_b = torch.Tensor(old_policies)[mini_batch].detach() ## old_policy 값은 backpropagation에 반영되지 않도록 detach 해준다.
            
            Advantages_b = torch.Tensor(Advantages)[mini_batch]
            Advantages_b = Advantages_b.unsqueeze(1) ## 차원을 맞추기위해 1추가.
                        
            ratio, L_CPI, entropies= surrogate_loss(actor, old_policies_b, Advantages_b, states_b, actions_b) ## ratio = policy_pi/old_policy_pi, L_CPI = ratio * Advantage
            
            entropies= entropies.mean() ## entropy도 평균 구함.
            
            clipped_surrogate = torch.clamp(ratio,1-eps,1+eps) * Advantages_b ## clip 부분

            actor_loss = -torch.min(L_CPI.mean(),clipped_surrogate.mean()) ## 최종적으로 ratio * Advantage 평균, clip 평균 중 작은 값을 actor_loss로 한다. -인 이유는 ascent 여서.
            ## actor loss end.
            
            L_CLIP_VF_S= actor_loss + c_1 * critic_loss - c_2 * entropies ## 논문의 algorithm 수식이다. L^CLIP+VF+S 식이다. + - 가 논문에서랑 반대인 이유는 gradient ascent를 적용해야 하기 때문이다.

            critic_optimizer.zero_grad()
            L_CLIP_VF_S.backward(retain_graph=True) ## retain_graph가 True 이유는 총 3개의 항이 하나의 L_CLIP_VF_S식을 이루므로 뒤에 있을 actor loss를 계산할때도 반영하기 위해.
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            L_CLIP_VF_S.backward()
            actor_optimizer.step()
            
    writer.add_scalar('loss_NN', L_CLIP_VF_S.item(), n_iter) ## 텐서보드에 epoch 10이 끝나면 loss 값 기록.
    
def main():
    writer = SummaryWriter('C:/SAIDA_RL/python/saida_agent_example/avoidReaver/save_ppo/log_file') ## 텐서보드 로그 디렉토리지정.
    
    env = AvoidReavers(move_angle=15, move_dist=2, frames_per_step=8, verbose=0, action_type=0) ## * Saida RL library 환경 불러오기. frames_per_step 기본값은 16입니다.
    
    print_interval = 10 ## 몇 n_iter 마다 출력할 것인지.
    
    load = False ## 저장된 모델 불러오기
    episode = 0
    batch_size = 64
    epoch = 10
    learning_rate=0.000003
    T_horizon = 2048 ## 얼마큼의 trajectory를 수집할건지. 여러 actor를 한다면 여러 actor가 수집한 총 trajectory 갯수.
    
    torch.manual_seed(500) ## 혹시 모를 랜덤시드 고정.
    
    state_size = 32 ## 드랍쉽:8 리버:8 * 3 =24 8+24=32
    action_size = env.action_space.n ## 환경에서 받아온 action.
    

    actor = Actor(state_size, action_size) ## actor 생성
    critic = Critic(state_size) ## critic 생성
    
    
    if load: ## True면 저장된 모델을 불러옴.
        actor.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidReaver/save_ppo/','ppo_actor_'+str(episode)+'.pkl')))
        critic.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidReaver/save_ppo/','ppo_critic_'+str(episode)+'.pkl')))
        
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate) ## actor에 대한 optimizer Adam으로 설정하기.
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate) ## critic에 대한 optimizer Adam으로 설정하기.
    
    score = 0.0
    
    for n_iter in range(10000): ## 반복
        trajectories = deque() ## (s,a,r,done 상태) 를 저장하는 history or trajectories라고 부름. 학습을 위해 저장 되어지는 경험 메모리라고 보면됨.
                                ## PPO는 on-policy라 한번 학습하고 버림.
        step = 0
        
        while step < T_horizon: ## trajectory를 T_horizon 이상으로 수집.
            state = env.reset() ## * Saida RL library 게임 환경 초기화 및 초기 state 받기    
            state = rearrange_State(state, state_size)
            
            episode+=1
            damage_count = 0
            temp_score = 0.0
            
            while True: ## history 얼마나 모을지
                ep_start = time.time() ## 학습시간 측정용.
                
                step+=1
                
                prob_each_actions = actor(torch.Tensor(state), soft_dim=0) ##actor로 부터 softmax화 시킨 각 action의 확률 값을 받습니다.
                
                distribution = Categorical(prob_each_actions) ## pytorch categorical 함수는 array에 담겨진 값들을 확률로 정해줍니다.
                action = distribution.sample().item() ## ex) prob_each_actions = [0.25, 0.35, 0.1, 0.3] 이라면 각각의 인덱스가 25%, 35%, 10%, 30%
                                            ## 0,1,2,3 값 중 하나가 위 확률에 따라 선택됨. 
                old_policy = prob_each_actions[action].item() ## 실제 선택된 action의 확률 값만 저장해줌.
                
                next_state, reward, done, info = env.step([action]) ## * Saida RL library 분포에서 선택된 action이 다음 step에 들어감.
                next_state = rearrange_State(next_state, state_size)
                
                reward, damage_count = reward_reshape(next_state, reward, damage_count) ## damage_count = 0이면 한대도 안맞았다는 뜻.
                
                mask = 0 if done else 1 ## 게임이 종료됬으면, done이 1이면 mask =0 그냥 생존유무 표시용.
                
                trajectories.append((state, action, reward, mask, old_policy))
                
                state = next_state ## current state를 이제 next_state로 변경

                score += reward ## reward 갱신.
               
                temp_score += reward ## 출력용 변수
                              
                if done: ## 죽었다면 게임 초기화를 위한 반복문 탈출
                    if damage_count == 0: ## 한번도 리버에게 맞지 않고 goal에 도달하면 log 출력하고 모델 저장.
                        print("[ clear Objective ] Clear_reward_score: ",temp_score," clear_time",time.time()-ep_start)
                        torch.save(actor.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidReaver/save_ppo/','clear_ppo_actor_'+str(episode)+'.pkl'))
                        torch.save(critic.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidReaver/save_ppo/','clear_ppo_critic_'+str(episode)+'.pkl'))
                    else: ## goal에 들어가면, step과 reward score 출력.
                        print("step: ", step, "per_episode_score: ",temp_score)
                    temp_score = 0.0
                    break
            
            if episode%10==0 and episode !=0: ## 10 episode마다 평균 reward 값과 episode를 텐서보드에 띄움.
                writer.add_scalar('score/episode_10', float(score//print_interval), episode)
                score = 0.0
                
        actor.train() ## model actor를 train모드로 변경
        critic.train() ## critic actor를 train모드로 변경
            
        train(writer, n_iter, actor, critic, trajectories, actor_optimizer, critic_optimizer, T_horizon, batch_size, epoch) ## train 함수.

        if n_iter%print_interval==0 and n_iter!=0: ## 10 n_iter마다 score 출력 및 score 초기화.
            print("# n_iter :{}, avg score : {:.1f}".format(n_iter, score//print_interval))
            
        
    env.close() ## * Saida RL library 만약 정해둔 n_iter만큼 학습이 끝났다면 환경 종료.
    
if __name__ == '__main__':
    main()


