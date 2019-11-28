from saida_gym.starcraft.avoidObservers import AvoidObservers

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
    def __init__(self):
        super(Actor, self).__init__()
        self.scur_cnn1 = nn.Sequential(
            nn.Conv2d(4,16,kernel_size=7,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3)
            )
        self.scur_cnn2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4), ## maxpool size가 너무 커서 정보들이 많이 없어졌을 수도 있습니다. conv의 filter, channel, maxpool은 적절히 조절해주세요.
            nn.Flatten()
            ) ## 32채널 4*4 로 맞춰둠.
       
        
        self.ob_cnn1 = nn.Sequential(
            nn.Conv2d(4,16,kernel_size=8,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.ob_cnn2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
            ) ## 32채널 4*4 로 맞춰둠.
        
        self.fc1 = nn.Linear((5*5*32)+(4*4*32),512) ## input state, 512+512 = 1024 , 512는 DQN논문에서 input의 반절을 fc의 hidden으로 만들어서 따라함.
        self.fc2 = nn.Linear(512,19)

    def forward(self, scurge_state, observer_state, soft_dim):
        x = self.scur_cnn1(scurge_state)
        x = self.scur_cnn2(x) ## scurge 이미지를 처리하는 cnn
       
        
        y = self.ob_cnn1(observer_state) ## observer 이미지를 처리하는 cnn 
        y = self.ob_cnn2(y)
        
        concat = torch.cat((x,y),1) ## 행 기준으로 concatenate 한다. (1,512) (1,512) = (1,1024) 
        
        out = torch.tanh(self.fc1(concat))        
        prob_each_actions = F.softmax(self.fc2(out),dim=soft_dim) ## NN에서 각 action에 대한 확률을 추정한다.

        return prob_each_actions

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.scur_cnn1 = nn.Sequential(
            nn.Conv2d(4,16,kernel_size=7,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3)
            )
        self.scur_cnn2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4), ## maxpool size가 너무 커서 정보들이 많이 없어졌을 수도 있습니다. conv의 filter, channel, maxpool은 적절히 조절해주세요.
            nn.Flatten()
            ) ## 32채널 4*4 로 맞춰둠.
       
        
        self.ob_cnn1 = nn.Sequential(
            nn.Conv2d(4,16,kernel_size=8,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.ob_cnn2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
            ) ## 32채널 4*4 로 맞춰둠.
        
        self.fc1 = nn.Linear((5*5*32)+(4*4*32),512) ## input state, 512+512 = 1024 , 512는 DQN논문에서 input의 반절을 fc의 hidden으로 만들어서 따라함.
        self.fc2 = nn.Linear(512,1)## output value
        
    def forward(self, scurge_state, observer_state):
        x = self.scur_cnn1(scurge_state)
        x = self.scur_cnn2(x) ## scurge 이미지를 처리하는 cnn
        
        y = self.ob_cnn1(observer_state) ## observer 이미지를 처리하는 cnn 
        y = self.ob_cnn2(y)
        
        concat = torch.cat((x,y),1) ## 행 기준으로 concatenate 한다. (1,512) (1,512) = (1,1024) 
        
        out = torch.tanh(self.fc1(concat))  
        value = self.fc2(out)

        return value


def state_to_image(observation, last_action=None, verbose=False): ## image 처럼 mapping 하는 부분은 그대로 가져왔습니다.
    LOCAL_OBSERVABLE_TILE_SIZE = 10

    # scurge's map
    map_of_scurge = np.zeros(shape=(64, 64))

    me_x = observation.my_unit[0].pos_x
    me_y = observation.my_unit[0].pos_y

    me_x_t = np.clip(int(me_x/32), 0, 64)
    me_y_t = np.clip(int(me_y/32), 0, 64)
    if verbose:
        print('my location:', [me_x_t, me_y_t])

    # Safe zone : left-top (896, 1888) right-bottom (1056, 2048) with additional (marginal) space
    for x in range(int(896/32), int(1056/32)): # 28~33
        for y in range(int(1888/32), int(2048/32)): # 59~64
            map_of_scurge[y][x] = -1  # masking safe zone

    # Goal line : left-top (0, 0) right-bottom (2048, 64) with additional (marginal) space
    for x in range(int(0/32), int(2048/32)): # 0~64
        for y in range(int(0/32), int(64/32)): # 0~2
            map_of_scurge[y][x] = -1  # masking safe zone

    # masking observer's location
    map_of_scurge[me_y_t][me_x_t] = 1
    #map_of_scurge = np.expand_dims(map_of_scurge, -1)

    # observer map
    map_of_observer = np.zeros(shape=(LOCAL_OBSERVABLE_TILE_SIZE*2+1, LOCAL_OBSERVABLE_TILE_SIZE*2+1))
    map_of_observer[LOCAL_OBSERVABLE_TILE_SIZE, LOCAL_OBSERVABLE_TILE_SIZE] = -1

    for ob in observation.en_unit:
        en_x_t = ob.pos_x / 32
        en_y_t = ob.pos_y / 32

        # scurge를 중앙에 두기 위해
        rel_x = int(en_x_t - me_x_t) + LOCAL_OBSERVABLE_TILE_SIZE
        rel_y = int(en_y_t - me_y_t) + LOCAL_OBSERVABLE_TILE_SIZE

        rel_x = np.clip(rel_x, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
        rel_y = np.clip(rel_y, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
        if verbose:
            print('enemy location:', [en_x_t, en_y_t], '(relevant:', [rel_x, rel_y], ')')

        map_of_observer[rel_y][rel_x] = map_of_observer[rel_y][rel_x] + 1  # if two or more observers are duplicated, we use sum

    # display out of map where scurge can't go based on current location of scurge
    scurge_out_of_map_left = me_x_t - LOCAL_OBSERVABLE_TILE_SIZE
    scurge_out_of_map_right = me_x_t + LOCAL_OBSERVABLE_TILE_SIZE
    scurge_out_of_map_up = me_y_t - LOCAL_OBSERVABLE_TILE_SIZE
    scurge_out_of_map_down = me_y_t + LOCAL_OBSERVABLE_TILE_SIZE

    if scurge_out_of_map_left < 0:
        map_of_observer[:, 0:-scurge_out_of_map_left] = -2
    if scurge_out_of_map_right > 64:
        map_of_observer[:, -(scurge_out_of_map_right-64):] = -2
    if scurge_out_of_map_up < 0:
        map_of_observer[0:-scurge_out_of_map_up,:] = -2
    if scurge_out_of_map_down > 64:
        map_of_observer[-(scurge_out_of_map_down-64):,:] = -2

    #map_of_observer = np.expand_dims(map_of_observer, -1)

    if not last_action:
        last_action = np.full((64, 64), -1)
    else:
        last_action = np.full((64, 64), last_action)
        
    if verbose:
        print(map_of_scurge.shape)
        print(map_of_observer.shape)
        print(last_action.shape)
    
    return [map_of_scurge, map_of_observer, last_action]

def reward_shape(highest, state, next_state, reward, done):
        
    if done: ## 게임이 끝났으면 
        if reward == -5.0: ## 죽어서 끝난경우
            return -100, highest
        else: ## 클리어 한 경우.
            print("-------------- Clear!!! ---------------------")
            return 50, highest
    else:
        if highest > next_state.my_unit[0].pos_y: ## 신기록 세우면 + 20, highest 기록 변경.
            highest = next_state.my_unit[0].pos_y
            return 15, highest
        """
        elif (state.my_unit[0].pos_y - next_state.my_unit[0].pos_y) > 0: ## 이전 state보다 더 위로 올라갔으면 이동 거리만큼 + reward
            return (state.my_unit[0].pos_y - next_state.my_unit[0].pos_y) * 0.1, highest
        elif (state.my_unit[0].pos_y - next_state.my_unit[0].pos_y) < 0: ## 이전 state보다 더 아래로 내려가면 이동 거리만큼 - reward
            return (state.my_unit[0].pos_y - next_state.my_unit[0].pos_y) * 0.1, highest
        """
    return 0, highest ## 왼쪽 오른쪽 이동이면 그냥 0

def GAE(critic, scurge_states, observer_states, rewards, masks):## GAE Generalized Advantage Estimator 논문의 수식을 참고해주세요.

    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks) ## 여기서 masks가 게임이 진행중인지, 종료 됬는지 구별하는 값이므로, 죽었을때의 trajectory 기록이면 0으로 아무것도 계산하지 않습니다.
    

    V_Target_G = torch.zeros_like(rewards) ## L^VF를 구할때 사용되는 V^target 값입니다. G를 붙인 이유는 reinforce 논문에서 G로 표현되서 같이 표기해 봤습니다.
    Advantages = torch.zeros_like(rewards) ## 구해진 advantage 값을 담기 위한 배열.

    gamma = 0.99 ## 감마
    lmbda = 0.95 ## 람다
    
    next_value = 0
    Return = 0
    advantage = 0

    values = critic(torch.Tensor(scurge_states).cuda(), torch.Tensor(observer_states).cuda()) ## delta 계산에 사용되는 V(St)
    
    for t in reversed(range(0,len(rewards))): ## 뒤에서부터 계산하는 이유는 알고리즘상 효율성 때문입니다. 
        
        ##  L^VF를 구할때 사용되는 V^target계산
        Return = rewards[t] + gamma * Return * masks[t] ## Return이 논문 수식의 R을 나타냄. 여기서 mask를 잘 생각해보면, 여러 게임의 trajactory지만, 순서대로 계산을 해도 문제 없음을 알 수 있습니다.
        V_Target_G[t] = Return ## return이 critic loss에 쓰이는 V^target에 해당하는 값이다.

        ## advantage 계산
        delta = rewards[t] + gamma * next_value * masks[t] - values.data[t] ## 여기서 mask를 잘 생각해보면, 여러 게임의 trajactory지만, 순서대로 계산을 해도 문제 없음을 알 수 있습니다.
        next_value = values.data[t] ## 뒤에서 부터 계산하니 당연히 마지막 데이터는 next_value가 0입니다.
        
        advantage = delta + gamma* lmbda * advantage * masks[t]
        Advantages[t] = advantage
        
    Advantages = (Advantages - Advantages.mean()) / Advantages.std()
    return V_Target_G, Advantages
    
def surrogate_loss(actor, old_policy, Advantages, scurge_states_b, observer_states_b, actions):
    policy = torch.zeros_like(old_policy)## policy를 담기위한 0 배열
    
    prob_each_actions = actor(torch.Tensor(scurge_states_b).cuda(), torch.Tensor(observer_states_b).cuda(), soft_dim=1) ##actor로 부터 softmax화 시킨 각 action의 확률 값을 받습니다.
    distribution = Categorical(prob_each_actions) ## Categorical 함수를 이용해 하나의 분포도로 만들어줍니다.
    entropies = distribution.entropy() ## 논문에 pi_seta에서 St에 따른 entropy bonus여서 action들의 분포에 대한 entropy로 구했습니다.
    
    actions = actions.unsqueeze(1).cuda()
    policy = prob_each_actions.gather(1,actions).cuda()
    old_policy = old_policy.unsqueeze(1).cuda()
    
    ratio = torch.exp(torch.log(policy) - torch.log(old_policy)) ## 원래는 policy_pi/policy_old_pi 식인데 = exp(log(policy_pi)-log(policy_old_pi)) 로 변경한것. 정확한 이유는 모르지만, 더 효율적이라 이렇게 쓴다고함.
    
    ratio_A = ratio.cuda() * Advantages.cuda()

    return ratio, ratio_A, entropies

def train(writer, n_iter, actor, critic, trajectories, actor_optimizer, critic_optimizer, T_horizon, batch_size, epoch):
    
    c_1 = 1 ## coefficient c1
    c_2 = 0.01 ## coefficient c2
    eps = 0.2 ## clipped에 사용되는 epsilon
    
    trajectories = np.array(trajectories) ##deque인 type을 np.array로 바꿔 줍니다.
    scurge_states = list(trajectories[:, 0])
    observer_states = list(trajectories[:, 1])
    actions = list(trajectories[:, 2]) ## 나머지 action, reward, mask는 1개의 상수 값이기때문에 그냥 list화 해줍니다.
    rewards = list(trajectories[:, 3])  
    masks = list(trajectories[:, 4]) 
    old_policies = list(trajectories[:, 5])

    
    V_Target_G, Advantages = GAE(critic, scurge_states, observer_states, rewards, masks) ## random mini batch 전에 미리 구해놔야함. 아무래도 t 순서대로 계산되다보니 미리 구하는 게 편함.

    r_batch = np.arange(len(scurge_states)) ## random mini batch를 위해 저장된 states 만큼 숫자순서대로 배열을 만듬. 1~
    
    for i in range(epoch): ## 전체 데이터를 몇번 반복해 학습할지. 10으로 해둠. 즉 trajactories에 2048개 데이터가 저장되어있다면, 2048*10번 함.
        np.random.shuffle(r_batch) ## 1~ 들어있는 숫자들을 무작위로 섞음.
        
        for j in range(T_horizon//batch_size): ##2048/64  0 ~ 31 
            mini_batch = r_batch[batch_size * j: batch_size * (j+1)] ## batch 크기 간격으로 앞에서부터 자름.
            
            mini_batch = torch.LongTensor(mini_batch) ## LongTensor가 찾아보니 정수형이였음.
            
            scurge_states_b = torch.Tensor(scurge_states)[mini_batch] ## 선택된 mini_batch index에 맞게 batch 크기 만큼 데이터를 선별함.
            observer_states_b = torch.Tensor(observer_states)[mini_batch]
            
            actions_b = torch.LongTensor(actions)[mini_batch]
                 
            
            V_Target_G_b = torch.Tensor(V_Target_G)[mini_batch].detach() ## Target은 변하면 안되기 때문에 detach()를 해준다.
            
            ## critic loss
            mse_loss = torch.nn.MSELoss() ## mse_loss 정의
            
            values = critic(scurge_states_b.cuda(), observer_states_b.cuda()).squeeze(1) ## V_Target_G_b = [64] 인데 values = [64,1] 이여서 dimension을 맞춰주기위해 values=[64]로 만듬.
            critic_loss = mse_loss(values.cuda(),V_Target_G_b.cuda()) ## mean square loss라서 이 값 자체가 mean임.

            ## critic loss end.
            
            ## actor loss
                        
            old_policies_b = torch.Tensor(old_policies)[mini_batch].detach() ## old_policy 값은 backpropagation에 반영되지 않도록 detach 해준다.
            
            Advantages_b = torch.Tensor(Advantages)[mini_batch]
            Advantages_b = Advantages_b.unsqueeze(1) ## 차원을 맞추기위해 1추가.
                        
            ratio, L_CPI, entropies= surrogate_loss(actor, old_policies_b, Advantages_b, scurge_states_b, observer_states_b, actions_b) ## ratio = policy_pi/old_policy_pi, L_CPI = ratio * Advantage
            
            entropies= entropies.mean() ## entropy도 평균 구함.
            
            clipped_surrogate = torch.clamp(ratio,1-eps,1+eps).cuda() * Advantages_b.cuda() ## clip 부분

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
    writer = SummaryWriter('C:/SAIDA_RL/python/saida_agent_example/avoidObserver/save_ppo/log_file')
    load = False
    episode = 0
    
    env = AvoidObservers(action_type=0, frames_per_step=6, move_angle=20, move_dist=3, verbose=0, no_gui=False)
    torch.manual_seed(500) ## seed는 안줘도 되는 데 혹시 몰라서 줬습니다.
    ## 환경을 불러온다.
    ## 환경은 기존에 저희가 파악했던대로 파라미터 값을 넣어주면 됩니다.
    
    print_interval = 10 ## 출력
    batch_size = 5 ## batch가 작을 수록 학습에 오래걸립니다.. 아무래도 tensor 연산이 한번에 적은 양만 되서 오래걸려요.
    epoch = 2 ## epoch도 늘릴 수록 학습에 오래걸립니다. 
    learning_rate=0.00003
    T_horizon = 500 ## 얼마큼의 trajectory를 수집할건지. 여러 actor를 한다면 여러 actor가 수집한 총 trajectory 갯수.
                    ## 1024
    

    actor = Actor().cuda() ## actor 생성
    critic = Critic().cuda()## critic 생성
        
    if load:
        actor.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidObserver/save_ppo/','ppo_actor_'+str(episode)+'.pkl')))
        critic.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidObserver/save_ppo/','ppo_critic_'+str(episode)+'.pkl')))
        
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate) ## actor에 대한 optimizer Adam으로 설정하기.
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate) ## critic에 대한 optimizer Adam으로 설정하기.
    
    temp_score = 0.0
    score = 0.0
    for n_iter in range(10000): ## 반복
        trajectories = deque() ## (s,a,r,done 상태) 를 저장하는 history or trajectories라고 부름. 학습을 위해 저장 되어지는 경험 메모리라고 보면됨.
                                ## PPO는 on-policy라 한번 학습하고 버림.
        step = 0
        scurge_state_frames_4 = deque(maxlen=4) ## deque 자료구조를 이용해 4채널 이상이 되면 마지막 채널 삭제 유도. 
        observer_state_frames_4 = deque(maxlen=4)
        
        while step < T_horizon: ## trajectory를 T_horizon 이상으로 수집.
            state = env.reset() ## 게임 환경 초기화 및 초기 state 받기
            
            im_state = state_to_image(state) ## image로 mapping
            empty_64 = np.zeros_like(im_state[0]) ## 맨 처음 4채널중 3개 채널은 0으로 임의로 채워줍니다.
            scurge_state_frames_4.append(empty_64) ## 스커지 관련 귀찮아서 하드코딩.. 
            scurge_state_frames_4.append(empty_64)
            scurge_state_frames_4.append(empty_64)
            scurge_state_frames_4.append(im_state[0]) ## 초기에 4채널 중 3채널은 0으로 채우고, 마지막 채널은 초기 state로 임베딩한다.
            ##print(torch.Tensor(scurge_state_frames_4).shape) ## 4*64*64 : 채널 * 행 * 열 임.

            empty_21 = np.zeros_like(im_state[1]) ## 맨 처음 4채널중 3개 채널은 0으로 임의로 채워줍니다.
            observer_state_frames_4.append(empty_21) ## 옵저버 관련 귀찮아서 하드코딩.. 
            observer_state_frames_4.append(empty_21)
            observer_state_frames_4.append(empty_21)
            observer_state_frames_4.append(im_state[1]) ## 초기에 4채널 중 3채널은 0으로 채우고, 마지막 채널은 초기 state로 임베딩한다.
            ##print(torch.Tensor(observer_state_frames_4).shape) ## 4*21*21 : 채널 * 행 * 열 임.
            ## 스커지 이미지와 옵저버 이미지를 따로 따로 convnet으로 처리해 concat 하는 방식을 사용할 예정. 2-way cnn.
            
            episode+=1
            highest = 9999 ## 변수명은 highest 지만 실제로는 가장 작은 y 값임
            while True: ## history 얼마나 모을지
                step+=1
                
                prob_each_actions = actor(torch.Tensor(scurge_state_frames_4).unsqueeze(0).cuda(), torch.Tensor(observer_state_frames_4).unsqueeze(0).cuda(), soft_dim=1) ##actor로 부터 softmax화 시킨 각 action의 확률 값을 받습니다.
                prob_each_actions = prob_each_actions.squeeze(0)
                distribution = Categorical(prob_each_actions) ## pytorch categorical 함수는 array에 담겨진 값들을 확률로 정해줍니다.
                
                action = distribution.sample().item() ## ex) prob_each_actions = [0.25, 0.35, 0.1, 0.3] 이라면 각각의 인덱스가 25%, 35%, 10%, 30%
                                            ## 0,1,2,3 값 중 하나가 위 확률에 따라 선택됨.
               
                old_policy = prob_each_actions[action] ## 실제 선택된 action의 확률 값만 저장해줌.
                
                next_state, reward, done, info = env.step([action]) ## * Saida RL library 분포에서 선택된 action이 다음 step에 들어감.
                
                reward, highest = reward_shape(highest, state, next_state, reward, done) ## reward 수정해야함.
                
                
                mask = 0 if done else 1 ## 게임이 종료됬으면, done이 1이면 mask =0 그냥 생존유무 표시용.
                
                trajectories.append((scurge_state_frames_4, observer_state_frames_4, action, reward, mask, old_policy))
                
                state = next_state ## current state를 이제 next_state로 변경
                
                next_state = state_to_image(next_state) ## 이미지로 변환
                scurge_state_frames_4.append(next_state[0]) ## 가장 최근 frame 추가
                observer_state_frames_4.append(next_state[1]) ## 가장 최근 frame 추가
                
                score += reward ## reward 갱신.
               
                temp_score += reward ## 출력용 변수
                
                if done and reward == 50: ## 게임이 끝나고 goal reward를 얻었다면, 모델 저장.
                    
                    torch.save(actor.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidObserver/save_ppo/','clear_ppo_actor_'+str(episode)+'.pkl'))
                    torch.save(critic.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/avoidObserver/save_ppo/','clear_ppo_critic_'+str(episode)+'.pkl'))
                
                if done: ## 죽었다면 게임 초기화를 위한 반복문 탈출
                    print("step: ", step, "per_episode_score: ",temp_score)
                    temp_score = 0.0
                    break
            
            if episode%print_interval==0: ## 10 episode마다 score 출력 및 score 초기화.
                print("# of episode :{}, avg score : {:.1f}".format(episode, score//print_interval))
                writer.add_scalar('log/score', float(score//print_interval), episode)
                score = 0.0
        actor.train() ## model actor를 train모드로 변경
        critic.train()
            
        train(writer, n_iter, actor, critic, trajectories, actor_optimizer, critic_optimizer, T_horizon, batch_size, epoch) ## train 함수.

        
        
    env.close()
    
if __name__ == '__main__':
    main()
