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
        self.fc1 = nn.Linear(state_size,64) ## input state
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,action_size) ## output each action

    def forward(self, x, soft_dim):
        x = torch.tanh(self.fc1(x))        
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        prob_each_actions = F.softmax(self.fc4(x),dim=soft_dim) ## NN에서 각 action에 대한 확률을 추정한다.

        return prob_each_actions

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size,64) ## input state
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,1)## output value

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.fc4(x)

        return value


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
    DAMAGED_REWARD = -6
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
        
        if my_pre_hp - my_cur_hp > 0: ## 벌쳐가 맞았을 때
            reward += DAMAGED_REWARD
        if en_pre_hp - en_cur_hp > 0: ## 질럿을 때렸을 때
            reward += HIT_REWARD
        
        ## 벌쳐가 맞고, 질럿도 때리는 2가지 동시 case가 있을 거 같아. reward를 +=을 했고 각각 if문으로 처리했습니다.
    
    return reward
    
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
        
    Advantages = (Advantages - Advantages.mean()) / Advantages.std() ## 논문에는 없지만, 논문 저자의 openAI baseline code를 보면 Advantage를 정규화 시킵니다.
    return V_Target_G, Advantages
    
def surrogate_loss(actor, old_policy, Advantages, states, actions):
    policy = torch.zeros_like(old_policy) ## policy를 담기위한 0 배열
    
    prob_each_actions = actor(torch.Tensor(states), soft_dim=1) ##actor로 부터 softmax화 시킨 각 action의 확률 값을 받습니다.
    distribution = Categorical(prob_each_actions) ## Categorical 함수를 이용해 하나의 분포도로 만들어줍니다.
    entropies = distribution.entropy() ## 논문에 pi_seta에서 St에 따른 entropy bonus여서 action들의 분포에 대한 entropy로 구했습니다.
    
    actions = actions.unsqueeze(1) ## [batch_size, 1] shape 맞추기
    policy = prob_each_actions.gather(1,actions) ## column 기준으로 [batch_size, 1] 실제로 선택되었던 actions index의 policy들만 모아줍니다.
    old_policy = old_policy.unsqueeze(1) ## [batch_size, 1] shape 맞추기

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
            actions_b = torch.LongTensor(actions)[mini_batch] ## action은 정수라 LongTensor로 함.
                 
            
            V_Target_G_b = torch.Tensor(V_Target_G)[mini_batch].detach() ## Target은 변하면 안되기 때문에 detach()를 해준다.
            
            ## critic loss
            mse_loss = torch.nn.MSELoss() ## mse_loss 정의
            
            values = critic(states_b).squeeze(1) ## V_Target_G_b = [64] 인데 values = [64,1] 이여서 dimension을 맞춰주기위해 values=[64]로 만듬.
            critic_loss = mse_loss(values,V_Target_G_b) ## mean square loss라서 이 값 자체가 mean임.

            ## critic loss end.
            
            ## actor loss
                        
            old_policies_b = torch.Tensor(old_policies)[mini_batch].detach() ## old_policy 값은 backpropagation에 반영되지 않도록 detach 해준다.
            
            Advantages_b = torch.Tensor(Advantages)[mini_batch]
            Advantages_b = Advantages_b.unsqueeze(1) ## 차원을 맞추기위해 1추가. [batch_size, 1] 임.
                        
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
    writer = SummaryWriter('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_ppo3_clear/log_file')
    load = False
    episode = 0
    
    env = VultureVsZealot(version=0, frames_per_step=12, action_type=0, move_angle=20, move_dist=3, verbose=0, no_gui=True
                          ,auto_kill=False) ## clear frame = 12 move = 45 move_dist = 6
                                            ## clear frame = 12 move = 30 move_dist = 3 .. best
    print_interval = 10 ## 출력
    batch_size = 5 ## 8
    epoch = 10
    learning_rate=0.00003
    T_horizon = 3000 ## 얼마큼의 trajectory를 수집할건지. 여러 actor를 한다면 여러 actor가 수집한 총 trajectory 갯수.
                    ## 1024
    torch.manual_seed(500)
    ## 환경을 불러온다.
    ## 환경은 기존에 저희가 파악했던대로 파라미터 값을 넣어주면 됩니다.

    state_size=env.observation_space
    print(state_size)
    state_size = 38
    ## state 갯수가 반환된다. 주의하셔야할게 state_size는 고정 31인데, move_angle에 따라 state_size가 변합니다. 이걸 주의하셔서 NN input 사이즈랑 잘 고려해주세요.
    action_size = env.action_space
    print(action_size)
    action_size= 19
    ## action 갯수가 반환된다.

    actor = Actor(state_size, action_size) ## actor 생성
    critic = Critic(state_size)
    
    
    if load:
        actor.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_ppo3_clear/','ppo_actor_'+str(episode)+'.pkl')))
        critic.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_ppo3_clear/','ppo_critic_'+str(episode)+'.pkl')))
        
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate) ## actor에 대한 optimizer Adam으로 설정하기.
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate) ## critic에 대한 optimizer Adam으로 설정하기.
    
    temp_score = 0.0 ## 출력용 변수
    score = 0.0
    for n_iter in range(10000): ## 반복
        trajectories = deque() ## (s,a,r,done 상태) 를 저장하는 history or trajectories라고 부름. 학습을 위해 저장 되어지는 경험 메모리라고 보면됨.
                                ## PPO는 on-policy라 한번 학습하고 버림.
        step = 0
        
        while step < T_horizon: ## trajectory를 T_horizon 이상으로 수집.
            state = env.reset() ## 게임 환경 초기화 및 초기 state 받기    
            
            state = rearrange_State(state, state_size, env)
            episode+=1
            
            while True: ## history 얼마나 모을지
                step+=1
                
                prob_each_actions = actor(torch.Tensor(state), soft_dim=0) ##actor로 부터 softmax화 시킨 각 action의 확률 값을 받습니다.
   
                distribution = Categorical(prob_each_actions) ## pytorch categorical 함수는 array에 담겨진 값들을 확률로 정해줍니다.
                
                action = distribution.sample().item() ## ex) prob_each_actions = [0.25, 0.35, 0.1, 0.3] 이라면 각각의 인덱스가 25%, 35%, 10%, 30%
                                            ## 0,1,2,3 값 중 하나가 위 확률에 따라 선택됨. 
                old_policy = prob_each_actions[action] ## 실제 선택된 action의 확률 값을 old_policy로 저장함.
                
                next_state, reward, done, info = env.step([action]) ## * Saida RL library 분포에서 선택된 action이 다음 step에 들어감.
                next_state = rearrange_State(next_state, state_size, env)
                
                reward = reward_reshape(state, next_state, reward, done) ## damage_count = 0이면 한대도 안맞았다는 뜻.
                
                mask = 0 if done else 1 ## 게임이 종료됬으면, done이 1이면 mask =0 그냥 생존유무 표시용.
                
                trajectories.append((state, action, reward, mask, old_policy))
                
                state = next_state ## current state를 이제 next_state로 변경

                score += reward ## reward 갱신.
               
                temp_score += reward ## 출력용 변수
                
                if next_state[3] == 1.0 and next_state[-6] == 0:
                    print("clear: ",next_state[3],next_state[-6],"clear_score: ",temp_score)
                    torch.save(actor.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_ppo3_clear/','clear_ppo_actor_'+str(episode)+'.pkl'))
                    torch.save(critic.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_ppo3_clear/','clear_ppo_critic_'+str(episode)+'.pkl'))
                if done: ## 죽었다면 게임 초기화를 위한 반복문 탈출
                    print("step: ", step, "per_episode_score: ",temp_score)
                    temp_score = 0.0
                    break
            
            if episode%print_interval==0: ## 10 episode마다 score 출력 및 score 초기화.
                print("# of episode :{}, avg score : {:.1f}".format(episode, score//print_interval))
                writer.add_scalar('log/score', float(score//print_interval), episode)
                score = 0.0
        actor.train() ## model actor를 train모드로 변경, 몇몇 코드를 보니 굳이 안하고 바로 밑에 train함수로 끝내기도 함.
        critic.train()
            
        train(writer, n_iter, actor, critic, trajectories, actor_optimizer, critic_optimizer, T_horizon, batch_size, epoch) ## train 함수.

        
        
    env.close()
    
if __name__ == '__main__':
    main()


