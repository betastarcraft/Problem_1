from saida_gym.starcraft.vultureVsZealot import VultureVsZealot
## gym 환경 import VultureVsZealot

from collections import deque
import numpy as np
import random
import os
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


class DQN_agent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN_agent, self).__init__()
        self.fc1 = nn.Linear(state_size,64) ## input state
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,action_size) ## output each action

    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        Q_value = self.fc4(x)

        return Q_value ## Q_value로 표현하는데, 어떤 state에 action들이 가지는 미래추정치 정도로 이해하시면 더 수월하실거에요.

def rearrange_State(state, state_size):
    state_arr = deque(maxlen=state_size) ## 구현이 그렇게 매끄럽진 않아서 원하는 state size 만큼 deque를 만들어줍니다.
    my_pos_x=0
    my_pos_y=0

    
    if state.my_unit: ## 벌쳐가 죽지 않았다면, 실제로 환경에서 벌쳐나 질럿이 죽게되면 아예 state에서 정보가 제거 됩니다.
        for my in state.my_unit: ## 이게 실제로는 반복이 1번 도는 데, dictionary type 처리가 이게 편해요.
            my_pos_x = my.pos_x
            my_pos_y = my.pos_y
            
            state_arr.append(my.hp/80)
            state_arr.append(my.pos_x)
            state_arr.append(my.pos_y)
            state_arr.append((my.angle-math.pi)/math.pi)
            for dir_my_unit in my.pos_info:
                state_arr.append(dir_my_unit.nearest_obstacle_dist/320)
    else: ## 벌쳐가 죽었다면, 벌쳐관련 state들을 임의로 0값으로 대신 채워 줍니다.
        for _ in range(state_size - 4):
            state_arr.append(0)

    if state.en_unit: ## 질럿이 죽지 않았다면,
        for en in state.en_unit:
            state_arr.append(abs(my_pos_x - en.pos_x)+abs(my_pos_y - en.pos_y))
            state_arr.append((en.hp+en.shield)/160)
            state_arr.append(en.pos_x)
            state_arr.append(en.pos_y)
            state_arr.append((en.angle-math.pi)/math.pi)
    else: ## 질럿이 죽었다면, 질럿과련 state들을 임의로 0값으로 대신 채워 줍니다.
        for _ in range(5):
            state_arr.append(0)
    ## 저는 최대한 간단하게 기존 환경이 주는 state와 함께, hp부분만 shield랑 합쳤습니다.
    
    
    return state_arr



def reward_reshape(state, next_state, reward, done):

    KILL_REWARD = 10
    DEAD_REWARD = -10
    DAMAGED_REWARD = -5
    HIT_REWARD = 4
    BASE_PAN = -0.02
    
    if done:
        if reward > 0: ## env에서 반환된 reward가 1 이면, 질럿을 잡음.
            reward = KILL_REWARD
            # 잡은  경우
        else: ## 게임이 종료되고 -1 값을 받게 된다면, 
            reward = DEAD_REWARD
            
    else: ## 게임이 종료되지 않았다면,
        my_pre_hp = state[0]
        my_cur_hp = next_state[0]
        
        en_pre_hp = state[-4]
        en_cur_hp = next_state[-4]
        
        my_r=(1-my_cur_hp)*DAMAGED_REWARD
        en_r=(1-en_cur_hp)*HIT_REWARD

        reward = my_r+en_r

        if en_pre_hp - en_cur_hp == 0:
            reward += BASE_PAN
        """
        if my_pre_hp - my_cur_hp > 0: ## 벌쳐가 맞아 버렸네 ㅠㅠ
            reward += (1-my_cur_hp)*DAMAGED_REWARD
        if en_pre_hp - en_cur_hp > 0: ## 질럿을 때려 버렸네 ㅠㅠ
            reward += (1-en_cur_hp)*HIT_REWARD
        """
        ## 벌쳐가 맞고, 질럿도 때리는 2가지 동시 case가 있을 거 같아. reward를 +=을 했고 각각 if문으로 처리했습니다.
    
    return reward

def epsilon_greedy(q_values, action_size, epsilon):
    if np.random.rand() <= epsilon: ## 정해준 epsilon 값보다 작은 random 값이 나오면 
        action = random.randrange(action_size) ## action을 random하게 선택합니다.
        return action
    
    else: ## epsilon 값보다 크면, 학습된 Q_player NN 에서 얻어진 q_value 값중 가장 큰 action을 선택합니다.
        return q_values.argmax().item()

    
def train_model(writer,step,Q_player, Q_target, optimizer,random_mini_batch):
    ## state에 대한 q_value는 Q_player NN에서 얻고
    ## next_state에 대한 q_value는 Q_target NN에서 얻는다.
    gamma = 0.99
    
    mini_batch = np.array(random_mini_batch) ## deque type인 mini batch를 array로 변경해준다.
    states = np.vstack(mini_batch[:, 0]) ## state는 1차원이 아니므로 vstack을 이용해 쌓아준다.
    actions = list(mini_batch[:, 1]) ##  actions을 list화
    rewards = list(mini_batch[:, 2]) ## rewards를 list화
    next_states = np.vstack(mini_batch[:, 3]) ## next_state도 vstack.
    masks = list(mini_batch[:, 4]) ## list화

    actions = torch.LongTensor(actions) ## 확률이여서 long으로.
    rewards = torch.Tensor(rewards) ## loss 수식에 들어가는 값들은 torch.tensor로 만들어주자.
    masks = torch.Tensor(masks) ## loss 수식에 들어가는 값들은 torch.tensor로 만들어주자.
    
    MSE = torch.nn.MSELoss() ## mean squear error 사용.

    # get Q-value
    Q_player_q_values = Q_player(torch.Tensor(states)) ## 계속 학습중인 Q_player NN에서 예상되는 action의 q_value를 얻어온다.
    q_value = Q_player_q_values.gather(1, actions.unsqueeze(1)).view(-1) ## 각 state별로 가장 높은 q_value 값만 불러온다.

    # get target
    Q_target_q_values = Q_target(torch.Tensor(next_states))## 실제 발생된 next_state를 넣어, target에서 예상되어지는 가치 q_value를 구한다.
    target = rewards + masks * gamma * Q_target_q_values.max(1)[0] ## 죽었다면, next가 없으므로 얻어진 reward만 추린다.
    
    
    loss = MSE(q_value, target.detach()) ## target은 단순히 주기적으로 업데이트해 네트워크를 유지시키므로, parameter가 미분되선 안된다. 그래서 detach() 해줌.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss_NN',loss.item(),step)
    
def main():
    writer = SummaryWriter('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_dqn_2/log_file')
    load = True
    st_num = 56700
    
    initial_exploration = 1000000 ## 1000000개의 memory가 쌓이고 나서 학습 시작!
    learning_rate = 0.000005
    epsilon = 0.4888
    epsilon_decay = 0.0000001
    replay_buffer = deque(maxlen=1200000) ## history or trajectory를 저장할 replay 파이썬은 list가 deque에 비해 많이 느려요.
    batch_size = 200
    print_interval = 10 ## 몇 episode 마다 출력할건지.
    update_target = 5000
    
    
    env = VultureVsZealot(version=0, frames_per_step=3, action_type=0, move_angle=30, move_dist=9, verbose=0, no_gui=False
                          ,auto_kill=False)
    ## 환경을 불러온다.
    ## 환경은 기존에 저희가 파악했던대로 파라미터 값을 넣어주면 됩니다.

    state_size=env.observation_space
    print(state_size)
    state_size = 21
    ## state 갯수가 반환된다. 주의하셔야할게 state_size는 고정 31인데, move_angle에 따라 state_size가 변합니다. 이걸 주의하셔서 NN input 사이즈랑 잘 고려해주세요.
    action_size = env.action_space
    print(action_size)
    action_size=13
    ## action 갯수가 반환된다.

    Q_player = DQN_agent(state_size, action_size) ## 게임중 계속 학습되는 네트워크
    Q_target = DQN_agent(state_size, action_size) ## 중간 중간 player 네트워크를 복사해 유지하는 target 네트워크
    
    
    if load:
        Q_player.load_state_dict(torch.load(os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_dqn_2/','dqn_2_'+str(st_num)+'.pkl')))
        with open('replay_2_'+str(st_num)+'.pkl','rb') as f:
            replay_buffer=pickle.load(f)
        
    optimizer = optim.Adam(Q_player.parameters(), lr=learning_rate) ## 그러므로 직접 뛰는 Q_player의 parameter만 변경.
    Q_target.load_state_dict(Q_player.state_dict()) ## player의 뉴럴넷 파라미터를 target에 복사!
    
    step = 6241230
    score = 0
    
    for episode in range(st_num,9000000): ## episode
        
        state = env.reset() ## episode마다 초기화.
        state = rearrange_State(state, state_size) ## dictionary 형태를 array형태로 변경하기 위한 함수.
        
        
        
        while True: ## game이 종료되기 전까지 진행된다.
            
            q_values = Q_player(torch.Tensor(state)) ## state를 통해 적절한 Q_values를 얻는다.
            
            action = epsilon_greedy(q_values, action_size, epsilon) ## Q_values와 epsilon greedy 정책을 통해 action을 선택
                
            next_state, reward, done, info = env.step([action]) ## action에 따른 next_state, reward, done을 반환해줍니다.
            next_state = rearrange_State(next_state, state_size) ## dictionary type으로 되어있는 state 값을 list로 변환해줍니다.
            
            mask = 0 if done else 1 ## 굳이 mask 값을 표현 안하고 싶으면 done으로 반대로 표현하면 됩니다.


            
            reward = reward_reshape(state, next_state, reward, done) ##reward 재처리
        
            replay_buffer.append((state,action,reward,next_state, mask)) ## state, action, reward, next_state, mask 저장.
        
            state = next_state
            
            score += reward
            step += 1
            if next_state[0] == 1 and next_state[-4] == 0: ## 최종 목표, 벌쳐 hp 80, 질럿 hp 0
                torch.save(Q_target.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_dqn_2/','dqn_2_'+str(episode)+'_clear'+'.pkl'))
                with open('replay_2_'+str(episode)+'_clear'+'.pkl','wb') as f:
                    pickle.dump(replay_buffer, f)
                    
            if step > initial_exploration: ## 일정 step 이후로 학습과 업데이트 진행, 이전에는 계속 replay_buffer에 채움.
                epsilon -= epsilon_decay ## 학습해 감에 따라 epilon에 의존하기보단 학습된 정책에 의존되게.
                epsilon = max(epsilon, 0.1) ## 그래도 가끔씩 새로운 exploration을 위해 최소 0.1은 주기.

                random_mini_batch = random.sample(replay_buffer, batch_size) ## 쌓여진 replay_buffer에서 정해진 batch_size개만큼 random으로 선택. 
            
                Q_player.train(), Q_target.train() ## 둘다 train 모드로
                train_model(writer, step, Q_player, Q_target, optimizer, random_mini_batch) ## train 함수.
            
                if step % update_target == 0: ## 일정 step마다 target network업데이트
                    Q_target.load_state_dict(Q_player.state_dict()) ## Q_player NN에 학습된 weight를 그대로 Q_target에 복사함. tensorflow는 다른 함수가 있는걸로 암.

                       
            if done:
                break
            
        if episode%print_interval==0 and episode != st_num: ## 10 episode마다 score 출력 및 score 초기화.
            print("# of episode :{}, avg score : {:.1f}".format(episode, score//print_interval))
            print("# of step: ",step," # of epsilon: ", epsilon)
            writer.add_scalar('log/score', float(score//print_interval), episode)
            score = 0.0
        if episode%150==0 and episode !=0: ## 150 episode마다 저장.
            torch.save(Q_target.state_dict(), os.path.join('C:/SAIDA_RL/python/saida_agent_example/vultureZealot/save_dqn_2/','dqn_2_'+str(episode)+'.pkl'))

            with open('replay_2_'+str(episode)+'.pkl','wb') as f:
                pickle.dump(replay_buffer, f)
        
if __name__ == '__main__':
    main()
