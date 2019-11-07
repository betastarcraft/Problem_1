from core.common.processor import Processor
from saida_gym.starcraft.avoidObservers import AvoidObservers

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



learning_rate = 0.0003
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 2
T_horizen     = 100
MOVE_ANGLE    = 15


class Qnet(nn.Module):
    def __init__(self, output_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.last_action = None
        self.memory = []
        
    def put_data(self, data):
        self.memory.append(data)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon: # exploration
            return random.randint(0, 1)
        else : # exploitation
            return out.argmax().item()
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a


def process_step(observation, reward, done, info):
    state = self.process_observation(observation)
    reward = self.reward_shape(observation, done)
    return state, reward, done, info

def reward_shape(observation, done):
    """
    reward range
    :param observation:
    :param done:
    :return:
    """
    # Goal 에 도달하거나 죽으면
    if done:
        self.highest_height = 1900
        if 0 < observation.my_unit[0].pos_y and observation.my_unit[0].pos_y < 65 + MARGINAL_SPACE:
            return 10 * REWARD_SCALE
        # Safe zone : left-top (896, 1888) right-bottom (1056, 2048) with additional (marginal) space -> more penalty
        elif 896 - 32*MARGINAL_SPACE >= observation.my_unit[0].pos_x and observation.my_unit[0].pos_x <= 1056 + 32*MARGINAL_SPACE and observation.my_unit[0].pos_y >= 1888 - 32*MARGINAL_SPACE:
            return -10 * REWARD_SCALE
        return -5 * REWARD_SCALE

    # give important weight per height rank
    # 0 ~ 1888(59 tiles) / 32 : ratio
    if observation.my_unit[0].pos_y < self.highest_height:
        rank = int(observation.my_unit[0].pos_y / 32)  # 2 ~ 59
        weight = (59 / (rank + sys.float_info.epsilon)) / 59
        self.highest_height = observation.my_unit[0].pos_y
        return weight * 3 * REWARD_SCALE

    # 시간이 지나면
    return -0.02 * REWARD_SCALE


def process_observation(observation, last_action=None):
    LOCAL_OBSERVABLE_TILE_SIZE = 10

    # scurge's map
    map_of_scurge = np.zeros(shape=(64, 64))

    me_x = observation.my_unit[0].pos_x
    me_y = observation.my_unit[0].pos_y

    me_x_t = np.clip(int(me_x/32), 0, 64)
    me_y_t = np.clip(int(me_y/32), 0, 64)
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
    map_of_scurge = np.expand_dims(map_of_scurge, -1)

    # observer map
    map_of_observer = np.zeros(shape=(LOCAL_OBSERVABLE_TILE_SIZE*2, LOCAL_OBSERVABLE_TILE_SIZE*2))

    for ob in observation.en_unit:
        en_x_t = ob.pos_x / 32
        en_y_t = ob.pos_y / 32

        # scurge를 중앙에 두기 위해
        rel_x = int(en_x_t - me_x_t) + LOCAL_OBSERVABLE_TILE_SIZE
        rel_y = int(en_y_t - me_y_t) + LOCAL_OBSERVABLE_TILE_SIZE

        rel_x = np.clip(rel_x, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
        rel_y = np.clip(rel_y, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
        print('enemy location:', [en_x_t, en_y_t], '(relevant:', [rel_x, rel_y], ')')

        map_of_observer[rel_y][rel_x] = map_of_observer[rel_y][rel_x] + 1  # if two or more observers are duplicated, we use sum

    # display out of map where scurge can't go based on current location of scurge
    scurge_out_of_map_left = me_x_t - LOCAL_OBSERVABLE_TILE_SIZE
    scurge_out_of_map_right = me_x_t + LOCAL_OBSERVABLE_TILE_SIZE
    scurge_out_of_map_up = me_y_t - LOCAL_OBSERVABLE_TILE_SIZE
    scurge_out_of_map_down = me_y_t + LOCAL_OBSERVABLE_TILE_SIZE

    if scurge_out_of_map_left < 0:
        map_of_observer[:, 0:-scurge_out_of_map_left] = -1
    if scurge_out_of_map_right > 64:
        map_of_observer[:, -(scurge_out_of_map_right-64):] = -1
    if scurge_out_of_map_up < 0:
        map_of_observer[0:-scurge_out_of_map_up,:] = -1
    if scurge_out_of_map_down > 64:
        map_of_observer[-(scurge_out_of_map_down-64):,:] = -1

    map_of_observer = np.expand_dims(map_of_observer, -1)

    if not last_action:
        last_action = np.full((64, 64), -1)
    else:
        last_action = np.full((64, 64), last_action)
        
    print(map_of_scurge.shape)
    print(map_of_observer.shape)
    print(last_action.shape)
    
    return np.array([map_of_scurge, map_of_observer, last_action])


def process_action(action):
    act = []
    actions = []
    act.append(4)  # radiuqs tile position
    act.append(action)  # angle between 0 and 1
    act.append(0)   # move(0) attack(1)
    act[1] = np.clip(act[1], 0, 1)
    actions.append(act)
    
    last_action = act[1]

    return [actions, last_action]
        
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def main():
    env = AvoidObservers(action_type=0, verbose=0, frames_per_step=4, move_angle=MOVE_ANGLE, \
                         bot_runner=r"SAIDA_RL\cpp\Release\SAIDA\SAIDA.exe", no_gui=False)
    output_size = int(360 / MOVE_ANGLE + 1)
    q = Qnet(output_size)
    q_target = Qnet(output_size)

    print_interval = 1
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    for n_epi in range(10000):
        epsilon = max(0.01, 0.5 - (n_epi/20000)) # Linear annealing from 50% to 1%
        s = env.reset()
        s = process_observation(s)
        done = False
        
        while not done:
            for t in range(T_horizen):
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
                s_prime, r, done, info = env.step(a)
                s_prime = process_observation(s_prime, a)
                done_mask = 0.0 if done else 1.0
                q.memory.put_data((s, a, r/100.0, s_prime, done_mask))
                s = s_prime

                score += r
                if done:
                    break

            if memory.size() > 2000:
                train(q, q_target, memory, optimizer)

            if n_epi % print_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())
                print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(n_epi, \
                                                                                                         score/print_interval, \
                                                                                                         memory.size(), \
                                                                                                         epsilon*100))
                score = 0.0
    env.close()
    
    
if __name__ == '__main__':
    main()




