## gym 환경 import avoidreavers
from saida_gym.starcraft.avoidReavers import AvoidReavers
import numpy as np
#import for ppo
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#Hyperparameters
learning_rate = 0.0003
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 2
T_horizon     = 100

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(32,256)
        self.fc_pi = nn.Linear(256,25)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    
    def put_data(self, transition):
        self.data.append(transition)
        
        
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
    
    
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            
def reward_reshape(reward):
    ''' Reshape the reward
        Starcraft Env returns the reward according to following conditions.
        1. Invalid action : -0.1
        2. get hit : -1
        3. goal : 1
        4. others : 0'''

    if (reward + 0.1) < 0.01:
        reward = -1
    elif reward == -1:
        reward = -10
    elif reward == 1:
        reward = 10
    elif reward == 0:
        reward = -0.1

    return reward


def process_observation(observation):
        if len(observation.my_unit) > 0:
            s=np.zeros(32)

            me = observation.my_unit[0]
            # Observation for Dropship
            s[0] = (me.pos_x)  # X of coordinates
            s[1] = (me.pos_y)  # Y of coordinates
            s[2] = (me.pos_x - 320)  # relative X of coordinates from goal
            s[3] = (me.pos_y - 320)  # relative Y of coordinates from goal
            s[4] = (me.velocity_x)  # X of velocity
            s[5] = (me.velocity_y)  # y of coordinates
            s[6] = (me.angle)  # Angle of head of dropship
            s[7] = 1 if me.accelerating else 0  # True if Dropship is accelerating

            # Observation for Reavers
            for ind, ob in enumerate(observation.en_unit):
                s[ind * 8 + 8] = (ob.pos_x - me.pos_x)  # X of relative coordinates
                s[ind * 8 + 9] = (ob.pos_y - me.pos_y)  # Y of relative coordinates
                s[ind * 8 + 10] = (ob.pos_x - 320)  # X of relative coordinates
                s[ind * 8 + 11] = (ob.pos_y - 320)  # Y of relative coordinates
                s[ind * 8 + 12] = (ob.velocity_x)  # X of velocity
                s[ind * 8 + 13] = (ob.velocity_y)  # Y of velocity
                s[ind * 8 + 14] = (ob.angle)  # Angle of head of Reavers
                s[ind * 8 + 15] = 1 if ob.accelerating else 0  # True if Reaver is accelerating

        return s

    
def main():
    env = AvoidReavers(frames_per_step=4, action_type=0, move_angle=15, move_dist=1, verbose=0, \
                       bot_runner=r"SAIDA_RL\cpp\Release\SAIDA\SAIDA.exe", no_gui=False)
    model = PPO()
    score = 0.0
    print_interval = 1

    for n_epi in range(10000):
        s = env.reset()
        s = process_observation(s)
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step([a])
                s_prime = process_observation(s_prime)
                r = reward_reshape(r)
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()
        if (n_epi % print_interval == 0) and (n_epi != 0):
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()
    
    
if __name__ == '__main__':
    main()




