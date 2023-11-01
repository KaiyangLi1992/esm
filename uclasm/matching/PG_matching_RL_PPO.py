import os
import datetime
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import pickle
import numpy as np
import sys
import time



sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/") 


# Custom module imports
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver
from NSUBS.model.OurSGM.train import train, cross_entropy_smooth
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.src.utils import OurTimer, save_pickle
from environment import environment, update_state, calculate_cost

from torch.optim.lr_scheduler import StepLR
# Constants
dataset_file_name = './data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_RWSE.pkl'
matching_file_name = './data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_matching.pkl'
dim = 47
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = torch.device(FLAGS.device)
imitationlearning = True
checkpoint_path = '/home/kli16/ISM_custom/esm_NSUBS/esm/ckpt_imitationlearning/2023-10-26_11-48-59/checkpoint_80000.pth'
checkpoint = torch.load(checkpoint_path)

def _create_model(d_in_raw):
    model = create_dvn(d_in_raw, FLAGS.d_enc)
    saver.log_model_architecture(model, 'model')
    return model.to(FLAGS.device)

    


with open(dataset_file_name,'rb') as f:
    dataset = pickle.load(f)

with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)


def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result


def _preprocess_NSUBS(state):
    g1 = state.g1
    g2 = state.g2
    u = state.action_space[0][0]
    v_li = [action[1] for action in state.action_space]
    CS = _get_CS(state,g1,g2)
    nn_map = state.nn_mapping
    candidate_map = {u:v_li}
    return (g1,g2,u,v_li,nn_map,CS,candidate_map)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, lr, gamma, K_epochs, eps_clip):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr = lr
        self.buffer = RolloutBuffer()

        self.policy = _create_model(dim).to(device)
        if imitationlearning is True:
            self.policy.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = optim.Adam(self.policy.parameters(), self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.1)

        self.policy_old = _create_model(dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()



    def select_action(self, state,env):
            with torch.no_grad():
                update_state(state,env.threshold)
                assert ~np.any(np.all(state.candidates == False, axis=1))
                action_exp = state.get_action_heuristic()
                state.action_space = state.get_action_space(action_exp)
                pre_processed = _preprocess_NSUBS(state)
                out_policy, out_value, out_other = \
                    self.policy_old(*pre_processed,
                        True,
                        graph_filter=None, filter_key=None,
                )
                action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10
                m = Categorical(action_prob)
                action_ind = m.sample()
                action = state.action_space[action_ind]
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(m.log_prob(action_ind))
            self.buffer.state_values.append(out_value)

            return action

    def policy_evaluate(self,states,actions):
        action_logprobs_li = []
        dist_entropy_li= []
        state_values_li = []

        for state,action in zip(states,actions):
            pre_processed = _preprocess_NSUBS(state)
            # start_time = time.time()
            out_policy, out_value, out_other = \
                self.policy(*pre_processed,
                    True,
                    graph_filter=None, filter_key=None,
            )
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"Execution time: {elapsed_time:.2f} seconds")
            action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10
            ind = state.action_space.index(action)
            dist = Categorical(action_prob)
            action_logprob = dist.log_prob(torch.tensor(ind).to(device))
            # dist_entropy = dist.entropy()
            state_value = out_value

            action_logprobs_li.append(action_logprob)
            dist_entropy_li.append(dist.entropy())
            state_values_li.append(state_value)

        return torch.stack(action_logprobs_li),torch.stack(state_values_li),torch.stack(dist_entropy_li)



    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            
            logprobs, state_values, dist_entropy = self.policy_evaluate(old_states, old_actions)
     
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # loss.mean().item()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return loss.mean().item()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

################ PPO hyperparameters ################


update_timestep = 1 * 4      # update policy every n timesteps
K_epochs = 10               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor
max_training_episodes = int(1e5) 
lr = 1e-4
random_seed = 0         # set random seed if required (0 = no random seed)



###################### logging ######################
#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir = log_dir + '/' + timestamp + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)


#### create new log file for each run 
log_f_name = log_dir + '/PPO_' + timestamp + "_log.txt"

print("current logging run number for " + timestamp)
print("logging at : " + log_f_name)

############# print all hyperparameters #############
log_f = open(log_f_name,"w+")
log_f.write("--------------------------------------------------------------------------------------------\n")


log_f.write("PPO update frequency : " + str(update_timestep) + " timesteps\n") 
log_f.write(f"PPO K epochs : {K_epochs}\n")
log_f.write(f"PPO epsilon clip : {eps_clip}\n")
log_f.write(f"discount factor (gamma) : {gamma}\n", )
log_f.write(f"learning rate : {lr}\n")
log_f.write("--------------------------------------------------------------------------------------------\n")
log_f.write(f"d_enc : {FLAGS.d_enc}\n")
log_f.write(f"device : {FLAGS.device}\n")
if imitationlearning is True:
    log_f.write(f"im_ckpt_path : {checkpoint_path}\n")

log_f.write("--------------------------------------------------------------------------------------------\n")




def main():
    # model = _create_model(dim).to(device)
    
    # model.load_state_dict(checkpoint['model_state_dict'])
    writer = SummaryWriter(f'plt_RL/{timestamp}')
    env  = environment(dataset)
    ppo_agent = PPO(lr, gamma, K_epochs, eps_clip)
   

    for episode in range(max_training_episodes):
        state_init = env.reset()
        update_state(state_init,env.threshold)
        stack = [state_init]       
        while stack:
            state = stack.pop()
            action = ppo_agent.select_action(state,env)
            newstate, state,reward, done = env.step(state,action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            stack.append(newstate)
            if done:
                cost = calculate_cost(newstate.g1,newstate.g2,newstate.nn_mapping)
                break

        if episode % update_timestep == 0:
            loss = ppo_agent.update()
        ppo_agent.scheduler.step()
         


        
        if episode % 10 == 0:
            writer.add_scalar('Cost', cost, episode)
            writer.add_scalar('Loss', loss, episode)
            print(f"Cost:{cost}")
            print(f"Lost:{loss}")


        if episode % 100 == 0:

        # 创建一个检查点每隔几个时期
            checkpoint = {
                'epoch': episode,
                'model_state_dict': ppo_agent.policy_old.state_dict(),
                'optimizer_state_dict': ppo_agent.policy_old.state_dict(),
                # ... (其他你想保存的元数据)
            }
            directory_name = f"ckpt_RL/{timestamp}/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            torch.save(checkpoint, f'ckpt_RL/{timestamp}/checkpoint_{episode}.pth')
    

if __name__ == '__main__':
    main()
    log_f.close()



