import torch
import gym
from agent import Agent
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('..')
from transformer import Transformer


env = gym.make("HalfCheetah-v2")
traj_target_length = 200
device = 'cuda:3'


# set seed
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)
env.seed(3407)



agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=512,
    action_dim=env.action_space.shape[0],
    action_highs=env.action_space.high,
    action_lows=env.action_space.low,
    device=device,
)
# agent.load()

rm_trans = Transformer(n_layer=3, 
                        in_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                        out_dim=1, 
                        n_head=8,
                        dim=256, 
                        dropout=0.01, 
                        max_seqlen=traj_target_length+1).to(device)

weights_path = '../checkpoints/reward_model_epoch_5.pt'

rm_trans.load_state_dict(torch.load(weights_path))

episode_return_rm_list = []
episode_return_rr_list = []

reward_step_rm = []
reward_step_rr = []
for episode_i in range(1000):
    state = env.reset()

    episode_return_rm = 0
    episode_return_rr = 0

    done = False
    
    state_action_last = None
    while not done:
        action = agent.take_action(state)
        next_state, reward_rule, terminated, info = env.step(action)
        
        # get reward
        state_r = torch.tensor(state, dtype=torch.float, device=device)
        action_r = torch.tensor(action, dtype=torch.float, device=device)
        state_r = state_r.unsqueeze(0)
        action_r = action_r.unsqueeze(0)
        state_action = torch.cat([state_r, action_r], dim=-1).unsqueeze(0)
        if state_action_last is not None:
            state_action = torch.cat([state_action_last, state_action], dim=1)
        
        if state_action.shape[1] > traj_target_length:
            state_action = state_action[:, -traj_target_length:, :]
            
        state_action_last = state_action
        reward = rm_trans(state_action).squeeze(0)[-1].item()
        
        
        if terminated:
            done = True
            
        agent.buffer.states.append(state)
        agent.buffer.actions.append(action)
        agent.buffer.rewards.append(reward)
        agent.buffer.next_states.append(next_state)
        agent.buffer.dones.append(done)
        
        state = next_state   
        
        episode_return_rm += reward
        episode_return_rr += reward_rule
        
        reward_step_rm.append(reward)
        reward_step_rr.append(reward_rule)
        

    agent.update()
    if episode_i % 10 == 0:
        agent.save()
    
    episode_return_rm_list.append(episode_return_rm)
    episode_return_rr_list.append(episode_return_rr)
    
    print(f'{episode_i=} {episode_return_rr=}')


agent.save()

plt.figure(figsize=(18, 8))

plt.subplot(2, 2, 1)
plt.plot(reward_step_rm)
plt.title('reward_step_rm')

plt.subplot(2, 2, 2)
plt.plot(reward_step_rr)
plt.title('reward_step_rr')

plt.subplot(2, 2, 3)
plt.plot(episode_return_rm_list)
plt.title('episode_return_rm_list')

plt.subplot(2, 2, 4)
plt.plot(episode_return_rr_list)
plt.title('episode_return_rr_list')


plt.savefig('train_online.png')


env.close()
    
    