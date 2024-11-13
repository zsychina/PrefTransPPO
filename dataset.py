import d4rl
import gym
import numpy as np
import random


env_name = "halfcheetah"
num_pairs = 10000
traj_target_length = 200
easy_example_ratio = 0.7


env_random = gym.make(f"{env_name}-random-v2")
env_medium = gym.make(f"{env_name}-medium-v2")
env_expert = gym.make(f"{env_name}-expert-v2")


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        qposes: np.ndarray,
        qvels: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.qposes = qposes
        self.qvels = qvels
        self.size = size


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = env.get_dataset()
        
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i]) > 1e-5
                or dataset["terminals"][i] == 1.0 or dataset["timeouts"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            qposes=dataset["infos/qpos"].astype(np.float32),
            qvels=dataset["infos/qvel"].astype(np.float32),
            size=len(dataset["observations"]),
        )


# create preference dataset
# query['observations_1'], query['actions_1'], query['observations_2'], query['actions_2'], query['preference']
class PreferenceDataset:
    def __init__(self, pairs, labels):
        self.preference_dataset = []
        
        for (traj1, traj2), label in zip(pairs, labels):
            # Extract observations and actions from each trajectory
            observations_1 = traj1["observations"]
            actions_1 = traj1["actions"]
            observations_2 = traj2["observations"]
            actions_2 = traj2["actions"]
            
            # Create query dictionary with required fields
            query = {
                'observations_1': observations_1,
                'actions_1': actions_1, 
                'observations_2': observations_2,
                'actions_2': actions_2,
                'preference': label
            }
            
            self.preference_dataset.append(query)
            
        print(f"Created preference dataset with {len(self.preference_dataset)} queries")
    
    def __len__(self):
        return len(self.preference_dataset)
        
    def __getitem__(self, idx):
        return self.preference_dataset[idx]


def seperate_trajectories(dataset: D4RLDataset):
    trajs = []
    last_idx = 0
    
    for i, done in enumerate(dataset.dones_float):
        if done:
            traj = {
                'observations': dataset.observations[last_idx:i+1],
                'actions': dataset.actions[last_idx:i+1],
                'rewards': dataset.rewards[last_idx:i+1],
                'next_observations': dataset.next_observations[last_idx:i+1],
                'qposes': dataset.qposes[last_idx:i+1],
                'qvels': dataset.qvels[last_idx:i+1]
            }
            trajs.append(traj)
            last_idx = i + 1
            
    return trajs


# clip trajectories (now >= 1000) into smaller length (about 100)for model to train
def clip_trajectories(trajectories, target_length=64):
    clipped_trajs = []
    
    for traj in trajectories:
        traj_length = len(traj['observations'])
        
        # Skip trajectories shorter than target length
        if traj_length < target_length:
            continue
            
        # Split longer trajectories into chunks
        for start_idx in range(0, traj_length - target_length + 1, target_length):
            end_idx = start_idx + target_length
            clipped_traj = {
                'observations': traj['observations'][start_idx:end_idx],
                'actions': traj['actions'][start_idx:end_idx], 
                'rewards': traj['rewards'][start_idx:end_idx],
                'next_observations': traj['next_observations'][start_idx:end_idx],
                'qposes': traj['qposes'][start_idx:end_idx],
                'qvels': traj['qvels'][start_idx:end_idx]
            }
            clipped_trajs.append(clipped_traj)
            
    return clipped_trajs



def create_preference_pairs_based_on_reward(trajs1, trajs2, num_pairs=1000):
    # Create preference pairs based on cumulative rewards
    pairs = []
    labels = []
    
    for _ in range(num_pairs):
        # Randomly sample one trajectory from each set
        traj1 = random.choice(trajs1)
        traj2 = random.choice(trajs2)
        
        if sum(traj1['rewards']) > sum(traj2['rewards']):
            preference = 0
        else:
            preference = 1
        
        # Add both orderings with appropriate labels
        pairs.append((traj1, traj2))
        labels.append(preference)
        
    return pairs, labels


dataset_random = D4RLDataset(env_random)
dataset_medium = D4RLDataset(env_medium)
dataset_expert = D4RLDataset(env_expert)

trajectories_random = seperate_trajectories(dataset_random)
trajectories_medium = seperate_trajectories(dataset_medium)
trajectories_expert = seperate_trajectories(dataset_expert)

# Clip all trajectory sets
trajectories_random_clipped = clip_trajectories(trajectories_random, target_length=traj_target_length)
trajectories_medium_clipped = clip_trajectories(trajectories_medium, target_length=traj_target_length) 
trajectories_expert_clipped = clip_trajectories(trajectories_expert, target_length=traj_target_length)


trajectories_all = trajectories_random_clipped + \
                   trajectories_medium_clipped + \
                   trajectories_expert_clipped

# easy examples
pairs, labels = create_preference_pairs_based_on_reward(
    trajectories_all, trajectories_all, num_pairs=int(easy_example_ratio*num_pairs)
)

easy_count = len(pairs)

# add hard examples
pairs_medium, labels_medium = create_preference_pairs_based_on_reward(
    trajectories_medium_clipped, trajectories_medium_clipped, 
    num_pairs=int(0.7*(1-easy_example_ratio)*num_pairs)
)

pairs_expert, labels_expert = create_preference_pairs_based_on_reward(
    trajectories_expert_clipped, trajectories_expert_clipped, 
    num_pairs=int(0.3*(1-easy_example_ratio)*num_pairs)
)


pairs += pairs_medium + pairs_expert
labels += labels_medium + labels_expert

print(f"Total number of preference pairs: {len(pairs)}, among which easy preference pairs count: {easy_count}")
# print(f"Total number of labels: {len(labels)}")


# Create dataset instance
preference_dataset = PreferenceDataset(pairs, labels)

