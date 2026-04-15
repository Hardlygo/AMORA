import numpy as np
import torch
import random
import os



class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def rescale_action(action, env):
    """
    对神经网络输出的动作输出到环境需要的范围
    """
    action_space = env.action_space

    # (caching,offloading,uplink,downlink,frequency)
    caching_offloading_low = np.zeros((2*env.num_service_type,))
    caching_offloading_high = np.ones((2*env.num_service_type,))

    low = np.concatenate(
        (caching_offloading_low, action_space["uplink"].low, action_space["downlink"].low, action_space["frequency"].low))
    high = np.concatenate(
        (caching_offloading_high, action_space["uplink"].high, action_space["downlink"].high, action_space["frequency"].high))
    action_range = [low, high]

    return action * (action_range[1] - action_range[0]) / 2.0 + (action_range[1] + action_range[0]) / 2.0

# [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
#        4., 4., 4., 4., 4., 4., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.])]


def normal_action(action, env):
    """
    对sample出的动作转化到[-1,1]
    """
    action_space = env.action_space

    # (caching,offloading,uplink,downlink,frequency)
    caching_offloading_low = np.zeros((2*env.num_service_type,))
    caching_offloading_high = np.ones((2*env.num_service_type,))

    low = np.concatenate(
        (caching_offloading_low, action_space["uplink"].low, action_space["downlink"].low, action_space["frequency"].low))
    high = np.concatenate(
        (caching_offloading_high, action_space["uplink"].high, action_space["downlink"].high, action_space["frequency"].high))
    action_range = [low, high]
    return (action - ((action_range[1] + action_range[0]) / 2.0)) / ((action_range[1] - action_range[0]) / 2.0)
