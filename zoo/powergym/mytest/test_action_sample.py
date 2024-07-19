import torch
from torch.distributions import Normal, Independent
'''
这行代码创建了一个独立分布 dist。
Normal(mu, sigma) 创建了一个三维正态分布，其中每个维度的均值和标准差分别由 mu 和 sigma 定义。
Independent(Normal(mu, sigma), 1) 将这个三维正态分布转换为一个独立的分布对象，
其中 1 表示有几个互相独立的维度（在这种情况下是3个）。
'''
mu = torch.tensor([0.1, 0.1, 0.2])
sigma = torch.tensor([0.1, 0.1, 0.2])
dist = Independent(Normal(mu, sigma), 1)
'''
从分布 dist 中采样5个样本
'''
samples = dist.sample((5,))
print(samples)  # [5, 3]
sampled_actions_before_tanh = dist.sample(torch.tensor([20])) # [20,3]
# 这段代码定义了一个二维的独立正态分布，维度1,2的均值为 0.1，标准差为 0.1。维度3的均值是0.2,标准差是0.2
sampled_actions = torch.tanh(sampled_actions_before_tanh)
y = 1 - sampled_actions.pow(2) + 1e-6
# keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
log_prob = dist.log_prob(sampled_actions_before_tanh).unsqueeze(-1) #[20,] --> [20,1]
log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

import torch
import torch.distributions as dists
# 创建一个未归一化的Tensor
unnormalized_probs = torch.randn([2, 2])
# 使用softmax来归一化这个Tensor，使其成为有效的概率分布
probs = torch.softmax(unnormalized_probs, dim=1)
# 创建Categorical分布
dist = dists.categorical.Categorical(probs=probs)    # dist tensor(1,4) actions tensor(1,) eg: Tensor(3)

actions = torch.Tensor([[1], [0]])
print(dist.log_prob(actions))

actions = torch.Tensor([1, 0])
print(dist.log_prob(actions))

if __name__ == '__main__':
    pass