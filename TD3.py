import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import PositionalEncoding, check_for_nan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dims=256, K=10, N=15, M=3, T=3, nlayer=1, nhead=1):
        super(Actor, self).__init__()
        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.T_input_dim = int(state_dim / T)

        # ?encoder
        # d_model: the number of expected features in the input (required).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.T_input_dim, nhead=nhead)
        # ? d_model 在nlp里面是embedding后的维度 代表有一个seq len 有50个维度特征，在这里是 [caching, input, output,requirement,times]
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayer)
        self.pos_encoder = PositionalEncoding(self.T_input_dim)   # 位置编码
        # 1.请求趋势矩阵，这个是送入transformer的 NxK
        # 2.上传的数据量，长度为N
        # 3.上传计算量，长度为N
        # 4.每个用户的最大容忍 数组长度N
        # 5.每个用户对每个服务器的距离，信道增益，长度为NxM
        # 6.上时隙的服务缓存决策MxK
        # # ? LayerNorm是对同一batch的不同特征的进行归一化，我觉得这里像batchnorm，因为都是对同一特征进行归一化
        self.ffn_norm1 = nn.LayerNorm(int(N * K), eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(int(N), eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(int(N), eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(int(N), eps=1e-6)
        self.ffn_norm5 = nn.LayerNorm(int(N * M), eps=1e-6)
        self.ffn_norm6 = nn.LayerNorm(int(K * M), eps=1e-6)

        self.l1 = nn.Linear(state_dim, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, hidden_dims)
        self.l3 = nn.Linear(hidden_dims, action_dim)

        self.max_action = max_action

    def forward(self, x):
        # print('self.K,self.N,self.M,self.T', self.K, self.N, self.M, self.T)#self.K,self.N,self.M,self.T 12 30 3 3
        check_for_nan(x, "input")
        batch_size = x.shape[0]  # Batch_size
        state = []
        for i in range(self.T):
            state.append(
                x[:, (i) * self.T_input_dim: (i + 1)
                  * self.T_input_dim].unsqueeze(1)
            )

        x = torch.cat(state, 1)  # 调整顺序
        x = x.reshape(batch_size, self.T, self.T_input_dim)
        x = torch.cat(
            [
                self.ffn_norm1(x[:, :, : self.N * self.K]),
                self.ffn_norm2(
                    x[:, :, self.N * self.K: self.N * self.K + self.N]),
                self.ffn_norm3(
                    x[:, :, self.N * self.K + self.N: self.N * self.K + 2 * self.N]
                ),
                self.ffn_norm4(
                    x[:, :, self.N * self.K + 2 *
                        self.N: self.N * self.K + 3 * self.N]
                ),
                self.ffn_norm5(
                    x[
                        :,
                        :,
                        self.N * self.K
                        + 3 * self.N: self.N * self.K
                        + 3 * self.N
                        + self.N * self.M,
                    ]
                ),
                self.ffn_norm6(x[:, :, -self.K * self.M:]),
            ],
            -1,
        )
        # ? state先embedding,后position embedding，再进入transformer encoder
        # print(xu.shape)#torch.Size([256, 3, 50])
        # 进入到pos时的shape是[3, 256, 50]，encoder出来时shape不变
        embed_s = self.transformer_encoder1(
            self.pos_encoder(x.transpose(0, 1))
        )  # 输入序列的形状：(sequence length, batch size, feature size)
        # print(embed_s.shape)# torch.Size([3, 256, 50]) 第一个像 seq len
        embed_s = embed_s.transpose(0, 1)
        # flattern 来喂进线性层，变成（bs，dim）
        embed_s = torch.flatten(embed_s, start_dim=1)
        check_for_nan(embed_s, "000")
        a = F.relu(self.l1(embed_s))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=256, nhead=1,
                 nlayer=1,
                 K=10, T=3,
                 N=15,
                 M=3,):
        super(Critic, self).__init__()
        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.T_input_dim = int(state_dim / T)
        # ?encoder
        # d_model: the number of expected features in the input (required).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.T_input_dim, nhead=nhead)
        # ? d_model 在nlp里面是embedding后的维度 代表有一个seq len 有50个维度特征，在这里是 [caching, input, output,requirement,times]
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayer
        )

        # d_model: the number of expected features in the input (required).
        encoder_layer_ = nn.TransformerEncoderLayer(
            d_model=self.T_input_dim, nhead=nhead)
        # ? d_model 在nlp里面是embedding后的维度 代表有一个seq len 有50个维度特征，在这里是 [caching, input, output,requirement,times]
        self.transformer_encoder_ = nn.TransformerEncoder(
            encoder_layer_, num_layers=nlayer
        )

        self.pos_encoder = PositionalEncoding(self.T_input_dim)  # 位置编码

        # # ? LayerNorm是对同一batch的不同特征的进行归一化，我觉得这里像batchnorm，因为都是对同一特征进行归一化
        # # ? LayerNorm是对同一batch的不同特征的进行归一化，我觉得这里像batchnorm，因为都是对同一特征进行归一化
        self.ffn_norm1 = nn.LayerNorm(int(N * K), eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(int(N), eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(int(N), eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(int(N), eps=1e-6)
        self.ffn_norm5 = nn.LayerNorm(int(N * M), eps=1e-6)
        self.ffn_norm6 = nn.LayerNorm(int(K * M), eps=1e-6)

        self.embeds_action = nn.Linear(
            action_dim, action_dim, bias=False
        )  # 这是线性层
        self.embeds_action_ = nn.Linear(
            action_dim, action_dim, bias=False
        )  # 这是线性层

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, hidden_dims)
        self.l3 = nn.Linear(hidden_dims, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dims)
        self.l5 = nn.Linear(hidden_dims, hidden_dims)
        self.l6 = nn.Linear(hidden_dims, 1)

    def forward(self, x, action):
        batch_size = x.shape[0]  # Batch_size
        state = []
        for i in range(self.T):
            state.append(
                x[:, (i) * self.T_input_dim: (i + 1)
                  * self.T_input_dim].unsqueeze(1)
            )

        x = torch.cat(state, 1)  # 调整顺序
        x = x.reshape(batch_size, self.T, self.T_input_dim)
        x = torch.cat(
            [
                self.ffn_norm1(x[:, :, : self.N * self.K]),
                self.ffn_norm2(
                    x[:, :, self.N * self.K: self.N * self.K + self.N]),
                self.ffn_norm3(
                    x[:, :, self.N * self.K + self.N: self.N * self.K + 2 * self.N]
                ),
                self.ffn_norm4(
                    x[:, :, self.N * self.K + 2 *
                        self.N: self.N * self.K + 3 * self.N]
                ),
                self.ffn_norm5(
                    x[
                        :,
                        :,
                        self.N * self.K
                        + 3 * self.N: self.N * self.K
                        + 3 * self.N
                        + self.N * self.M,
                    ]
                ),
                self.ffn_norm6(x[:, :, -self.K * self.M:]),
            ],
            -1,
        )
        x_ = x.clone()

        # ? state先embedding,后position embedding，再进入transformer encoder
        # print(xu.shape)#torch.Size([256, 3, 50])
        # 进入到pos时的shape是[3, 256, 50]，encoder出来时shape不变
        embed_s = self.transformer_encoder(self.pos_encoder(x.transpose(0, 1)))
        # print(embed_s.shape)# torch.Size([3, 256, 50]) 第一个像 seq len
        embed_s = embed_s.transpose(0, 1)
        embed_s = torch.flatten(embed_s, start_dim=1)  # flattern 来喂进线性层
        action1 = self.embeds_action(action)

        # 进入到pos时的shape是[3, 256, 50]，encoder出来时shape不变
        embed_s_ = self.transformer_encoder_(
            self.pos_encoder(x_.transpose(0, 1)))
        # print(embed_s.shape)# torch.Size([3, 256, 50]) 第一个像 seq len
        embed_s_ = embed_s_.transpose(0, 1)
        # flattern 来喂进线性层，直接全部喂进去，没有concat其他东西
        embed_s_ = torch.flatten(embed_s_, start_dim=1)
        action2 = self.embeds_action_(action)

        sa1 = torch.cat([embed_s, action1], 1)
        sa2 = torch.cat([embed_s_, action2], 1)

        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, x, action):
        batch_size = x.shape[0]  # Batch_size
        state = []
        for i in range(self.T):
            state.append(
                x[:, (i) * self.T_input_dim: (i + 1)
                  * self.T_input_dim].unsqueeze(1)
            )

        x = torch.cat(state, 1)  # 调整顺序
        x = x.reshape(batch_size, self.T, self.T_input_dim)
        x = torch.cat(
            [
                self.ffn_norm1(x[:, :, : self.N * self.K]),
                self.ffn_norm2(
                    x[:, :, self.N * self.K: self.N * self.K + self.N]),
                self.ffn_norm3(
                    x[:, :, self.N * self.K + self.N: self.N * self.K + 2 * self.N]
                ),
                self.ffn_norm4(
                    x[:, :, self.N * self.K + 2 *
                        self.N: self.N * self.K + 3 * self.N]
                ),
                self.ffn_norm5(
                    x[
                        :,
                        :,
                        self.N * self.K
                        + 3 * self.N: self.N * self.K
                        + 3 * self.N
                        + self.N * self.M,
                    ]
                ),
                self.ffn_norm6(x[:, :, -self.K * self.M:]),
            ],
            -1,
        )
        # ? state先embedding,后position embedding，再进入transformer encoder
        # print(xu.shape)#torch.Size([256, 3, 50])
        # 进入到pos时的shape是[3, 256, 50]，encoder出来时shape不变
        embed_s = self.transformer_encoder(self.pos_encoder(x.transpose(0, 1)))
        # print(embed_s.shape)# torch.Size([3, 256, 50]) 第一个像 seq len
        embed_s = embed_s.transpose(0, 1)
        embed_s = torch.flatten(embed_s, start_dim=1)  # flattern 来喂进线性层
        action1 = self.embeds_action(action)

        sa = torch.cat([embed_s, action1], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2, hidden_dims=256,
            a_lr=3e-4, c_lr=3e-4, K=10, N=15, M=3, T=3, **kwargs):

        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim,
                           max_action, hidden_dims=hidden_dims, K=K, N=N, M=M, T=T).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), a_lr)

        self.critic = Critic(state_dim, action_dim,
                             hidden_dims=hidden_dims,  K=K, N=N, M=M, T=T).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), c_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(
            state).unsqueeze(0).to(device)
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
