import numpy as np
from gymnasium import spaces
from GeneticSolver import GeneticSolver
from env_utils import *
from covxpy_loss import *
# from solve_convex import *
from ChannelAllocation import minimize_total_transmission_time


class Environment(object):

    def __init__(self, user_N, server_M, service_K, T, channels, bandwidth) -> None:
        super(Environment, self).__init__()
        self.user_N = user_N
        self.server_M = server_M
        self.service_K = service_K
        self.T = T
        self.B = [bandwidth for i in range(server_M)]  # 40MHz
        self.ES2Clod_rate = 10  # 8  # 15  # Mbit/s
        self.Clod_computingrate = 5000  # 10g MHz
        self.num_channels = channels
        #!初始化服务器集群
        self.Mu_K, self.Space_K = gen_service_attribute_list(
            service_K)  # 根据服务个数生成与服务相关的参数，计算计算需求，存储需求
        self.frequency_M, self.storage_M = gen_server_attribute_list(
            server_M)  # 根据服务器个数生成与服务器相关的参数，计算能力，存储能力
        self.Local_Frequency = gen_user_attribute_list(
            user_N)  # 根据用户个数生相关的参数，计算能力
        self.action_space = spaces.Dict({'offloading': spaces.Box(low=-1.0, high=1.0, shape=(
            user_N * server_M,)), 'caching': spaces.MultiBinary(server_M*service_K)})  # 卸载决策是连续决策输出一个长度为NXM的数组然后转为矩阵取每行的最大值的下标就是卸载的目标服务器，服务缓存决策是决定每个服务是否缓存在当前的服务，直至内存满为止

        self.action_space.seed(123)  # 设置一下种子
        # ? state相关的：
        # ? 1.请求趋势矩阵，这个是送入transformer的 NxK
        # ? 2.上传的数据量，长度为N
        # ? 3.上传计算量，长度为N
        # ? 4.每个用户的最大容忍 数组长度N
        # ? 5.每个用户对每个服务器的距离，信道增益，长度为NxM
        # ? 6.上时隙的服务缓存决策MxK
        self.channel_niose = 1e-3 * \
            pow(10., -174./10.)   # 信道噪声1e-13
        self.trans_power = 1.0  # 0.5W  # 0.1W  # 0.5W  # 0.1W

        self.T_input_dim = user_N*service_K+user_N + \
            user_N+user_N+user_N*server_M+server_M*service_K
        self.action_dim = user_N*server_M+server_M*service_K  # 卸载决策和缓存决策
        self._max_episode_steps = 100  # 一个episode 102步
        self.current_step = 0
        # 初始化 GeneticSolver
        self.ga_solver = GeneticSolver(self, population_size=60, generations=90,
                                       crossover_rate=0.9, mutation_rate=0.08, elite_size=10)

    def step(self, state, action, start_use_agent=True):
        '''
        1.根据动作获得任务卸载NM和缓存矩阵MK
        2.根据状态获取凸优化的参数
        partial_offloading_convex(N,M,K,F_N,F_ES,Price_M,Mu_K,Lamda_NK,Input_data,Caching_dec,Rate_NM,beta=1e5)
        '''
        print('action', action.shape)
        s_t = state[-self.T_input_dim:]  # 最后的那个self input dim 就是当前时隙的请求state
        T_lamda_NK = (s_t[:self.user_N *
                          self.service_K][:]).reshape((self.user_N, self.service_K))  # 当前时隙的请求矩阵 n*k
        action = np.array(action)  # 转换到np里面，它是一个一维数组
        o_t = action[:self.user_N *
                     self.server_M].reshape((self.user_N, self.server_M))  # 卸载
        o_t = np.argmax(o_t, axis=1)  # 每一行最大值的索引 卸载决策
        o_t_mid = np.zeros((self.user_N, self.server_M))
        # 使用np.arange生成行索引
        rows = np.arange(self.user_N)
        # 使用布尔索引更新o_t_mid数组
        o_t_mid[rows, o_t] = 1
        o_t = o_t_mid[:]  # 转换成N*M的01数组

        # print(action.shape)
        # print('o_t', o_t.shape, '\n', o_t)
        self.current_step += 1
        a_tplus1 = action[self.user_N *
                          self.server_M:][:].reshape((self.server_M, self.service_K))  # 缓存

        #!###############################################################################
        # requet_to_service_count = np.sum(
        #     T_lamda_NK, axis=0)  # shape = (service_K,) 对j排序，按照在服务器M中对不同J的访问量决定j循环的循序，然后再进行约束判断
        # # 按照服务被请求的次数从大到小排序服务ID
        # sorted_idx = np.argsort(-requet_to_service_count)
        # # 使用numpy.where函数进行条件赋值
        # a_tplus1 = np.where(a_tplus1 > 0, 1, 0)  # <=0的会等于0,>0会等于1,因为缓存最大值是01
        # deploy_mask = np.zeros_like(a_tplus1)   # 输出的0/1部署矩阵
        # left_sapce_array = []
        # for i in range(self.server_M):  # 每个服务器独立排序
        #     left_space = self.storage_M[i]
        #     # 依次尝试部署
        #     for j in sorted_idx:
        #         require = self.Space_K[j]
        #         if a_tplus1[i, j] == 1 and left_space >= require:  # 这样能保证left_space一直能大于0
        #             deploy_mask[i, j] = 1
        #             left_space -= require
        #         # else: 跳过这项，不部署
        #     left_sapce_array.append(left_space)
        #!###############################################################################

        #!###############################################################################
        # a_tplus1 = np.round(a_tplus1)  # <=0.5的会等于0,>0.5会等于1,因为缓存最大值是01
        # # 使用numpy.where函数进行条件赋值
        # a_tplus1 = np.where(a_tplus1 > 0, 1, 0)  # <=0的会等于0,>0会等于1,因为缓存最大值是01
        # # 根据缓存决策来消耗内存
        # left_sapce_array = []
        # for i in range(self.server_M):
        #     left_space = self.storage_M[i]
        #     for j in range(self.service_K):#!对j排序，按照在服务器M中对不同J的访问量决定j循环的循序，然后再进行约束判断
        #         #!能缓存尽量缓存，无法缓存了置决策为0
        #         # 缓存服务，减去空间
        #         if a_tplus1[i, j] == 1 and left_space >= self.Space_K[j]:  # 这样能保证left_space一直能大于0
        #             left_space -= self.Space_K[j]
        #             # caching_tplus1[i]=1
        #         else:
        #             a_tplus1[i, j] = 0
        #     left_sapce_array.append(left_space)
        #!###############################################################################
        # ?#########################################################################################
        deploy_mask = np.zeros_like(a_tplus1)   # 输出的0/1部署矩阵
        left_sapce_array = []
        for i in range(self.server_M):  # 每个服务器独立排序
            left_space = self.storage_M[i]
            # 取出优先级 score
            scores = a_tplus1[i]  # shape = (service_K,)
            # 按 score 从大到小排序服务ID
            sorted_idx = np.argsort(-scores)
            # 依次尝试部署
            for j in sorted_idx:
                require = self.Space_K[j]
                if left_space >= require:
                    deploy_mask[i, j] = 1
                    left_space -= require
                # else: 跳过这项，不部署
            left_sapce_array.append(left_space)
        # ?#########################################################################################
        print(f'm1 deployment decisions: {deploy_mask[0]}')
        usage_space = 1 - \
            np.sum((np.array(left_sapce_array)/self.storage_M)) / \
            len(self.storage_M)
        # ? np.concatenate([lamda_ik_matrix.flatten(), input_data.flatten(), computing_cycle.flatten(),
        # ?  max_tolerance.flatten(),  g_N_M.flatten()])

        # user_N*service_K+user_N + user_N+user_N+user_N*server_M+server_M*service_K

        g_N_M = (s_t[-self.user_N*self.server_M-self.server_M*self.service_K:-self.server_M *
                 self.service_K]).reshape((self.user_N, self.server_M))  # 倒数user_N*server_M为信道增益
        # pg_N_M = self.trans_power*g_N_M  # pg

        # 根据卸载决策决定干扰
        # o_pgNM = o_t * pg_N_M  # 乘以卸载决策（对应位置相乘）
        # sum_o_pgNM = np.sum(o_pgNM)  # 一个数组的每项和
        # inference_M = []
        # for i in range(self.server_M):
        #     sum_o_pgNj = np.sum(o_pgNM[:, i])  # 取出某一列元素并求和
        #     inference_j = sum_o_pgNM-sum_o_pgNj  # 因为服务器j的干扰是除去连接他的其他所有干扰和
        #     inference_M.append(inference_j)
        # inference_M = np.array(inference_M)

        # 固定的干扰Computation Off-Loading in Resource-Constrained
        # Edge Computing Systems Based on Deep
        # Reinforcement Learning
        # inference_M = np.ones((self.server_M,))*1e-9

        # inference_Mplus_noise = inference_M + \
        #     np.ones((self.server_M,))*self.channel_niose
        # channel_efficiency = np.zeros((self.user_N, self.server_M))  # 形状N*M

        # for i in range(self.user_N):
        #     for j in range(self.server_M):
        #         channel_efficiency[i, j] = math.log2(
        #             1+(pg_N_M[i, j]/(inference_Mplus_noise[j])))

        # print('channel_efficiency', channel_efficiency.shape, channel_efficiency)

        requst_size_N = s_t[self.user_N * self.service_K:self.user_N *
                            self.service_K+self.user_N][:]  # 当前用的是请求流量
        requst_size_N = requst_size_N[:, np.newaxis]  # 加一维度，变成N*1
        # print('requst_size_N', requst_size_N)

        computing_size_N = s_t[self.user_N * self.service_K +
                               self.user_N:self.user_N * self.service_K+2*self.user_N][:]  # 当前用的是计算流量
        computing_size_N = computing_size_N[:, np.newaxis]  # 加一维度，变成N*1，多除了100

        # print('computing_size_N', computing_size_N)
        caching_decision = (s_t[-self.server_M*self.service_K:]
                            ).reshape((self.server_M, self.service_K))  # 上一个时隙的缓存决策
        if_can_offloading = np.dot(
            T_lamda_NK, caching_decision.T)  # 用户生成了该服务，且服务器缓存了该服务 N*K X M*K==>N*M
        # print('if_can_offloading', if_can_offloading)
        hit_rate = np.sum(o_t * if_can_offloading)
        print('hit rate:', hit_rate)

        #!信道分配开始
        # ? 1先根据卸载决策确定每个用户的卸载服务器，这样就能确定信道增益了
        o_gNM = o_t * g_N_M  # 乘以卸载决策（对应位置相乘）NxM对应位置相乘
        o_pg_N_M = self.trans_power*o_gNM  # pg
        o_gN1 = o_pg_N_M[np.arange(o_pg_N_M.shape[0]), np.argmax(
            o_pg_N_M, axis=1)].reshape(-1, 1)  # NxM转为Nx1
        o_pg_N_L = np.repeat(o_gN1, self.num_channels, axis=1)

        if start_use_agent is not True:  # 如果还是在随机探索部分，则随机信道分配
            # best_channel_allocatte_action = np.random.randint(
            #     0, self.num_channels, size=self.user_N)  # 初始随机信道分配
            best_channel_allocatte_action = minimize_total_transmission_time(
                o_pg_N_L, requst_size_N[:, 0], self.channel_niose, max_iter=3, threshold=1.0)
        else:  # 否则使用贪心算法
            best_channel_allocatte_action = minimize_total_transmission_time(
                o_pg_N_L, requst_size_N[:, 0], self.channel_niose, max_iter=100, threshold=1.0)
        rates = self._compute_rate(
            best_channel_allocatte_action, gains=o_gN1[:, 0])
        delay = self._compute_time(rates, data=requst_size_N[:, 0])
        #!信道分配结束
        # ? 1先根据卸载决策确定每个用户的卸载服务器，这样就能确定信道增益了
        # o_gNM = o_t * g_N_M  # 乘以卸载决策（对应位置相乘）NxM对应位置相乘
        # o_gN1 = o_gNM[np.arange(o_gNM.shape[0]), np.argmax(
        #     o_gNM, axis=1)].reshape(-1, 1)  # NxM转为Nx1
        # best_channel_allocatte_action, _, gains, datas = self.ga_solver.run(
        #     gains=o_gN1[:, 0], data=requst_size_N[:, 0])

        # rates = self._compute_rate(best_channel_allocatte_action, gains)
        # delay = self._compute_time(rates, datas)
        delay_wireless = np.sum(delay)  # 无线传输时延
        # 卸载计算时延
        # 计算分配的带宽
        # 分配的计算能力
        # bandwidth_NM, delay_wireless = calculate_bandwidth(self.user_N, self.server_M, self.B,
        #                                                    o_t, requst_size_N/10, channel_efficiency)

        if start_use_agent is not True:  # 如果还是在随机探索部分，
            frequency_NM, delay_frequency = generate_frequency_and_delay(self.user_N, self.server_M, self.frequency_M,
                                                                         o_t, if_can_offloading, computing_size_N)  # !self.frequency_M/1000,,,computing_size_N/1000
        else:
            # frequency_NM, delay_frequency = calculate_frequency(self.user_N, self.server_M, self.frequency_M/1000,
            #                                                           o_t, if_can_offloading, computing_size_N/1000)
            frequency_NM, delay_frequency, _, _ = calculate_frequency_closed_form(self.user_N, self.server_M, self.frequency_M,
                                                                                  o_t, if_can_offloading, computing_size_N)  # !computing_size_N/1000  self.frequency_M/1000

        # print('bandwidth_NM*channel_efficiency',bandwidth_NM*channel_efficiency)
        print('delay_wireless', delay_wireless, 'delay_frequency',
              delay_frequency)
        # sum_bandwidth_M = np.sum(bandwidth_NM, axis=0)
        frequency_NM = frequency_NM*o_t  # 没有卸载的用户计算频率置0
        sum_frequency_M = np.sum(frequency_NM, axis=0)
        # print('sum_bandwidth_M', sum_bandwidth_M)
        # print('sum_frequency_M', sum_frequency_M)

        # usage_bandwidth = np.sum(sum_bandwidth_M/self.B)/len(self.B)
        usage_frequency = (
            np.sum(sum_frequency_M/(self.frequency_M)))/len(self.frequency_M)  # !self.frequency_M/1000

        # print(bandwidth_NM, frequency_NM)
        o_if_not_can_offloading = o_t*(1-if_can_offloading)
        delay_E2Ctran = o_if_not_can_offloading * \
            (requst_size_N/self.ES2Clod_rate)
        delay_E2Ctran = np.sum(delay_E2Ctran)
        # print(delay_E2Ctran)

        delay_Cloudcomputing = np.sum(o_if_not_can_offloading *
                                      (computing_size_N/self.Clod_computingrate))

        # print(delay_E2Ctran)
        print('delay_Cloudcomputing', delay_Cloudcomputing)
        print('delay_E2Ctran', delay_E2Ctran)

        delay_localcomputing = np.sum(
            computing_size_N/self.Local_Frequency)

        delay_offloading = (delay_wireless+delay_frequency +
                            delay_E2Ctran+delay_Cloudcomputing)
        print('delay_localcomputing', delay_localcomputing/self.user_N)
        print('delay_offloading', delay_offloading/self.user_N)

        # reward = -delay_offloading / self.user_N+1.5 * \
        #     (usage_space+hit_rate/self.user_N)  # + 0.5*(usage_frequency)
        avg_delay = delay_offloading / self.user_N
        reward = -np.log1p(avg_delay) + 0.3 * usage_space + \
            1.3*(hit_rate / self.user_N)

        done = True if self.current_step == (
            self._max_episode_steps+self.T) else False
        s_tminu1 = state[self.T_input_dim:]
        # 生成新的请求时候要将请求历史更新，当前的请求历史替换掉最新的加入，将缓存决策加入
        next_state = self.gen_next_state(
            current_step=self.current_step, caching_decs=deploy_mask, s_tminu1=s_tminu1)

        # print(reward/self.user_N, done,  delay_offloading/self.user_N)

        return next_state, reward, done,  delay_offloading/self.user_N, usage_space, hit_rate/self.user_N, usage_frequency

    def reset(self):
        self.current_step = 0
        return self.gen_next_state(self.current_step)

    def gen_slot_T_state(self):
        lamda_ik_matrix, g_N_M, input_data, computing_cycle, max_tolerance = gen_state(
            self.user_N, self.server_M, self.service_K, self.Mu_K, self.Local_Frequency)  # 产生当前时隙的请求

        return np.concatenate([lamda_ik_matrix.flatten(), input_data.flatten(), computing_cycle.flatten(), max_tolerance.flatten(),  g_N_M.flatten()])

    def gen_next_state(self, current_step, caching_decs=None, s_tminu1=None):
        # ?因为是一条state含有T条历史，所以走完T步才有一条完整的state state=[st-2,st-1,st]
        s_t = self.gen_slot_T_state()  # 当前时隙最新的请求动态
        if current_step == 0:
            state = np.zeros((self.T_input_dim*self.T,))  # 历史个输入维度
            caching_decs_ = np.array(self.action_space.sample(
            )['caching']).reshape((self.server_M, self.service_K))
            for i in range(self.server_M):
                left_space = self.storage_M[i]
                for j in range(self.service_K):
                    #!能缓存尽量缓存，无法缓存了置决策为0
                    # 缓存服务，减去空间
                    # 这样能保证left_space一直能大于0
                    if caching_decs_[i, j] == 1 and left_space >= self.Space_K[j]:
                        left_space -= self.Space_K[j]
                    else:
                        caching_decs_[i, j] = 0
            state[-self.T_input_dim:] = np.concatenate(
                [s_t.flatten(), caching_decs_.flatten()])
            for i in range(self.T):  # 这三步是为了构造一个完整的state
                # (input_size_service,output_size_service,computing_service_current,num_times_current,caching)
                a_t = self.sample()
                # print(len(state))
                state, _, _, _, _, _, _ = self.step(state, a_t)
        else:
            state = np.concatenate(
                [s_tminu1.flatten(), s_t.flatten(), caching_decs.flatten()])

        return state

    def sample(self):
        offloading_act = np.array(self.action_space.sample(
        )['offloading']).reshape((self.user_N, self.server_M))
        # print(offloading_act)

        caching_act = np.array(self.action_space.sample(
        )['caching']).reshape((self.server_M, self.service_K))
        action = np.hstack([offloading_act.flatten(), caching_act.flatten()])
        # print('action', action.shape)
        return action

    def _compute_rate(self, actions, gains):
        rates = np.zeros(self.user_N)
        for u in range(self.user_N):
            c = actions[u]
            interference = sum(
                self.trans_power * gains[v] for v in range(self.user_N) if v != u and actions[v] == c)
            sinr = self.trans_power * gains[u] / \
                (self.channel_niose + interference)
            rates[u] = self.B[0] * np.log2(1 + sinr)
        return rates

    def _compute_time(self, rates, data):
        return data / rates

# service_K = 10
# server_M = 3
# user_N = 15
# slot_T = 5
# env = Enviroment(user_N=user_N, server_M=server_M,
#                  service_K=service_K, T=slot_T)
# state_dim = env.input_dim
# action_dim = env.action_space['offloading'].shape[0] + \
#     env.action_space['caching'].shape[0]

# # print(state_dim, action_dim) # 896 75
# # print(env.action_space.sample())
# state = env.reset()
# print('reset state', state.shape)
# offloading_act = np.array(env.action_space.sample(
# )['offloading']).reshape((user_N, server_M))
# # print(offloading_act)

# caching_act = np.array(env.action_space.sample(
# )['caching']).reshape((server_M, service_K))
# action = np.hstack([offloading_act.flatten(), caching_act.flatten()])
# # print(caching_act)

# env.step(state=state, action=action)
