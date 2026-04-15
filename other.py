import numpy as np
import random
import math


# def gen_task(service_type=None, distance=None):
#     '''
#     就生成一个数组类型的任务[input_size,computational_size,service_type,distance]
#     '''
#     input_size = np.random.uniform(low=5, high=10)  # ?单位Mb
#     input_size = round(input_size, 2)
#     computational_size = (input_size / 8) * 330  # ?单位M*cycles
#     if service_type is None:
#         service_type = np.random.zipf(1.46, 1)  # 对应的任务类型
#         service_type = np.clip(service_type, a_max=10, a_min=1)
#     if distance is None:
#         distance = np.random.uniform(low=80, high=500)  # ?单位m
#         distance = round(distance, 0)

#     return [input_size, computational_size, service_type, distance]


# def gen_input_ouput_data():
#     """
#     根据泊松分布生成每个任务的输入输出任务数据大小
#     到达率 λ∈[8, 12, 18, 24]
#     """
#     sizes = np.random.poisson(lam=[8, 12, 18, 24], size=(80, 1))
#     sizes = sizes.reshape(40, 2)  # * 40个任务，每个任务包含输入数据和结果数据
#     # print(sizes)
#     return sizes


def gen_input_ouput_data(tasks=40):
    """
    根据泊松分布生成每个任务的输入输出任务数据大小
    到达率 λ∈[8, 12, 18, 24]Mb
    """
    sizes = np.random.uniform(low=2.4, high=11.2, size=(
        tasks, 2))  # Joint Computation Offoading and ResourceAllocation in Multi-Edge Smart CommunitiesWith Personalized Federated DeepReinforcement Learning
    # sizes = np.random.uniform(low=2.4, high=9.6, size=(
    #     tasks, 2))  # Computation Rate Maximization for SCMA-AidedEdge Computing in loT Networks: A Multi-AgentReinforcement Learning Approach
    # sizes = np.random.uniform(low=3.2, high=8.0, size=(
    #     tasks, 2))  # BARGAIN-MATCH: A Game Theoretical Approachfor Resource Allocation and Task Offoadingin Vehicular Edge Computing Networks
    # sizes = np.random.uniform(low=0.8, high=4.8, size=(
    #     tasks, 2))#A Task Offloading Method Based on UserSatisfaction in C-RAN With Mobile Edge Computing
    # sizes = np.random.uniform(low=2.4, high=4.0, size=(
    #     tasks, 2))  # Decentralized Navigation With HeterogeneousFederated Reinforcement Learning forUAV-Enabled Mobile Edge Computing
    # # sizes = np.random.uniform(low=1., high=3.0, size=(
    #     tasks, 2))
    # sizes = np.random.uniform(low=0.1, high=2.0, size=(
    #     tasks, 2))  # 生成均匀分布数据Hierarchical Task Offloading for Vehicular Fog  Computing Based on Multi-Agent Deep  Reinforcement Learning
    # sizes = np.random.uniform(low=0.8, high=1.2, size=(
    #     tasks, 2))
    # sizes = np.random.poisson(
    #     lam=[6.0], size=(int(tasks), 2))  # Mb
    sizes = sizes.reshape(tasks * 2)  # * 40个任务，每个任务包含输入数据和结果数据
    np.random.shuffle(sizes)  # 按照第一个维度shuffle
    #     print(sizes)
    sizes = sizes.reshape((tasks, 2))

    return sizes


# print(gen_input_ouput_data())


def gen_service_space():
    """
    10个服务类型占用空间服从均匀分布
    只需要生成一次？因为占用空间在一个episode内不改变的
    40 80 G
    """
    # spaces = np.random.randint(40, 81, size=[10, 1])#生成离散整数
    spaces = np.random.uniform(40, 81, size=(10,))  # 生成连续整数
    # spaces = np.around(spaces, decimals=2)  # 保留两位小数
    spaces = np.around(spaces)  # 保留整数
    print(spaces, spaces.sum())
    return spaces


# gen_input_ouput_data()
# gen_service_space()


class Task(object):
    def __init__(self, input_size, output_size, service_type, require_resource=None):
        """
        任务包含 （输入数据，结果数据，计算需求，所属类型）
        """
        super().__init__()
        self.input_size = input_size  # * 1e6  # Mb化成b
        self.output_size = output_size  # * 1e6
        self.type = service_type
        if require_resource is None:
            # * 330  # 先从b化成byte然后乘以单位330
            # Mcycles (self.input_size / 8)
            self.require_resource = (self.input_size)
        else:
            self.require_resource = require_resource

    def __str__(self):
        return "(input:{},output:{},require_computing:{},type:{})".format(
            self.input_size, self.output_size, self.require_resource, self.type
        )


SERVICE_TYPE = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# >>> np.random.randint(2, size=10)
# array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random


def gen_40_task():
    data_size_arry = gen_input_ouput_data()

    arr_len = len(data_size_arry)
    task_arry = []
    for idx in range(arr_len):
        if len(task_arry) < arr_len:
            task_arry.append(None)
        task_i_type = random.choice(SERVICE_TYPE)  # 随机选择一个服务类型
        task = Task(
            input_size=data_size_arry[idx][0], output_size=data_size_arry[idx][1], service_type=task_i_type)
        task_arry[idx] = task

    return task_arry


def Zipf_sample(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    #  离散采样来模拟截断的Zipf
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max+1)  # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)            # normalized

    return np.random.choice(v, size=size, replace=True, p=p)


def gen_efficiency(channel_gain, trans_power):
    # print(111, channel_gain)
    # trans_power = 1e-3*pow(10., 26./10.)  # 最大传输功率（瓦特 20 #30
    noise_power = 1e-3*pow(10., -174./10.)  # 174 #2*1e-13#-114
    interfrence = 1e-12
    # 香农公式 20=20db=100mw是发射功率，-100=-100dbm/Hz是噪声功率
    efficiency2server_M = np.log2(
        1+((trans_power*channel_gain) / (noise_power)))  # +interfrence
    # print(222, efficiency2server_M)
    return efficiency2server_M


# def gen_efficiency(size):
#     distance = np.random.uniform(
#         low=50, high=150, size=(size, ))  # ?单位m
#     channel_gain = 128.1+37.6*np.log10(distance/1000)  # 信道增益
#     trans_power = 20
#     noise_power = 174
#     # 香农公式 20=20db=100mw是发射功率，-100=-100dbm/Hz是噪声功率
#     efficiency2server_M = np.log2(
#         1+((trans_power*(-channel_gain))/- noise_power))
#     return efficiency2server_M


def gen_40_task_zipf(size=40, service_sizes=12):
    """
    生成目标数量的task数组
    """
    data_size_arry = gen_input_ouput_data(size)  # 生成40个任务
    computation_density_array = np.random.uniform(500, 900, size=(
        size, ))  # MHz(500, 900() # 800, 1500
    #!两种不一样的选服务类型的方法
    #!泽夫分布
    # 40个任务每个索引对应的任务类型 ,a越大它的分布越不均匀 默认这里设置2.0，要试试1.6,1.8,1.89,1.92
    service_types = Zipf_sample(
        1.2, min=1, max=service_sizes, size=size)  # 与服务类型个数一致
    #!指定概率
    # service_types=np.random.choice([ 1, 2, 3, 4, 5, 6, 7, 8, 9,10],size,p=[0.6,0.105,0.105,0.03,0.03,0.03,0.03,0.03,0.02,0.02])# 当前的概率数组能收敛。
    arr_len = len(data_size_arry)
    task_arry = []  # p=[0.4,0.1,0.1,0.06,0.06,0.06,0.06,0.06,0.05,0.05] p=[0.25,0.15,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05]
    # p=[0.6,0.15,0.15,0.03,0.03,0.03,0.03,0.03,0.02,0.02]  p=[0.4,0.1,0.1,0.065,0.065,0.065,0.065,0.06,0.04,0.04]
    for idx in range(arr_len):
        if len(task_arry) < arr_len:
            task_arry.append(None)
        task = Task(
            input_size=data_size_arry[idx][0], output_size=data_size_arry[idx][
                1], service_type=service_types[idx], require_resource=computation_density_array[idx]  # * 330
        )
        task_arry[idx] = task

    return task_arry


# print(gen_40_task_zipf(60))


def parse_offload_caching_decisions(array, num_types):
    """
    传入长度为30的数组，输出长度为10 的数组
    要求是每三个取最大的那个
    """
    input_array = np.array(array)
    input_array = input_array.reshape((num_types, int(len(array)/num_types)))
    decisions = np.argmax(input_array, axis=1)
    return decisions  # 里面的元素可能是0 1 2


def t2r(t):
    y = 1000 * math.pow(0.001, t)
    return y


def t2r_low(t):
    """
    让奖励R集中在中间，这样更偏向与卸载大的，不卸载小的
    """
    # y = 20 * math.pow(0.5, t)
    # y = 100 * math.pow(0.01, t)
    y = 1000 * math.pow(0.001, t)
    return y


def resource_constraint_func(x):
    """
    辅助目标，用来尽可能利用资源又不超出限制的
    """
    # y1 = math.pow(100, -x)
    y1 = math.pow(100, -x)
    y = max(y1, 1) - 1 + 0.1 * x
    return y


def get_minus_rescle1(x, k=3, threhold=-0.12):
    """
    负惩罚的计算1
    """
    y = 0
    if x < threhold:
        y = k / threhold
    elif x < 0 and x >= threhold:
        y = k / x
    else:
        raise NotImplementedError("x has to be minus")
    return y


def get_minus_rescle(x, k=3):
    """
    负惩罚的计算2
    """
    y = 0
    if x < 0:
        y = k / x - 20
    else:
        raise NotImplementedError("x has to be minus")
    return y


def normalize_actions(actions, max_act):
    """
    Normalize the actions to ensure that the sum of each column equals the max_act.
    """
    normalized_actions = np.copy(actions)
    arr_sum = np.sum(normalized_actions)
    if arr_sum > max_act:  # 仅在超过阈值时缩放
        return (normalized_actions / arr_sum) * max_act
    else:
        return normalized_actions  # 否则保持原样
