
import cvxpy as cp
import numpy as np
import math
import mosek
import time


def calculate_bandwidth(user_N, server_M, B,
                        o_t, requst_size_T_lamda_NK, channel_efficiency):
    '''
    求分配的带宽和结果值
    '''
    bandwidth_NM = cp.Variable((user_N, server_M), nonneg=True)
    # frac_fm = cp.multiply(bandwidth_NM, channel_efficiency)
    # print('requst_size_T_lamda_NK', requst_size_T_lamda_NK)

    o_requst_size_T_lamda_NK = (o_t*requst_size_T_lamda_NK)/channel_efficiency
    # print('channel_efficiency', channel_efficiency)
    frac = cp.multiply(o_requst_size_T_lamda_NK, cp.inv_pos(bandwidth_NM))
    prob1 = cp.sum(frac)
    objective = cp.Minimize(prob1)
    # constraints = [bandwidth_NM <= 1,1-cp.sum(bandwidth_NM, axis=0) >= 0]

    constraints = [bandwidth_NM <= B[0]]
    for i in range(len(B)):
        constraints += [cp.sum(
            bandwidth_NM[:, i]) <= B[0]]  # 某一列的值不能大过最大值

    result = cp.Problem(objective, constraints)
    result.solve(solver=cp.MOSEK, qcp=True)  # MOSEK , verbose=True
    # print(11, np.array(bandwidth_NM.value))
    # print(22,  np.sum(o_requst_size_T_lamda_NK /
    #       (np.array(bandwidth_NM.value)*channel_efficiency)), result.value)
    return np.array(bandwidth_NM.value), result.value  # 结果是N*M


def calculate_frequency(user_N, server_M, F,
                        o_t, if_can_offloading, computing_size_T_lamda_NK):
    '''
    求分配的计算频率和结果值 mosek经常会因为精度问题导致求解失败
    '''
    print(user_N, server_M, F,
          o_t, if_can_offloading, computing_size_T_lamda_NK)
    frequency_NM = cp.Variable((user_N, server_M), nonneg=True)

    o_if_can_offloading = o_t*if_can_offloading
    # print('o_if_can_offloading', o_if_can_offloading)
    # print('o_if_can_offloading *computing_size_T_lamda_NK',
    #       o_if_can_offloading * computing_size_T_lamda_NK)

    frac = cp.multiply(o_if_can_offloading *
                       computing_size_T_lamda_NK, cp.inv_pos(frequency_NM))
    prob2 = cp.sum(frac)
    objective = cp.Minimize(prob2)
    constraints = [frequency_NM <= F[0]]
    #  x[:, :self.K*self.N*self.T]
    for i in range(len(F)):
        constraints += [cp.sum(
            frequency_NM[:, i]) <= F[0]]
    # constraints = [cp.sum(
    #     frequency_NM, axis=0) <= F]  # frequency_NM >= 0,
    result = cp.Problem(objective, constraints)
    # print('F', F)
    # print('constraints', result.constraints)
    # result.solve(solver=cp.MOSEK)  # MOSEK这个求解器精度不能小于1e-7
    result.solve(solver=cp.MOSEK, mosek_params={
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-2,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_MU_RED": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 100,
        "MSK_IPAR_LOG": 0
    })
    # print('np.array(frequency_NM.value)', np.array(frequency_NM.value))
    return np.array(frequency_NM.value), result.value


def generate_frequency_and_delay(user_N, server_M, F, o_t, if_can_offloading, computing_size_T_lamda_NK):
    """
    只对 o_t[i,j] * if_can_offloading[i,j] == 1 的位置分配资源
    并计算 delay = computing_size / frequency
    """
    frequency_NM = np.zeros((user_N, server_M))
    delay_NM = np.zeros((user_N, server_M))

    o_if_can = o_t * if_can_offloading  # shape (N, M)

    for j in range(server_M):
        valid_users = np.where(o_if_can[:, j] == 1)[0]
        print('valid_users', valid_users)
        n_valid = len(valid_users)
        if n_valid == 0:
            continue

        rand_alloc = np.random.rand(n_valid)
        rand_alloc /= rand_alloc.sum()
        allocation = rand_alloc * F[0]

        for idx, i in enumerate(valid_users):
            freq = allocation[idx]
            frequency_NM[i, j] = freq
            delay_NM[i, j] = computing_size_T_lamda_NK[i, 0] / freq

    return frequency_NM, np.sum(delay_NM)


def calculate_frequency_closed_form(user_N, server_M, F, o_t, if_can_offloading, computing_size_T_lamda_NK):
    """
    不用求解器的闭式解方法：每列频率按 sqrt(w_ij) 分配，满足约束
    """
    start_time = time.time()
    eps = 1e-8
    F_total = F[0]

    # 权重矩阵
    W = (o_t * if_can_offloading) * computing_size_T_lamda_NK  # shape: [N, M]

    f = np.zeros((user_N, server_M))
    for j in range(server_M):
        wj = W[:, j]
        mask = wj > 0
        sqrt_wj = np.sqrt(wj[mask])
        denom = np.sum(sqrt_wj) + eps

        # 按比例分配频率
        f[mask, j] = sqrt_wj / denom * F_total

    # 目标函数值
    cost = np.sum(W / (f + eps))
    usage = np.sum(f, axis=0) / F_total
    elapsed = time.time() - start_time

    print(f" 闭式解运行时间：{elapsed:.6f} 秒")
    # print(f"🎯 总延迟目标值：{cost:.6f}")
    # print(f"📊 每个服务器利用率：{usage}")

    return f, cost, usage, elapsed


def cvxpy_loss(user_N, server_M, service_K, lamda_ik_matrix, caching_decision, offloading_decision, channel_efficiency, input_arry, output_arry, cpu_arry, ig=3, e2c_rate=15, cloud_computing_rate=5, upper_bandwidth=20, upper_cpu=20, cycle_proportion=330):
    '''
    根据服务缓存和任务卸载决策用凸优化求解上传带宽和频率分配
    参数：
    N:MEC环境中用户数量
    M:MEC环境中边缘服务器数量
    K:MEC环境中服务的种类数
    F_N:用户的本地计算能力（cycles/s）shape为（N,1）
    F_ES:边缘服务器的计算能力（cycles/s）shape为（1,M）
    Price_M:边缘服务器的计算能力的价格（$/cycles）shape为（1，M）
    Mu_K:每种服务的计算需求（cycles/bit）shape为（K，1）
    Lamda_NK:用户任务-类型矩阵，每个用户生成一种类型任务，shape为（N，K）,元素为0或1
    Input_data:任务输入矩阵（Mbit）shape为（N,1）
    Caching_dec:服务缓存矩阵shape为（M，K）
    Rate_NM:用户到服务器的速率矩阵shape为（N,K）
    beta:速度对价格的相对重要性系数
    param:
        caching_decision: 服务缓存决策
        offloading_decision: 服务卸载决策
        input_arry:输入数据量(Mb)
        output_arry:计算结果数据量(Mb)
        cpu_arry:cpu需求量
        ig:上行和下行的频谱效率
        e2c_rate: ES到Cloud的数据传输速度
        cloud_computing_rate: 云中心的CPU计算速度
        upper_bandwidth: 最多可分配带宽
        upper_cpu: ES最多可分配的CPU
        cycle_proportion:每个类型任务的CPU需求和输入的关系

    result:
        1. total delay
        2. usage uplink 上带宽利用率
        3. usage downlink 下带宽利用率
        4. ES的cpu利用率
        5. 每一步的上带宽分配数组
        6. 每一步的下带宽分配数组
        7. 每一步的frequency分配数组
    '''
    # for i in range(len(offloading_decision)): #不存在了只缓存，不卸载
    #     if offloading_decision[i]>caching_decision[i]:
    #         offloading_decision[i]=0
    # 矩阵，元素大于等于0(user_N, server_M)
    bandwidth_NM = cp.Variable((user_N, server_M), nonneg=True)
    CPU_MK = cp.Variable((server_M, service_K), nonneg=True)

    if_can_offloading = np.dot(
        lamda_ik_matrix, caching_decision.T)  # 用户生成了该服务，且服务器缓存了该服务

    if_canot_offloading = 1-if_can_offloading

    k1 = caching_decision*offloading_decision
    k2 = 1-offloading_decision

    param1 = input_arry/ig
    param2 = output_arry/ig

    param3 = cpu_arry/(1000*cycle_proportion)
    b = input_arry/e2c_rate+output_arry/e2c_rate+cpu_arry / \
        (cloud_computing_rate * 1000*cycle_proportion)

    #!变量定义
    up_bandwidth_allocated = cp.Variable(10)
    down_bandwidth_allocated = cp.Variable(10)
    cpu_allocated = cp.Variable(10)  # ?定义决策变量个数

    # 第一个问题，优化Cpu分配
    a = cp.multiply(k1, cp.multiply(param3, cp.inv_pos(cpu_allocated)))

    ob = cp.sum(a)
    # print("curvature of ob1:", ob.curvature)

    objective = cp.Minimize(ob)

    constraints = [cpu_allocated >= 0, cp.sum(cpu_allocated) <= upper_cpu]
    # print("constraints",constraints)
    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver=cp.ECOS)  # SCS
    # The optimal value for x is stored in x.value.
    # print("status:", prob.status) #status: optimal_inaccurate

    # print(cpu_allocated.value)
    # print("total cpu_allocated:",sum(cpu_allocated.value))
    # print(result)

    # 优化问题2 上下带宽资源分配
    bc = cp.multiply(k1, (cp.multiply(param1, cp.inv_pos(
        up_bandwidth_allocated))+cp.multiply(param2, cp.inv_pos(down_bandwidth_allocated))))
    c = cp.multiply(k2, (cp.multiply(param1, cp.inv_pos(up_bandwidth_allocated)) +
                         cp.multiply(param2, cp.inv_pos(down_bandwidth_allocated))+b))

    ob2 = cp.sum(c+bc)
    # print("curvature of ob2:", ob2.curvature)

    objective1 = cp.Minimize(ob2)

    constraints1 = [up_bandwidth_allocated >= 0, down_bandwidth_allocated >= 0, cp.sum(
        up_bandwidth_allocated) <= upper_bandwidth, cp.sum(down_bandwidth_allocated) <= upper_bandwidth]

    # print("constraints1",constraints1)
    prob1 = cp.Problem(objective1, constraints1)

    # The optimal objective is returned by prob.solve().
    result1 = prob1.solve(solver=cp.ECOS)  # SCS ,verbose=True
    # The optimal value for x is stored in x.value.
    # print("status:", prob1.status) #status: optimal_inaccurate
    # print(up_bandwidth_allocated.value)
    # print("total up_bandwidth_allocated:",sum(up_bandwidth_allocated.value))
    # print(down_bandwidth_allocated.value)
    # print("total down_bandwidth_allocated:",sum(down_bandwidth_allocated.value))

    # print(result1)

    return result+result1, sum(up_bandwidth_allocated.value)/upper_bandwidth, sum(down_bandwidth_allocated.value)/upper_bandwidth, sum(cpu_allocated.value*k1)/upper_cpu, up_bandwidth_allocated.value, down_bandwidth_allocated.value, cpu_allocated.value


# caching=np.array([1, 1 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0])
# offloading=np.array([1, 1 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0])

# Z_U=np.array([374. ,   115.   ,  78.   ,   0.  ,    0.  ,    0.   ,   0.  ,    0.   ,   0.,90.])
# Z_D=np.array([ 390.   ,  63.   ,  55. ,    17. ,    23.  ,    0. ,    18. ,    14., 0.  ,   20.])
# W=np.array([51.75 ,  15.375  , 5.25 ,   0.    ,  3.625  , 2.375  , 3.625, 0.   ,   0.    ,  0. ])
# ig=3
# dalay,loss1,loss2,loos3=cvxpy_loss(caching,offloading,Z_U,Z_D,W)

# #print(dalay,loss1,loss2,loos3)
