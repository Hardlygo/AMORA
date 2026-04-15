import numpy as np
import random
import math


def Zipf_sample(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    #  离散采样来模拟截断的Zipf
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    返回每个用户它的当前时隙访问的是第几个服务
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max + 1)  # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)  # normalized

    return np.random.choice(v, size=size, replace=True, p=p)


def gen_user_attribute_list(user_N):
    # return np.random.uniform(500, 1000, size=(user_N))  # 单位MHz 本地计算的能力
    return np.ones((user_N,)) * 800  # 单位MHz 本地计算的能力


def gen_service_attribute_list(service_k):
    """
    生成K中服务相关属性：1属性,2空间
    """
    Mu_K = np.random.randint(low=500, high=800, size=(service_k,))  # cycles/bit
    Space_K = np.random.randint(low=10, high=20, size=(service_k,))  # G
    return Mu_K, Space_K


def gen_server_attribute_list(server_M):
    frequency_M = np.ones((server_M)) * 40000  # 20000  # 单位MHz
    storage_M = np.ones((server_M)) * 100  # 单位G
    return frequency_M, storage_M


def gen_state(user_N, server_M, service_K, Mu_K, Local_Frequency):
    request_service_list = Zipf_sample(0.8, 1, service_K, user_N)
    lamda_ik_matrix = np.zeros((user_N, service_K))  # x_il数组
    for i in range(len(request_service_list)):
        # 第i用户它的服务访问向量，只有它访问的那个服务类型对应的下标才为1
        lamda_ik_matrix[i, request_service_list[i] - 1] = 1
    # ?完成了请求的矩阵生成

    distances = np.random.uniform(80, 300, size=(user_N, server_M))
    g_N_M = np.power(distances, -2)

    input_data = np.random.uniform(
        2.4, 9.6, size=(user_N,)
    )  # ?单位Mbit  1e6 2.4, 11.2, 4.0, 12.0,

    # ?完成了距离矩阵生成

    # input_data = np.random.uniform(
    #     low=10, high=30, size=(user_N,))  # ?单位Mbit A T ask Offloading Method Based on UserSatisfaction in C-RAN With Mobile Edge Computing
    # print(input_data)
    computing_cycle = np.zeros((user_N))  # 单位Mcycles
    for i in range(user_N):
        computing_cycle[i] = (
            input_data[i] * Mu_K[request_service_list[i] - 1]
        )  # 每个用户访问了哪种类型
    max_tolerance = computing_cycle / Local_Frequency
    return lamda_ik_matrix, g_N_M, input_data, computing_cycle, max_tolerance


# service_K = 10
# server_M = 3
# user_N = 15
# slot_T = 3
# # 目标：生成state
# # 一个用户生成一个任务（传输量、计算量），属于一种类型，与M个服务器有一个距离向量
# Mu_K, Space_K = gen_service_attribute_list(service_K)
# frequency_M, storage_M = gen_server_attribute_list(server_M)
# Local_Frequency = gen_user_attribute_list(user_N)

# gen_state(user_N, server_M, service_K, Mu_K, Local_Frequency)
