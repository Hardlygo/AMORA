import numpy as np
import torch
import gymnasium
import argparse
import os
import datetime
import utils
import TD3
import OurDDPG
import DDPG
from environment import Environment
from utilities import save_to_writer
from tensorboardX import SummaryWriter

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gymnasium.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="MOOC-v2")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=55, type=int)
    # Time steps initial random policy is used

    parser.add_argument("--start_timesteps", default=25000,
                        type=int)  # int(0.05 * 500001)
    # How often (time steps) we evaluate
    # parser.add_argument("--eval_freq", default=20e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=500001, type=int)  # 1000001
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.26, type=float)  # 0.25
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.90,
                        type=float)  # Discount factor
    # Target network update rate
    parser.add_argument("--tau", default=0.005, type=float)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.26)  # 0.25,0.3
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=4, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")
    parser.add_argument("--a_lr", default=1e-4, type=float)  # 3e-4 3e-3 3e-5
    parser.add_argument("--c_lr", default=3e-4, type=float)
    parser.add_argument("--hidden_dims", default=512, type=int)

    # Environment perparameters
    parser.add_argument("--user_N", default=50, type=float)  # 10 20 40
    parser.add_argument("--server_M", default=3, type=float)  # 3 1 2 4 5
    # 8  16 20 24  6 12 18 24 30
    parser.add_argument("--service_K", default=12, type=float)
    parser.add_argument("--channels", default=16, type=float)
    parser.add_argument("--bandwidth", default=20, type=float)
    parser.add_argument("--slot_T", default=3, type=float)

    # C:\Windows\System32\cmd.exe
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    np.random.seed(args.seed)
    utils.seed_torch(args.seed)
    # 设置随机种子
    env = Environment(
        user_N=args.user_N,
        server_M=args.server_M,
        service_K=args.service_K,
        T=args.slot_T,
        channels=args.channels,
        bandwidth=args.bandwidth,
    )
    env.action_space.seed(args.seed)
    state_dim = env.T * env.T_input_dim

    action_dim = (
        env.action_space["offloading"].shape[0] +
        env.action_space["caching"].shape[0]
    )
    writer = SummaryWriter(
        "runs/{}_MOOC_{}_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env,
            "TD3",
        )
    )

    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # max_action = float(env.action_space.high[0])
    max_action = 1.0  # ? 动作的最大值

    args.state_dim = state_dim
    args.action_dim = action_dim
    args.max_action = max_action

    args.K = args.service_K
    args.N = args.user_N
    args.M = args.server_M
    args.T = args.slot_T
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        args.policy_noise = args.policy_noise * max_action
        args.noise_clip = args.noise_clip * max_action
        policy = TD3.TD3(**vars(args))
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**vars(args))
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**vars(args))

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(
        state_dim, action_dim, max_size=args.max_timesteps
    )

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]

    observation, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    t_cost = 0
    sum_usage_R = 0
    sum_hite_rate = 0
    sum_usage_CPU = 0
    start_use_agent = False  # 是否开始使用agent生成动作
    start_time = datetime.datetime.now().replace(microsecond=0)

    for t in range(int(args.max_timesteps)):

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample()
            start_use_agent = False
            # action = utils.normal_action(action_, env)
        else:
            start_use_agent = True
            action = (
                policy.select_action(np.array(observation))
                + np.random.normal(0, max_action *
                                   args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            # action = utils.rescale_action(action, env)

        # # Perform action
        # new_observation, reward, done, t_cost_per_step, usage_R, hit_rate,  usage_CPU = env.step(
        #     observation, action, info={'start_use_agent': start_use_agent})
        new_observation, reward, done, t_cost_per_step, usage_R, hit_rate, usage_CPU = (
            env.step(observation, action, start_use_agent)
        )
        episode_timesteps += 1

        done_bool = 1.0 if done else 0.0

        # Store data in replay buffer
        replay_buffer.add(observation, action,
                          new_observation, reward, done_bool)
        sum_usage_R += usage_R
        sum_hite_rate += hit_rate
        sum_usage_CPU += usage_CPU

        t_cost += t_cost_per_step

        observation = new_observation
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            tensorboard_logs = {
                "train/reward": episode_reward,
                "train/time": t_cost,
                "train/avg_time_per_task": t_cost / (env._max_episode_steps),
                "train/avg_reward": episode_reward / (env._max_episode_steps),
            }
            save_to_writer(writer, tensorboard_logs, episode_num + 1)

            #!计算平均利用率
            avg_usage_logs = {
                "avg_usage/storage": sum_usage_R / env._max_episode_steps,
                "avg_usage/hite-rate": sum_hite_rate / env._max_episode_steps,
                "avg_usage/CPU": sum_usage_CPU / env._max_episode_steps,
            }
            save_to_writer(writer, avg_usage_logs, episode_num + 1)

            # Reset environment
            observation, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            t_cost = 0
            sum_usage_R = 0
            sum_hite_rate = 0
            sum_usage_CPU = 0
    writer.close()
    print(
        "============================================================================================"
    )
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )
