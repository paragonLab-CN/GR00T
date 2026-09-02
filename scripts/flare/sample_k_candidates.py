import random
import numpy as np
import torch


def sample_k_candidate_actions(policy, observation, k=4, base_seed=2026):
    """
    对同一个 observation 调用 GR00T k 次，生成 k 条候选动作轨迹。

    返回:
        action_key: 动作字段名
        candidate_actions: [B, K, T, D]
    """

    candidate_list = []

    for i in range(k):
        seed = base_seed + i

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        action, info = policy.get_action(observation)

        # GR00T 的 action 是 dict，先自动取第一个 action key
        action_key = list(action.keys())[0]

        # action[action_key] shape 一般是 [B, T, D]
        action_i = action[action_key]

        candidate_list.append(action_i)

        print(f"[candidate {i}] action_key = {action_key}, shape = {action_i.shape}")

    # [K, B, T, D] -> [B, K, T, D]
    candidate_actions = np.stack(candidate_list, axis=0)
    candidate_actions = np.transpose(candidate_actions, (1, 0, 2, 3))

    print("final candidate_actions shape:", candidate_actions.shape)

    return action_key, candidate_actions