# 強化学習勉強用サンプルプログラム 行動選択用関数群

import random

# greedy法で行動を選択する
# 行動選択の基準として行動価値関数Q(s, a)を用いる
def greedy(action_space, q_func, state):
    # Q(s, a)が最大となるようなaは複数存在しうるので、そのような場合は
    # ランダムにaを選択することにする
    q_values = [q_func[(state, a)] for a in action_space]
    greedy_indices = [i for i, q in enumerate(q_values) if q == max(q_values)]
    greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]
    return action_space[greedy_idx]

# ε-greedy法で行動を選択する
def epsilon_greedy(action_space, q_func, state, epsilon):
    v = random.uniform(0, 1)
    if v <= epsilon:
        # ランダム選択
        random_idx = random.randint(0, len(action_space) - 1)
        return action_space[random_idx]
    else:
        # greedy法による選択
        return greedy(action_space, q_func, state)
