# 強化学習勉強用サンプルプログラム 定数定義

from enum import Enum, auto

### Grid World環境定義 ###

# 盤面のサイズ
GRID_WIDTH = 5
GRID_HEIGHT = 5

# 初期位置
# 盤面の幅がW, 高さがHの時、左上を(0, 0)、右下を(W - 1, H - 1)とする。
INITIAL_POS = (0, 0)

# ゴール位置
# 複数存在することを想定し、リストにする。
GOAL_POS = [(GRID_WIDTH - 1, GRID_HEIGHT - 1)]

# ゴール時の報酬
# GOAL_POSの要素数と同じ数の報酬を定義すること
GOAL_REWARD = [100]

# 移動方向
class Direction(Enum):
    Up = auto() # 上
    Down = auto() # 下
    Left = auto() # 左
    Right = auto() # 右

### エージェントのハイパーパラメータ ### 
# 学習時のエピソード数
N_EPISODE = 10000

# ε-greedy法でランダムに行動選択する確率
EPSILON = 0.10

# 割引率
GAMMA = 0.80

# 学習率
ALPHA = 0.50
