# 強化学習勉強用サンプルプログラム 定数定義

from enum import Enum, auto

### Grid World環境定義 ###

# 盤面のサイズ
GRID_WIDTH = 5
GRID_HEIGHT = 5

# 初期位置
# 盤面の幅がW, 高さがHの時、左上を(0, 0)、右下を(W - 1, H - 1)とする。
INITIAL_POS = [0, 0]

# ゴール位置
# 複数存在することを想定し、リストにする。
GOAL_POS = [[GRID_WIDTH - 1, GRID_HEIGHT - 1]]

# 壁情報
# 定義方法はドキュメントを参照。
# 事前に手書きやペイントソフト、Excelなどで盤面を描いておくと良い。
WALL = [
    (False, False, True, False), # 1列目の垂直方向の壁情報
    (True, False, False, True, False), # 1列目の水平方向の壁情報

    (True, True, False, False),
    (False, False, True, False, True),

    (False, False, True, False),
    (False, True, True, True, False),

    (False, False, False, True),
    (False, False, True, False, True),

    (True, False, True, False),
]

# 移動方向
class Direction(Enum):
    Up = auto() # 上
    Down = auto() # 下
    Left = auto() # 左
    Right = auto() # 右
