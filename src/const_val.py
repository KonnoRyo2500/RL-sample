# 強化学習勉強用サンプルプログラム 定数定義

from enum import Enum, auto

# 盤面のサイズ
GRID_WIDTH = 5
GRID_HEIGHT = 5

# 移動方向
class Direction(Enum):
    Up = auto() # 上
    Down = auto() # 下
    Left = auto() # 左
    Right = auto() # 右
