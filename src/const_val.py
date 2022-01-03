# 強化学習勉強用サンプルプログラム 定数定義

from enum import Enum, auto

# 移動方向
class Direction(Enum):
    Up = auto() # 上
    Down = auto() # 下
    Left = auto() # 左
    Right = auto() # 右
