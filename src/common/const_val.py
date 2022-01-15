# 強化学習勉強用サンプルプログラム 定数定義

from enum import Enum, auto

# 注意:
#   1) ここには、ユーザによらない定数(IDなど)を記述すること。
#      ユーザが設定する値は、設定ファイルに記載する。
#   2) 複数ファイルで共通して利用する定数のみ定義すること。
#      1つのファイルでしか使わない定数は、各ファイルに定義する。

# 移動方向
class Direction(Enum):
    Up = auto() # 上
    Down = auto() # 下
    Left = auto() # 左
    Right = auto() # 右
