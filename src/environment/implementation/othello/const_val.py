# 強化学習勉強用サンプルプログラム オセロ用定数群

from enum import Enum, auto

# 盤面関連
# 盤面サイズ
GRID_WIDTH = 8  # 盤面の幅
GRID_HEIGHT = 8  # 盤面の高さ


# プレイヤーの手番
class Player(Enum):
    Light = auto()  # 白の手番
    Dark = auto()  # 黒の手番


# 石の定義
class Disk(Enum):
    Empty = auto()  # 石が置かれていない
    Dark = auto()  # 黒石
    Light = auto()  # 白石
