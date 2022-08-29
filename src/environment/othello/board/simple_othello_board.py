# 強化学習勉強用サンプルプログラム オセロ盤面クラス(シンプルな実装)

from enum import Enum, auto

from environment.othello.board.othello_board_base import OthelloBoardBase


# オセロ盤面クラス
# 特に特殊なアルゴリズムやデータ構造等は使わず、シンプルに実装する
class SimpleOthelloBoard(OthelloBoardBase):
    # 盤面上の石
    class Stone(Enum):
        White = auto()  # 白石
        Black = auto()  # 黒石
        Empty = auto()  # 石が置かれていない

    # コンストラクタ
    def __init__(self):
        super().__init__()

    # 石を置く
    def place_stone(self, pos):
        pass

    # 現在の手番で石を置けるマスを探し、座標のリストで取得する
    def search_available_grid(self):
        pass

    # 盤面の状態を8x8の2次元リストとして取得する
    def get_board(self):
        pass

    # 盤面を初期化する
    def reset(self):
        row = [self.Stone.Empty] * self.BOARD_WIDTH
        # [row.copy()] * self.BOARD_HEIGHTだと各行のlistのidがすべて同じになってしまうため、リスト内包表記にする
        initial_grid = [row.copy() for _ in range(self.BOARD_HEIGHT)]

        # 初期状態での石を配置
        initial_grid[3][3] = self.Stone.White
        initial_grid[3][4] = self.Stone.Black
        initial_grid[4][3] = self.Stone.Black
        initial_grid[4][4] = self.Stone.White

        self.grid = initial_grid
