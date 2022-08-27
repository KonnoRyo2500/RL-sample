# 強化学習勉強用サンプルプログラム オセロ盤面クラス(シンプルな実装)

from environment.othello.board.othello_board_base import OthelloBoardBase


# オセロ盤面クラス
# 特に特殊なアルゴリズムやデータ構造等は使わず、シンプルに実装する
class SimpleOthelloBoard(OthelloBoardBase):
    # コンストラクタ
    def __init__(self):
        super().__init__()

    # 石を置く
    def place_stone(self, pos):
        pass

    # 現在の手番で石を置けるマスを探し、座標のリストで取得する
    def find_available_board(self):
        pass

    # 盤面の状態を8x8の2次元リストとして取得する
    def get_board(self):
        pass

    # 盤面を初期化する
    def reset(self):
        pass
