# 強化学習勉強用サンプルプログラム オセロ盤面ベースクラス

from abc import ABCMeta, abstractmethod


# オセロ盤面ベースクラス
class OthelloBoardBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self):
        # 盤面の実体
        self.grid = None

        # 定数
        self.BOARD_WIDTH = 8  # 盤面の幅
        self.BOARD_HEIGHT = 8  # 盤面の高さ
        # 石については実装方法によりデータの持ち方が異なるため、定義はサブクラス側で行う

        # 盤面の初期化
        self.reset()

    # 石を置く
    @abstractmethod
    def place_stone(self, pos):
        pass

    # 現在の手番で石を置けるマスを探し、座標のリストで取得する
    @abstractmethod
    def find_available_board(self):
        pass

    # 盤面の状態を8x8の2次元リストとして取得する
    @abstractmethod
    def get_board(self):
        pass

    # 盤面を初期化する
    @abstractmethod
    def reset(self):
        pass
