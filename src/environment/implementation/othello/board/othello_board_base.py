# 強化学習勉強用サンプルプログラム オセロ盤面ベースクラス

from abc import ABCMeta, abstractmethod


# オセロ盤面ベースクラス
class OthelloBoardBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self):
        # 盤面の実体
        self.grid = None

        # 盤面の初期化(self.gridはここで初期化される)
        self.reset()

    # 指定された色と場所に基づき、石を置く
    @abstractmethod
    def put_disk(self, pos, player):
        pass

    # 指定された色の石を置けるマスを探し、座標のリストで取得する
    @abstractmethod
    def search_available_grid(self, player):
        pass

    # 盤面の状態を8x8の2次元リストとして取得する
    # 各要素にはDiskクラスのメンバ値(整数値)が入る
    @abstractmethod
    def get_grid(self):
        pass

    # 盤面を初期化する
    @abstractmethod
    def reset(self):
        pass
