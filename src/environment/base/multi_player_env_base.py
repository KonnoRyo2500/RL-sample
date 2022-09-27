# 強化学習勉強用サンプルプログラム 多人数ゲーム用環境ベースクラス

from abc import ABC, abstractmethod

from environment.base.env_base import EnvironmentBase


# 多人数ゲーム用環境ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class MultiPlayerEnvironmentBase(ABC, EnvironmentBase):
    # コンストラクタ
    def __init__(self, name):
        super().__init__(name)

        self.player = None  # 現手番のプレイヤー

    # 次の手番のプレイヤーに交代する
    @abstractmethod
    def switch_to_next_player(self):
        pass

    # 現手番のプレイヤーを取得する
    # プレイヤーは0始まりの数字で返される
    @abstractmethod
    def get_player(self):
        pass
