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

    # 現手番のプレイヤーを変更する
    # プレイヤーは0始まりの数字で渡され、本関数にて環境固有の表現に変換されてメンバ変数にセットされる
    @abstractmethod
    def change_player(self, player):
        pass
