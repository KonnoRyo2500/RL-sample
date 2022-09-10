# 強化学習勉強用サンプルプログラム ゲームフレームワークベースクラス

from abc import ABCMeta, abstractmethod


# ゲームフレームワークベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class GameBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents

    # エージェントを学習させる
    @abstractmethod
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    @abstractmethod
    def play(self):
        pass
