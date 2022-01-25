# 強化学習勉強用サンプルプログラム エージェントベースクラス

from abc import ABCMeta, abstractmethod

# エージェントベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class AgentBase(metaclass=ABCMeta):
    # 具体的な強化学習アルゴリズムはサブクラスで実装すること。

    # コンストラクタ
    def __init__(self, env, config):
        self.env = env
        self.config = config

    # 学習済みエージェントにエピソードをプレイさせる
    @abstractmethod
    def play(self):
        pass

    # エージェントを学習させる
    @abstractmethod
    def train(self):
        pass
