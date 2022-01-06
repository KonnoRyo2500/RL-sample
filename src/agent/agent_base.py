# 強化学習勉強用サンプルプログラム エージェントベースクラス

from abc import ABCMeta, abstractmethod

# エージェントベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
# 具体的な強化学習アルゴリズムはサブクラスで実装すること。
class AgentBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, env):
        self.env = env

    # 学習した価値関数を基にエピソードをプレイ
    @abstractmethod
    def play(self):
        pass

    # 学習の実行
    @abstractmethod
    def train(self):
        pass

    # エピソードの実行
    @abstractmethod
    def episode(self):
        pass

    # 1ステップ実行
    @abstractmethod
    def step(self):
        pass