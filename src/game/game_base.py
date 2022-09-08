# 強化学習勉強用サンプルプログラム ゲームプレイ環境ベースクラス

from abc import ABCMeta, abstractmethod


# ゲームプレイ環境ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class GameBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, env, agents, env_config, agent_configs):
        self.env = env
        self.env_config = env_config
        self.agents = agents
        self.agent_configs = agent_configs

    # エージェントを学習させる
    @abstractmethod
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    @abstractmethod
    def play(self):
        pass
