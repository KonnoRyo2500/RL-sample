# 強化学習勉強用サンプルプログラム ゲームフレームワークベースクラス

from abc import ABCMeta, abstractmethod

from common.config import load_config
from common.dirs import GAME_CONFIG_DIR


# ゲームフレームワークベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class GameBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, env, agents, name):
        self.env = env
        self.agents = agents
        self.config = self._load_config(name)

    # エージェントを学習させる
    @abstractmethod
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    @abstractmethod
    def play(self):
        pass

    # 設定ファイルを読み込む
    def _load_config(self, name):
        dir_name = GAME_CONFIG_DIR(name)
        file_name = f'{name}_config.yaml'

        return load_config(dir_name, file_name)
