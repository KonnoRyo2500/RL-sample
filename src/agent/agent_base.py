# 強化学習勉強用サンプルプログラム エージェントベースクラス

from abc import ABCMeta, abstractmethod
from enum import Enum, auto

from common.config import load_config
from common.dirs import AGENT_CONFIG_DIR


# エージェントベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class AgentBase(metaclass=ABCMeta):
    # 具体的な強化学習アルゴリズムはサブクラスで実装すること。

    # 動作モード
    class OperationMode(Enum):
        Train = auto()
        Play = auto()

    # コンストラクタ
    def __init__(self, env, name):
        self.config = self._load_config(name)
        self.mode = AgentBase.OperationMode.Train

    # 環境の情報を参照し、次の行動を決定する
    @abstractmethod
    def decide_action(self, env):
        pass

    # 環境からの情報を自身にフィードバックする
    @abstractmethod
    def feedback(self, reward, env):
        pass

    # エージェントを学習モードにする
    def switch_to_train_mode(self):
        self.mode = AgentBase.OperationMode.Train

    # エージェントをプレイモードにする
    def switch_to_play_mode(self):
        self.mode = AgentBase.OperationMode.Play

    # 設定ファイルを読み込む
    def _load_config(self, name):
        dir_name = AGENT_CONFIG_DIR(name)
        file_name = f'{name}_config.yaml'

        return load_config(dir_name, file_name)
