# 強化学習勉強用サンプルプログラム オセロ環境

from environment.env_base import EnvironmentBase
from enum import Enum, auto


# オセロ環境
# 盤面は、一般的な8x8の盤面とする
class Othello(EnvironmentBase):
    # プレイヤーの手番
    class Player(Enum):
        White = auto()
        Black = auto()

    # コンストラクタ
    def __init__(self, config):
        super().__init__(config)

    # 環境の行動空間を取得
    def get_action_space(self):
        pass

    # 現在選択可能な行動を取得
    def get_available_actions(self):
        pass

    # 指定された行動を実行し、報酬を得る
    def exec_action(self, action):
        pass

    # 環境の状態空間を取得
    # 状態が多すぎて(もしくは無限に存在して)取得できない場合はNoneを返す
    def get_state_space(self):
        # 8x8オセロは10^60個ほどの状態が存在する
        # それらをすべてメモリ上に保持することは非現実的なためNoneを返す
        return None

    # 現在の状態を取得
    def get_state(self):
        return self.state

    # 現在の状態が終端状態かどうかを返す
    def is_terminal_state(self):
        pass

    # 環境をリセットする
    def reset(self):
        pass
