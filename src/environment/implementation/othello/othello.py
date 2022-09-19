# 強化学習勉強用サンプルプログラム オセロ環境

from enum import Enum, auto

from environment.base.env_base import EnvironmentBase
from environment.implementation.othello.board.simple_othello_board import SimpleOthelloBoard
from common.const_val import Environment


# オセロ環境
# 盤面は、一般的な8x8の盤面とする
class Othello(EnvironmentBase):
    # プレイヤーの手番
    class Player(Enum):
        White = auto()  # 白の手番
        Black = auto()  # 黒の手番

    # コンストラクタ
    def __init__(self):
        super().__init__(Environment.Othello.value)

        self.board = self._create_board(self.config['board_implementation'])

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

    # 盤面を作成する
    def _create_board(self, implementation):
        board_class = {
            'simple': SimpleOthelloBoard
        }[implementation]
        board_instance = board_class()

        return board_instance
