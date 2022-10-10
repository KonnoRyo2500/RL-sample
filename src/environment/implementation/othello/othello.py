# 強化学習勉強用サンプルプログラム オセロ環境

from itertools import product

from environment.base.multi_player_env_base import MultiPlayerEnvironmentBase
from environment.implementation.othello.board.simple_othello_board import SimpleOthelloBoard
from environment.implementation.othello.const_val import *
from common.const_val import Environment


# オセロ環境
# 盤面は、一般的な8x8の盤面とする
class Othello(MultiPlayerEnvironmentBase):

    # コンストラクタ
    def __init__(self):
        super().__init__(Environment.Othello.value)

        self.board = self._create_board(self.config.board_implementation)

        # 初手プレイヤーの設定
        self.player = Player.Light

    # 環境の行動空間を取得
    def get_action_space(self):
        return [(x, y) for x, y in product(range(GRID_WIDTH), range(GRID_HEIGHT))]

    # 現在選択可能な行動を取得
    def get_available_actions(self):
        return self.board.search_available_grid(self.player)

    # 指定された行動を実行し、報酬を得る
    def exec_action(self, action):
        self.board.put_disk(action, self.player)
        return 0

    # 環境の状態空間を取得
    # 状態が多すぎて(もしくは無限に存在して)取得できない場合はNoneを返す
    def get_state_space(self):
        # 8x8オセロは10^60個ほどの状態が存在する
        # それらをすべてメモリ上に保持することは非現実的なためNoneを返す
        return None

    # 現在の状態を取得
    def get_state(self):
        return self.board.get_grid()

    # 現在の状態が終端状態かどうかを返す
    def is_terminal_state(self):
        # 双方のプレイヤーの打てる手がなければ終了
        available_grid_dark = self.board.search_available_grid(Player.Dark)
        available_grid_light = self.board.search_available_grid(Player.Light)

        return (len(available_grid_dark) == 0) and (len(available_grid_light) == 0)

    # 環境をリセットする
    def reset(self):
        self.board.reset()

    # 次の手番のプレイヤーに交代する
    def switch_to_next_player(self):
        next_player = Player.Dark if self.player == Player.Light else Player.Light
        self.player = next_player

    # 現手番のプレイヤーを取得する
    # プレイヤーは0始まりの数字で返される
    def get_player(self):
        return {
            Player.Light: 0,
            Player.Dark: 1,
        }[self.player]

    # 盤面を作成する
    def _create_board(self, implementation):
        board_class = {
            'simple': SimpleOthelloBoard
        }[implementation]
        board_instance = board_class()

        return board_instance
