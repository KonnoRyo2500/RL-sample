# 強化学習勉強用サンプルプログラム Grid World環境クラス

from copy import copy

from const_val import *

# Grid World環境クラス
class GridWorld:
    # コンストラクタ
    def __init__(self):
        # 盤面の作成
        self.wall, self.init_pos, self.goal_pos = self._create_grid()
        self.pos = copy(self.init_pos)

    # 移動する
    def move(self, direction):
        pos_diff = {
            Direction.Up: [0, -1],
            Direction.Down: [0, 1],
            Direction.Left: [-1, 0],
            Direction.Right: [1, 0],
        }[direction]

        next_pos = [i + d for i, d in zip(self.pos, pos_diff)]

        if self._can_move(direction):
            self.pos = copy(next_pos)

    # 盤面のリセット
    def reset(self):
        self.pos = copy(self.init_pos)

    # 移動できる方向の検索
    def get_available_direction(self):
        available_direction = []
        for name in Direction._member_names_:
            if self._can_move(Direction[name]):
                available_direction.append(Direction[name])

        return available_direction

    # 指定した方向に移動できるかどうか判定する
    def _can_move(self, direction):
        pos_diff = {
            Direction.Up: [0, -1],
            Direction.Down: [0, 1],
            Direction.Left: [-1, 0],
            Direction.Right: [1, 0],
        }[direction]

        next_pos = [i + d for i, d in zip(self.pos, pos_diff)]

        # すでにゴールにいる場合は移動しない
        if self.pos in self.goal_pos:
            return False

        # 範囲内の判定
        x_is_in_grid = (0 <= next_pos[0] < GRID_WIDTH)
        y_is_in_grid = (0 <= next_pos[1] < GRID_HEIGHT)
        if (not x_is_in_grid) or (not y_is_in_grid):
            return False

        # 壁の判定
        hit_vertical_wall = ((direction == Direction.Left) and (self.wall[2 * next_pos[1]][next_pos[0]])) or \
                            ((direction == Direction.Right) and (self.wall[2 * next_pos[1]][next_pos[0] - 1]))
        hit_horizontal_wall = ((direction == Direction.Up) and (self.wall[2 * next_pos[1] + 1][next_pos[0]])) or \
                              ((direction == Direction.Down) and (self.wall[2 * next_pos[1] - 1][next_pos[0]]))
        if hit_vertical_wall or hit_horizontal_wall:
            return False

        return True

    # 盤面(グリッド)を作成
    def _create_grid(self):
        # 盤面の各マスの周りにある壁情報を定義する。
        # 定義方法はドキュメントを参照。
        # 事前に手書きやペイントソフト、Excelなどで盤面を描いておくと良い。
        wall = [
            (False, False, True, False), # 1列目の垂直方向の壁情報
            (True, False, False, True, False), # 1列目の水平方向の壁情報

            (True, True, False, False),
            (False, False, True, False, True),

            (False, False, True, False),
            (False, True, True, True, False),

            (False, False, False, True),
            (False, False, True, False, True),

            (True, False, True, False),
        ]
        # 初期位置の定義
        # 盤面の幅がW, 高さがHの時、左上を(0, 0)、右下を(W - 1, H - 1)とする。
        initial_pos = [0, 0]
        # ゴール位置の定義
        # 複数存在することを想定し、リストにする。
        goal_pos = [[GRID_WIDTH - 1, GRID_HEIGHT - 1]]

        return wall, initial_pos, goal_pos

