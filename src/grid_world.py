# 強化学習勉強用サンプルプログラム Grid World環境クラス

from copy import copy

from const_val import *

# Grid World環境クラス
class GridWorld:
    # コンストラクタ
    def __init__(self):
        self.pos = copy(INITIAL_POS)

    # 移動+報酬の取得
    def move(self, direction):
        next_pos = self._get_next_pos(direction)

        # 報酬の確定
        reward = None
        if not self._can_move(direction):
            # 移動できない場合は想定しない
            # エージェントは事前に移動可能な方向をget_available_directionで取得する想定
            return reward
        elif next_pos in GOAL_POS:
            # ゴール時
            i = GOAL_POS.index(next_pos)
            reward = GOAL_REWARD[i]
            self.reset()
            return reward
        else:
            # 通常移動時
            reward = 0

        # 移動の実施
        self.pos = copy(next_pos)

        return reward

    # 盤面のリセット
    def reset(self):
        self.pos = copy(INITIAL_POS)

    # 移動できる方向の検索
    def get_available_direction(self):
        available_direction = []
        for name in Direction._member_names_:
            if self._can_move(Direction[name]):
                available_direction.append(Direction[name])

        return available_direction

    # 与えられた方向から、移動先のマスを取得する
    def _get_next_pos(self, direction):
        pos_diff = {
            Direction.Up: [0, -1],
            Direction.Down: [0, 1],
            Direction.Left: [-1, 0],
            Direction.Right: [1, 0],
        }[direction]

        next_pos = [i + d for i, d in zip(self.pos, pos_diff)]

        return next_pos

    # 指定した方向に移動できるかどうか判定する
    def _can_move(self, direction):
        next_pos = self._get_next_pos(direction)

        # すでにゴールにいる場合は移動しない
        if self.pos in GOAL_POS:
            return False

        # 範囲内の判定
        x_is_in_grid = (0 <= next_pos[0] < GRID_WIDTH)
        y_is_in_grid = (0 <= next_pos[1] < GRID_HEIGHT)
        if (not x_is_in_grid) or (not y_is_in_grid):
            return False

        # 壁の判定
        hit_vertical_wall = ((direction == Direction.Left) and (WALL[2 * next_pos[1]][next_pos[0]])) or \
                            ((direction == Direction.Right) and (WALL[2 * next_pos[1]][next_pos[0] - 1]))
        hit_horizontal_wall = ((direction == Direction.Up) and (WALL[2 * next_pos[1] + 1][next_pos[0]])) or \
                              ((direction == Direction.Down) and (WALL[2 * next_pos[1] - 1][next_pos[0]]))
        if hit_vertical_wall or hit_horizontal_wall:
            return False

        return True
