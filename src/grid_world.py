# 強化学習勉強用サンプルプログラム Grid World環境クラス

from const_val import *

# Grid World環境クラス
class GridWorld:
    # コンストラクタ
    def __init__(self):
        # 盤面の作成
        self.wall, self.cell_idx, self.goal_idx = self._create_grid()

    # 移動する
    def move(self, direction):
        # すでにゴールにいる場合は移動しない
        if self.cell_idx in self.goal_idx:
            print(f'すでにゴール座標 {self.cell_idx} にいます。')
            return self.cell_idx

        idx_diff = {
            Direction.Up: [0, -1],
            Direction.Down: [0, 1],
            Direction.Left: [-1, 0],
            Direction.Right: [1, 0],
        }[direction]

        next_idx = [i + d for i, d in zip(self.cell_idx, idx_diff)]

        # 範囲内の判定
        x_is_in_grid = (0 <= next_idx[0] < GRID_WIDTH)
        y_is_in_grid = (0 <= next_idx[1] < GRID_HEIGHT)
        if (not x_is_in_grid) or (not y_is_in_grid):
            print(f'グリッド外の座標 {next_idx} に移動しようとしています。')
            return self.cell_idx

        # 壁の判定
        hit_vertical_wall = ((direction == Direction.Left) and (self.wall[2 * next_idx[1]][next_idx[0]])) or \
                            ((direction == Direction.Right) and (self.wall[2 * next_idx[1]][next_idx[0] - 1]))
        hit_horizontal_wall = ((direction == Direction.Up) and (self.wall[2 * next_idx[1] + 1][next_idx[0]])) or \
                              ((direction == Direction.Down) and (self.wall[2 * next_idx[1] - 1][next_idx[0]]))
        if hit_vertical_wall or hit_horizontal_wall:
            print(f'座標 {self.cell_idx} から {next_idx} に移動しようしたとき、壁に衝突しました。')
            return self.cell_idx

        print(f'座標 {self.cell_idx} から {next_idx} に移動しました。')
        self.cell_idx = next_idx

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
        initial_idx = [0, 0]
        # ゴール位置の定義
        # 複数存在することを想定し、リストにする。
        goal_idx = [[GRID_WIDTH - 1, GRID_HEIGHT - 1]]

        return wall, initial_idx, goal_idx

