# 強化学習勉強用サンプルプログラム オセロ盤面クラス(シンプルな実装)

from itertools import product

from environment.implementation.othello.board.othello_board_base import OthelloBoardBase
from environment.implementation.othello.const_val import *


# オセロ盤面クラス
# 特に特殊なアルゴリズムやデータ構造等は使わず、シンプルに実装する
class SimpleOthelloBoard(OthelloBoardBase):
    # コンストラクタ
    def __init__(self):
        super().__init__()

    # 指定された色と場所に基づき、石を置く
    def put_disk(self, pos, player):
        disk = self._conv_player_to_disk(player)
        x, y = pos

        if self.grid[y][x] != Disk.Empty.value:
            # すでに石が置かれていた
            return
        self.grid[y][x] = disk

        # 石をひっくり返す
        self._flip_disk(pos, disk)

    # 指定された色の石を置けるマスを探し、座標のリストで取得する
    def search_available_grid(self, player):
        # 探索方向
        # 順に、上、右上、右、右下、下、左下、左、左上の時の(x, y)の増加量
        search_directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

        # 各探索方向について、ひっくり返せる石があればその場所には石が置ける
        available_squares = []
        disk = self._conv_player_to_disk(player)
        for x, y in product(range(GRID_WIDTH), range(GRID_HEIGHT)):
            pos = (x, y)
            if self.grid[y][x] != Disk.Empty.value:
                continue

            for direction in search_directions:
                flippable_disks = self._find_flippable_disks(pos, disk, direction)

                if len(flippable_disks) >= 1:
                    available_squares.append(pos)
                    break

        return available_squares

    # 盤面の状態を8x8の2次元リストとして取得する
    # 各要素にはDiskクラスのメンバ値(整数値)が入る
    def get_grid(self):
        return self.grid

    # 盤面を初期化する
    def reset(self):
        row = [Disk.Empty.value] * GRID_WIDTH
        # [row.copy()] * self.BOARD_HEIGHTだと各行のlistのidがすべて同じになってしまうため、リスト内包表記にする
        initial_grid = [row.copy() for _ in range(GRID_HEIGHT)]

        # 初期状態での石を配置
        initial_grid[3][3] = Disk.Light.value
        initial_grid[3][4] = Disk.Dark.value
        initial_grid[4][3] = Disk.Dark.value
        initial_grid[4][4] = Disk.Light.value

        self.grid = initial_grid

    # 指定された石の周りにある石をひっくり返す
    def _flip_disk(self, pos, disk):
        # 探索方向
        # 順に、上、右上、右、右下、下、左下、左、左上の時の(x, y)の増加量
        search_directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

        # 各探索方向について、ひっくり返せる石をすべてひっくり返す
        for direction in search_directions:
            flippable_disks = self._find_flippable_disks(pos, disk, direction)
            for x, y in flippable_disks:
                self.grid[y][x] = disk

    # 指定された方向にあるひっくり返せる石の座標を返す
    def _find_flippable_disks(self, start_pos, my_disk, direction):
        opponent_disk = Disk.Light.value if my_disk == Disk.Dark.value else Disk.Dark.value
        searching_pos = start_pos
        flippable_disk = []

        # 探索方向に相手の石が並んでおり、自分の石に到達したらその間にある相手の石はすべてひっくり返せる
        while self._is_inside(searching_pos):
            search_x, search_y = searching_pos
            dir_x, dir_y = direction
            searching_disk = self.grid[search_y][search_x]

            if (searching_disk == Disk.Empty.value) and (searching_pos != start_pos):
                return []

            if (searching_disk == my_disk) and (searching_pos != start_pos):
                return flippable_disk

            if searching_disk == opponent_disk:
                flippable_disk.append((search_x, search_y))

            searching_pos = (search_x + dir_x, search_y + dir_y)

        return []

    # 指定した座標が盤面の中にあるかどうか判定する
    def _is_inside(self, pos):
        x, y = pos
        return (0 <= x < GRID_WIDTH) and (0 <= y < GRID_HEIGHT)

    # 環境クラスからくるプレイヤー情報を、対応する石の色に変換する
    def _conv_player_to_disk(self, player):
        return {
            Player.Dark: Disk.Dark.value,
            Player.Light: Disk.Light.value,
        }[player]
