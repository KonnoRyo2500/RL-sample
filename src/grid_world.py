# 強化学習勉強用サンプルプログラム Grid World環境クラス

# Grid World環境クラス
class GridWorld:
    # コンストラクタ
    def __init__(self):
        # 盤面の作成
        self.wall, self.cell_idx, self.goal_idx = self._create_grid()

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
        initial_idx = (0, 0)
        # ゴール位置の定義
        # 複数存在することを想定し、リストにする。
        goal_idx = [(4, 4)]

        return wall, initial_idx, goal_idx

