# 強化学習勉強用サンプルプログラム Q学習エージェントクラス

from itertools import product

from common.const_val import *
from agent.agent_base import AgentBase

# Q学習エージェントクラス
class QAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env):
        super().__init__(env)
        self.q_func = self._make_initial_q_function()

    # 学習した価値関数を基にエピソードをプレイ
    def play(self):
        pass

    # 学習の実行
    def train(self):
        pass

    # エピソードの実行
    def episode(self):
        pass

    # 1ステップ実行
    def step(self):
        pass

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self):
        init_q_func = {}
        for x, y in product(range(GRID_WIDTH), range(GRID_HEIGHT)):
            for name in Direction._member_names_:
                state = (x, y)
                action = Direction[name]
                init_q_func[(state, action)] = 0

        return init_q_func

