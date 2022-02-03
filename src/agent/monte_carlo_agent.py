# 強化学習勉強用サンプルプログラム モンテカルロ法エージェントクラス

import random
from pprint import pprint

from agent.agent_base import AgentBase

# モンテカルロ法エージェントクラス
class MonteCarloAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)
        self.v_func = self._make_initial_v_function()

    # 学習した価値関数を基にエピソードをプレイ
    def play(self):
        pass

    # 学習の実行
    def train(self):
        # アルゴリズム詳細
        # 1) 各状態sについて、状態価値関数V(s)を0で初期化
        # 2) 各状態sを開始状態としてエピソードを実施し、収益
        #       R = r0 + γr1 + γ^(2)r2 + ... + γ^(T-1)rT
        #    を得る。
        # 3) 2)を一定関数繰り返し、sにおける平均収益avg(R)を計算する。
        # 4) V(s) = avg(R)として完了。
        # Note: モンテカルロ法には複数の亜種が存在するが、
        #       ここでは一番シンプルな方法を選択した。
        # TODO: 終端状態の状態価値関数の値が常に0になってしまうので、
        #       別の値を入れておきたい。そうしないと、プレイ時にgreedy方策をとった場合
        #       終端状態にたどりつけず無限ループになる可能性がある。

        for s in self.env.get_all_states():
            # 一定回数プレイアウトを繰り返し、平均収益を求める
            profit_sum = 0
            for i in range(self.config['num_playout']):
                profit_sum += self._playout(s)

            avg_profit = profit_sum / self.config['num_playout']

            # 平均収益をその状態sの状態価値関数とする
            self.v_func[s] = avg_profit

        pprint(self.v_func)

    # プレイアウトの実行
    def _playout(self, init_state):
        num_step = 0
        profit = 0 # プレイアウトの収益
        contribution_rate = 1.0 # 寄与率。収益の計算に使う

        # 初期状態を設定
        self.env.set_state(init_state)

        while not self.env.is_terminal_state():
            # 行動aを選択する
            action = self._select_action_randomly()

            # 行動aを行い、報酬rを得る
            reward = self.env.exec_action(action)
            profit += (reward * contribution_rate)
            num_step += 1
            contribution_rate *= self.config['gamma']

            # 無限ループ防止のため、一定の回数行動しても終端にたどり着かなければ打ち止め
            if num_step > self.config['step_limit']:
                break

        # 環境をリセット
        self.env.reset()

        return profit

    # ランダムに行動を選択する
    def _select_action_randomly(self):
        actions = self.env.get_actions()
        idx = random.randint(0, len(actions) - 1)

        return actions[idx]

    # 状態価値関数を初期化して返す
    def _make_initial_v_function(self):
        init_v_func = {}
        for state in self.env.get_all_states():
            init_v_func[state] = 0

        return init_v_func
