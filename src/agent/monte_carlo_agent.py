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
        step_count = 0
        step_limit = self.config['play_step_limit']
        while not self.env.is_terminal_state():
            # 状態を取得(表示用)
            state = self.env.get_state()

            # greedy法で行動を選択
            action = self._select_action_with_greedy()
            print(f'状態 {state} で行動 {action} を選択しました。')

            # 行動する
            reward = self.env.exec_action(action)

            # 無限ループ防止のため、一定回数移動してもゴールしなかったら
            # エピソードを途中で打ち切る
            step_count += 1
            if step_count > step_limit:
                print(f'行動数が上限({step_limit}回)を超えたため、エピソードを終了します。')
                break

        if reward != 0:
            print(f'エピソードをプレイし、報酬 {reward} が得られました。')
        self.env.reset()

    # 学習の実行
    def train(self):
        # アルゴリズム詳細
        # 1) 各状態sについて、状態価値関数V(s)を0で初期化
        # 2) 各状態sを開始状態としてエピソードを実施し、収益
        #       R = r0 + γr1 + γ^(2)r2 + ... + γ^(T-1)rT
        #    を得る。
        # 3) 2)を一定回数繰り返し、sにおける平均収益avg(R)を計算する。
        # ※ただし、終端状態での状態価値関数の値は、その終端状態での報酬値とする。
        # 4) V(s) = avg(R)として完了。
        # Note: モンテカルロ法には複数の亜種が存在するが、
        #       ここでは一番シンプルな方法を選択した。

        for s in self.env.get_all_states():
            # 一定回数プレイアウトを繰り返し、平均収益を求める
            profit_sum = 0
            for i in range(self.config['num_playout']):
                profit = self._playout(s)
                if profit is None:
                    profit_sum = None
                    break
                else:
                    profit_sum += profit

            if profit_sum is None:
                continue

            avg_profit = profit_sum / self.config['num_playout']

            # 平均収益をその状態sの状態価値関数とする
            self.v_func[s] = avg_profit

        pprint(self.v_func)

    # プレイアウトの実行
    def _playout(self, init_state):
        # 初期状態を設定
        self.env.set_state(init_state)

        # 初期状態が終端状態であれば、プレイアウトは実施しない
        if self.env.is_terminal_state():
            self.env.reset()
            return None

        step_count = 0
        profit = 0 # プレイアウトの収益
        contribution_rate = 1.0 # 寄与率。収益の計算に使う
        while not self.env.is_terminal_state():
            # 行動aを選択する
            action = self._select_action_randomly()

            # 行動aを行い、報酬rを得る
            reward = self.env.exec_action(action)
            profit += (reward * contribution_rate)
            step_count += 1
            contribution_rate *= self.config['gamma']

            # ただし、終端状態での状態価値関数の値はその終端状態での報酬値とする
            if self.env.is_terminal_state():
                terminal_state = self.env.get_state()
                self.v_func[terminal_state] = reward

            # 無限ループ防止のため、一定の回数行動しても終端にたどり着かなければ打ち止め
            if step_count > self.config['playout_step_limit']:
                break

        # 環境をリセット
        self.env.reset()

        return profit

    # ランダムに行動を選択する
    def _select_action_randomly(self):
        actions = self.env.get_actions()
        idx = random.randint(0, len(actions) - 1)

        return actions[idx]

    # greedy法で行動を選択する
    def _select_action_with_greedy(self):
        original_state = self.env.get_state()
        max_v = None
        selected_action = None

        # 行動後の状態価値関数の値が最大となるような行動を選択する
        for action in self.env.get_actions():
            self.env.exec_action(action)
            state = self.env.get_state()

            if (max_v is None) or (self.v_func[state] > max_v):
                selected_action = action
                max_v = self.v_func[state]

            # 状態を元に戻す
            self.env.set_state(original_state)

        return selected_action

    # 状態価値関数を初期化して返す
    def _make_initial_v_function(self):
        init_v_func = {}
        for state in self.env.get_all_states():
            init_v_func[state] = 0

        return init_v_func
