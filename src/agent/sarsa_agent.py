# 強化学習勉強用サンプルプログラム SARSAエージェントクラス

from agent.q_agent import QAgent
from common.const_val import *

# SARSAエージェントクラス
# 行動価値関数の更新式以外はQAgentと同じなので、QAgentを継承して実装する
class SarsaAgent(QAgent):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)
        self.next_action = None # 前回のステップで求めた行動a'

    # 1ステップ実行
    def _step(self):
        # 更新式: Q(s, a) = Q(s, a) + α(r + γQ(s', a') - Q(s, a))

        # 現状態sを得る
        state = self.env.get_state()

        # 状態sで可能な行動を得る
        actions = self.env.get_actions()

        # 行動aを決定する
        if self.next_action is not None:
            action = self.next_action
        else:
            action = self._select_action_with_epsilon_greedy()

        # 次状態s'と報酬rを得る
        reward = self.env.exec_action(action)
        next_state = self.env.get_state()

        # Q(s', a')を得る
        if self.env.is_terminal_state():
            next_q = 0
            self.next_action = None
        else:
            next_actions = self.env.get_actions()
            next_action = self._select_action_with_epsilon_greedy()
            next_q = self.q_func[(next_state, next_action)]
            self.next_action = next_action

        # Q(s, a)はそのまま記述すると表記が長いため、短い名前の変数に入れておく
        q = self.q_func[(state, action)]

        # 行動価値関数Q(s, a)を更新する
        td_error = reward + self.config['gamma'] * next_q - q
        self.q_func[(state, action)] = q + self.config['alpha'] * td_error

