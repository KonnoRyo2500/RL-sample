# 強化学習勉強用サンプルプログラム Q学習エージェントクラス

from itertools import product
from pprint import pprint

from agent.agent_base import AgentBase
from agent.util.action_selector.greedy import Greedy
from agent.util.action_selector.epsilon_greedy import EpsilonGreedy

# Q学習エージェントクラス
class QAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)
        self.q_func = self._make_initial_q_function()

        # 行動選択インスタンス
        action_space = self.env.get_action_space()
        self.greedy = Greedy(action_space) # greedy
        self.epsilon_greedy = EpsilonGreedy(action_space, self.config['epsilon']) # ε-greedy

    # 学習済みエージェントにエピソードをプレイさせる
    def play(self):
        while not self.env.is_terminal_state():
            # 状態を取得(表示用)
            state = self.env.get_state()

            # greedy法で行動を選択
            q_values = [self.q_func[(state, a)] for a in self.env.get_action_space()]
            available_actions = self.env.get_available_actions()
            action = self.greedy.select_action(available_actions, q_values)

            # 行動する
            reward = self.env.exec_action(action)
            print(f'状態 {state} で行動 {action} を選択しました。')

        if reward != 0:
            print(f'エピソードをプレイし、報酬 {reward} が得られました。')

        self.env.reset()

    # エージェントを学習させる
    def train(self):
        for i in range(self.config['num_episode']):
            self._episode()

        pprint(self.q_func)
        self.env.reset()

    # エピソードの実行(学習用)
    def _episode(self):
        state = self.env.get_state()
        while not self.env.is_terminal_state():
            self._step()
            state = self.env.get_state()

        self.env.reset()

    # 1ステップ実行
    def _step(self):
        # 更新式: Q(s, a) = Q(s, a) + α(r + γmax(a')Q(s', a') - Q(s, a))

        # 現状態sを取得
        state = self.env.get_state()

        # 行動aを決定する
        q_values = [self.q_func[(state, a)] for a in self.env.get_action_space()]
        available_actions = self.env.get_available_actions()
        action = self.epsilon_greedy.select_action(available_actions, q_values)

        # 次状態s'と報酬rを得る
        reward = self.env.exec_action(action)
        next_state = self.env.get_state()

        # max(a')Q(s', a')を得る
        if self.env.is_terminal_state():
            next_max_q = 0
        else:
            next_actions = self.env.get_action_space()
            next_max_q = max([self.q_func[(next_state, a)] for a in next_actions])

        # Q(s, a)はそのまま記述すると表記が長いため、短い名前の変数に入れておく
        q = self.q_func[(state, action)]

        # 行動価値関数Q(s, a)を更新する
        td_error = reward + self.config['gamma'] * next_max_q - q
        self.q_func[(state, action)] = q + self.config['alpha'] * td_error

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self):
        init_q_func = {}
        for state, action in product(self.env.get_state_space(), self.env.get_action_space()):
            init_q_func[(state, action)] = 0

        return init_q_func
