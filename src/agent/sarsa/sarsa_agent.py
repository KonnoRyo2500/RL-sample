# 強化学習勉強用サンプルプログラム SARSAエージェントクラス

from itertools import product

from agent.base.agent_base import AgentBase
from agent.util.action_selector.greedy import Greedy
from agent.util.action_selector.epsilon_greedy import EpsilonGreedy
from common.const_val import Agent


# SARSAエージェントクラス
class SarsaAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env):
        super().__init__(env, Agent.Sarsa.value)
        action_space = env.get_action_space()
        state_space = env.get_state_space()

        # 行動価値関数
        self.q_func = self._make_initial_q_function(action_space, state_space)

        # 行動選択インスタンス
        self.greedy = Greedy(action_space)  # greedy
        self.epsilon_greedy = EpsilonGreedy(action_space, self.config['epsilon'])  # ε-greedy

        # 直前の状態とそこで選択した行動
        self.last_state_action = None

    # 次の行動を決定する
    def decide_action(self, env):
        state = env.get_state()
        action_space = env.get_action_space()
        available_actions = env.get_available_actions()
        q_values = [self.q_func[(state, a)] for a in action_space]

        if self.mode == SarsaAgent.OperationMode.Train:
            action = self.epsilon_greedy.select_action(available_actions, q_values)
        else:
            action = self.greedy.select_action(available_actions, q_values)

        # feedbackでTD誤差を計算するため、現状態と選択した行動を記録しておく
        self.last_state_action = (state, action)

        return action

    # 環境からの情報を自身にフィードバックする
    def feedback(self, reward, env):
        # 更新式: Q(s, a) = Q(s, a) + α(r + γQ(s', a') - Q(s, a))

        # Q(s', a')を得る
        last_state_action = self.last_state_action
        if env.is_terminal_state():
            next_q = 0
        else:
            next_state = env.get_state()
            next_action = self.decide_action(env)
            next_q = self.q_func[(next_state, next_action)]

        # TD誤差の計算でlast_state_actionを利用するため、last_state_actionはdecide_actionで
        # 更新されないようにする
        self.last_state_action = last_state_action

        # Q(s, a)はそのまま記述すると表記が長いため、短い名前の変数に入れておく
        q = self.q_func[self.last_state_action]

        # 行動価値関数Q(s, a)を更新する
        td_error = reward + self.config['gamma'] * next_q - q
        self.q_func[self.last_state_action] += self.config['alpha'] * td_error

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self, action_space, state_space):
        init_q_func = {}
        for state, action in product(state_space, action_space):
            init_q_func[(state, action)] = 0

        return init_q_func
