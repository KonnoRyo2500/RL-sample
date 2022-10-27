# 強化学習勉強用サンプルプログラム DQNエージェントクラス

import copy
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn as nn

from agent.base.agent_base import AgentBase
from agent.implementation.dqn.dqn_network import DqnNetwork
from agent.implementation.dqn.experience_buffer import ExperienceBuffer
from agent.util.action_selector.greedy import Greedy
from agent.util.action_selector.epsilon_greedy import EpsilonGreedy
from common.const_val import Agent


# DQNエージェントクラス
class DqnAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env):
        super().__init__(env, Agent.Dqn.value)

        action_space = env.get_action_space()
        in_size = len(env.get_state())
        out_size = len(action_space)

        # 行動価値関数出力用ネットワーク
        self.q_network = DqnNetwork(in_size, out_size, self.config)
        self.target_network = copy.deepcopy(self.q_network)

        # 経験バッファ
        self.exp_buffer = ExperienceBuffer(
            self.config.batch_size,
            self.config.expbuf_capacity)

        # オプティマイザ
        self.optimizer = optim.RMSprop(
            self.q_network.parameters(),
            lr=self.config.alpha,
            alpha=self.config.rmsprop_alpha,
            eps=self.config.rmsprop_epsilon)

        # 行動選択アルゴリズム
        self.greedy = Greedy(action_space)  # greedy
        self.epsilon_greedy = EpsilonGreedy(
            action_space,
            self.config.epsilon_init,
            self.config.epsilon_diff,
            self.config.epsilon_min)  # ε-greedy

        # 誤差関数
        self.criterion = nn.MSELoss()

        # 総ステップ数
        self.total_step_count = 0

        # 直前の状態とそこで選択した行動
        self.last_state_action = None

    # 環境の情報を参照し、次の行動を決定する
    def decide_action(self, env):
        state = env.get_state()
        available_actions = env.get_available_actions()
        q_values = self.q_network(torch.tensor(state).float())

        if self.mode == DqnAgent.OperationMode.Train:
            action = self.epsilon_greedy.select_action(available_actions, q_values)
        else:
            action = self.greedy.select_action(available_actions, q_values)

        # feedbackでTD誤差を計算するため、現状態と選択した行動を記録しておく
        self.last_state_action = (state, action)

        return action

    # 環境からの情報を自身にフィードバックする
    def feedback(self, reward, env):
        self._step(reward, env)

        # Q Network, Target Networkの更新は一定ステップごとに行う
        if self.total_step_count % self.config.q_update_period == 0:
            self._update_q_network(env)
        if self.total_step_count % self.config.target_update_period == 0:
            self._update_target_network()

        # εを減少させる
        if self.total_step_count > self.config.epsilon_decrement_step:
            self.epsilon_greedy.decrement_epsilon()

        self.total_step_count += 1

    # 1ステップ実行
    def _step(self, reward, env):
        # 次状態s'を得る
        next_state = env.get_state()

        # Reward Clippingを実施し、rの値域を[-1,1]に制限する
        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1

        # 経験e = (s, a, s', r, t)を作成し、経験バッファに追加する
        # tはs'が終端状態かを表す真偽値
        state, action = self.last_state_action
        is_terminated = env.is_terminal_state()
        exp_class = namedtuple(
            'Experience',
            ['state', 'action', 'next_state', 'reward', 'is_terminated']
        )
        experience = exp_class(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            is_terminated=is_terminated,
        )
        self.exp_buffer.append(experience)

        # 各種ネットワークのパラメータ更新は別関数にて行う

    # Experience Replayにより、Q Networkのパラメータを更新する
    def _update_q_network(self, env):
        # 経験バッファから一定個数の経験(経験バッチ)を取り出す
        exp_batches = self.exp_buffer.sample()

        # 経験が十分な量蓄積されていなければ更新しない
        if exp_batches is None:
            return

        for exp_batch in exp_batches:
            # 取り出した各経験バッチで、Q Networkを更新する
            self._experience_replay(env, exp_batch)

    # 経験バッファより再生した経験バッチを用いて、Q Networkのパラメータを更新する(Experience Replayを1回行う)
    def _experience_replay(self, env, exp_batch):
        # Q NetworkからQ(s, a)を得る
        states_tensor = torch.tensor(exp_batch.states).float()
        q_values = self.q_network(states_tensor)

        # Target NetworkからはQ(s', a)を得る
        next_states_tensor = torch.tensor(exp_batch.next_states).float()
        target_q_values = self.target_network(next_states_tensor)

        # max(a)[Q(s', a)]を求める
        max_q_func, _ = torch.max(target_q_values.data, 1)

        # 報酬rを得る
        rewards_tensor = torch.tensor(exp_batch.rewards).float()

        # 教師信号r + γmax(a)[Q(s', a)]を計算する
        # なお、Q(s', a)が最大値をとらないaについてはQ(s, a)と同じとする
        target = q_values.detach().clone()
        actions = exp_batch.actions
        terminations = exp_batch.terminations
        for i in range(len(exp_batch)):
            action_idx = env.get_action_space().index(actions[i])
            target[i, action_idx] = rewards_tensor[i] + self.config.gamma * max_q_func[i] * (not terminations[i])

        # Q Networkの勾配を初期化
        self.optimizer.zero_grad()

        # 誤差を計算する
        loss = self.criterion(q_values, target)

        # ミニバッチ学習により、Q Networkのパラメータを更新する
        loss.backward()
        self.optimizer.step()

    # Q NetworkのパラメータをTarget Networkのパラメータに反映する
    def _update_target_network(self):
        # Q Networkのパラメータをそのままコピーする(Hard Update)
        self.target_network = copy.deepcopy(self.q_network)
