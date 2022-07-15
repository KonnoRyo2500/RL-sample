# 強化学習勉強用サンプルプログラム DQNエージェントクラス

from enum import Enum

import torch
import torch.optim as optim

from agent.agent_base import AgentBase
from agent.dqn.dqn_network import DqnNetwork
from agent.dqn.experience_buffer import ExperienceBuffer
from agent.dqn.huber_td_loss import HuberTDLoss
from agent.util.select_action import epsilon_greedy

# DQNエージェントクラス
class DqnAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)

        in_size = len(self.env.get_state())
        out_size = len(self.env.get_whole_action_space())

        # 行動価値関数出力用ネットワーク
        self.q_network = DqnNetwork(in_size=in_size, out_size=out_size)
        self.target_network = DqnNetwork(in_size=in_size, out_size=out_size)

        # 経験バッファ
        self.exp_buffer = ExperienceBuffer(self.config['batch_size'])

        # オプティマイザ
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['alpha'])

        # 誤差関数
        self.criterion = HuberTDLoss()

    # 学習済みエージェントにエピソードをプレイさせる
    def play(self):
        pass

    # エージェントを学習させる
    def train(self):
        for i in range(self.config['num_episode']):
            self._episode()

        self.env.reset()

    # エピソードの実行(学習用)
    def _episode(self):
        state = self.env.get_state()
        step_count = 0
        while not self.env.is_terminal_state():
            self._step()

            self._update_q_network()

            # Target Networkの更新は一定ステップごとに行う
            if step_count % self.config['target_update_period'] == 0:
                self._update_target_network()

            step_count += 1

        # 環境と経験バッファをリセットする
        self.env.reset()
        self.exp_buffer.clear()

    # 1ステップ実行
    def _step(self):
        # 現状態sを取得する
        state = self.env.get_state()

        # Target Networkにsを入力し、sに対応する
        # 行動価値関数Q(s, a)を取得する
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_func = self.target_network(state_tensor)

        # 現在の行動空間で選択できない行動に対応する行動価値をQ(s, a)から除外する
        whole_action_space = self.env.get_whole_action_space()
        action_space = self.env.get_current_action_space()
        q_func = [q_func[i].item() for i, a in enumerate(whole_action_space) if a in action_space]

        # Q(s, a)をもとに、ε-greedy法で行動aを決定する
        action = epsilon_greedy(action_space, q_func, state, self.config['epsilon'])

        # 環境上で行動aを実行し、次状態s'と報酬rを得る
        reward = self.env.exec_action(action)
        next_state = self.env.get_state()

        # Reward Clippingを実施し、rの値域を[-1,1]に制限する
        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1

        # 経験e = (s, a, s', r)を作成し、経験バッファに追加する
        experience = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward
        }
        self.exp_buffer.append(experience)

        # 各種ネットワークのパラメータ更新は別関数にて行う

    # Experience Replayにより、Q Networkのパラメータを更新する
    def _update_q_network(self):
        # 経験バッファから一定個数の経験(経験バッチ)を取り出す
        exp_batch = self.exp_buffer.sample()

        # 経験が十分な量蓄積されていなければ更新しない
        if exp_batch is None:
            return

        # Q Networkの勾配を初期化
        self.optimizer.zero_grad()

        # Q Networkから行動価値関数を取得
        q_func = self.q_network(torch.tensor(exp_batch['states']).float())

        # Target Networkからも行動価値関数を取得(これを教師データとする)
        # TD誤差を求めるため、Target Networkには次の状態s'を入力する
        next_states_tensor = torch.tensor(exp_batch['next_states']).float()
        target_q_func = self.target_network(next_states_tensor)

        # 経験から得られた各行動を、全行動空間中のインデックスに変換する
        # Tensorに変換した際に縦ベクトルにするため、リストのリストにする
        whole_action_space = self.env.get_whole_action_space()
        action_indices = [[whole_action_space.index(a)] for a in exp_batch['actions']]
        action_indices = torch.tensor(action_indices)

        # 各種行動価値関数と経験に記録された行動からTD誤差の計算に必要な行動価値を抽出する
        q_values = q_func.gather(dim=1, index=action_indices).squeeze()
        target_q_values = torch.max(target_q_func, dim=1).values

        # TD誤差を求め、そのHuber誤差を算出する(これを損失とする)
        rewards_tensor = torch.tensor(exp_batch['rewards']).float()
        loss = self.criterion(q_values, target_q_values, rewards_tensor, self.config['gamma'])

        # ミニバッチ学習により、Q Networkのパラメータを更新する
        loss.backward()
        self.optimizer.step()

    # Q Networkのパラメータを利用し、Target Networkのパラメータを更新する
    def _update_target_network(self):
        # Q Networkのパラメータをそのままコピーする(Hard Update)
        self.target_network.load_state_dict(self.q_network.state_dict())
