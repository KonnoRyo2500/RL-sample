# 強化学習勉強用サンプルプログラム 複数人プレイヤーゲームフレームワーククラス

from tqdm import tqdm

from game.base.game_base import GameBase
from common.const_val import Game


# 複数人プレイヤーゲームフレームワーククラス
class MultiPlayerGame(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.MultiPlayer.value)

    # エージェントを学習させる
    def train_agent(self):
        total_reward = 0

        with tqdm(range(self.config.num_episode), desc='[学習中]') as train_progress_bar:
            for i in train_progress_bar:
                # エピソードを実施
                total_reward += self._train_episode()
                self.env.reset()

                # 各エピソードの合計報酬の平均を更新し、表示する
                # tqdmは表示する値の桁数が増えると小数点以下の値を表示しなくなるため、文字列にして渡す
                avg_reward = total_reward / (i + 1)
                train_progress_bar.set_postfix({'平均報酬': f"{avg_reward:.2f}"})

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        # すべてのプレイヤーが終端状態になるまで、ゲームを続ける
        is_terminated = [False] * len(self.agents)

        print('プレイ開始')
        while not all(is_terminated):
            # 現在のプレイヤーがすでに終端状態にある
            player = self.env.get_player()
            if self.env.is_terminal_state():
                is_terminated[player] = True
                self.env.switch_to_next_player()
                continue

            # エージェントが次の行動を決定する
            agent = self.agents[player]
            action = agent.decide_action(self.env)

            # 決定した行動を環境に対して行い、報酬を得る
            reward = self.env.exec_action(action)

            # 行動を表示する
            print(f'行動{action}を選択し、報酬{reward}が得られました')

            # 次のプレイヤーに交代
            self.env.switch_to_next_player()

        # 環境をリセット
        self.env.reset()

    # 学習時の1エピソード
    def _train_episode(self):
        is_terminated = [False] * len(self.agents)
        total_reward = 0

        # すべてのプレイヤーが終端状態になるまで、ゲームを続ける
        while not all(is_terminated):
            # 現在のプレイヤーがすでに終端状態にある
            player = self.env.get_player()
            if self.env.is_terminal_state():
                is_terminated[player] = True
                self.env.switch_to_next_player()
                continue

            # 選択できる行動がなければパス
            if len(self.env.get_available_actions()) == 0:
                self.env.switch_to_next_player()
                continue

            # エージェントが次の行動を決定する
            agent = self.agents[player]
            action = agent.decide_action(self.env)

            # 決定した行動を環境に対して行い、報酬を得る
            reward = self.env.exec_action(action)
            total_reward += reward

            # エージェントは、得られた報酬と変化した環境の状態をもとに自身のパラメータを更新する
            agent.feedback(reward, self.env)

            # 次のプレイヤーに交代
            self.env.switch_to_next_player()

        return total_reward
