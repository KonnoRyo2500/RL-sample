# 強化学習勉強用サンプルプログラム 一人用ゲームフレームワーククラス

from game.game_base import GameBase
from common.const_val import Game


# 一人用ゲームフレームワーククラス
class SinglePlayerGame(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.SinglePlayer.value)

        if len(agents) > 1:
            print(f'警告: 複数のエージェントが指定されました。最初に指定されたエージェントのみ使用されます')

        # 一人用ゲームのエージェントに対して複数形の名前を用いるのは不自然なので、単数形にしておく
        # 型もリストではなくエージェントのクラスそのものとする
        self.agent = self.agents[0]

        # 環境を初期化
        self.env.reset()

    # エージェントを学習させる
    def train_agent(self):
        self.agent.switch_to_train_mode()

        for i in range(self.config['num_episode']):
            self._train_episode()
            self.env.reset()

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        self.agent.switch_to_play_mode()

        while not self.env.is_terminal_state():
            # エージェントが行動を決定する
            action = self.agent.decide_action(self.env)

            # 決定した行動を環境に対して行い、報酬を得る
            reward = self.env.exec_action(action)

            # 行動を表示する
            print(f'行動{action}を選択し、報酬{reward}が得られました')

        self.env.reset()

    # 1エピソード分の学習を行う
    def _train_episode(self):
        while not self.env.is_terminal_state():
            # エージェントが次の行動を決定する
            action = self.agent.decide_action(self.env)

            # 決定した行動を環境に対して行い、報酬を得る
            reward = self.env.exec_action(action)

            # エージェントは、得られた報酬と変化した環境の状態をもとに自身のパラメータを更新する
            self.agent.feedback(reward, self.env)
