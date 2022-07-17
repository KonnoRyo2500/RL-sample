# 強化学習勉強用サンプルプログラム 経験バッファクラス

from collections import deque
import random

# 経験バッファクラス
class ExperienceBuffer:
    # コンストラクタ
    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        self.exp_buffer = deque(maxlen=capacity)

    # 経験バッファに経験を追加する
    def append(self, exp):
        # 最大サイズを超過した場合は、一番古い経験がバッファから削除される
        self.exp_buffer.append(exp)

    # 経験バッファから経験をミニバッチとして取り出す
    def sample(self):
        # 経験が足りない場合はNoneを返す
        if len(self.exp_buffer) < self.batch_size:
            return None

        # 経験をランダムに取り出す
        exp_batch = random.sample(self.exp_buffer, self.batch_size)

        # 辞書で取り出せるようにする
        exp_batch = {
            'states': [e['state'] for e in exp_batch],
            'actions': [e['action'] for e in exp_batch],
            'next_states': [e['next_state'] for e in exp_batch],
            'rewards': [e['reward'] for e in exp_batch],
        }

        return exp_batch
