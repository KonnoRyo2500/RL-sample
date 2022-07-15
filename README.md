# 強化学習サンプル
## 概要

本ソフトは、シンプルな環境上で強化学習を行うサンプルプログラムです。

強化学習アルゴリズム、およびその実装方法の理解補助を目的としています。



## 実行環境

以下の環境で開発および動作確認を行っております。

|  環境  |                             詳細                             |
| :----: | :----------------------------------------------------------: |
|   OS   |       Windows 10 Home バージョン21H2 ビルド19044.1415        |
|  CPU   |                     Intel Core i5-10210U                     |
| Python |                         Python 3.9.2                         |
| Excel  | Microsoft Excel 2019 MSO(バージョン2112 ビルド16.0.14729.20156) 64ビット |



## インストール方法

pip+requirements.txtで必要なパッケージをインストールすればOKです。

```powershell
> pip install -r requirements.txt
```



## 実行方法

src/main.pyを、コマンドラインから実行します。実行時のオプションについては「実行オプション」の章をご参照ください。

```powershell
> cd src\
> python main.py (実行オプション)
```



## 実行オプション

実行時、以下の実行オプションを指定可能です。

| オプション名 | 必須? |     引数     |     デフォルト値      | 説明                                                         |
| :----------: | :---: | :----------: | :-------------------: | :----------------------------------------------------------- |
|   --config   |   ×   | ファイルパス | ..\config\config.yaml | 設定ファイル(YAML)のパス。                                   |
|   --method   |   ×   |    文字列    |      q_learning       | 使用する強化学習アルゴリズム。指定する文字列とアルゴリズムとの対応については下記参照。 |
|    --env     |   ×   |    文字列    |      grid_world       | 使用する環境。指定する文字列と環境との対応については下記参照。 |
|    --help    |   ×   |     なし     |           -           | ヘルプをコンソールに表示し、プログラムを終了する。           |



"--method"オプションと、選択される強化学習アルゴリズムの対応は以下の通りです。

| オプション指定文字列 | 強化学習アルゴリズム |
| :------------------: | :------------------: |
|      q_learning      |        Q学習         |
|        sarsa         |        SARSA         |
|     monte_carlo      |    モンテカルロ法    |
|        dqn           |        DQN          |



"--env"オプションと、選択される環境の対応は以下の通りです。

| オプション指定文字列 |                   環境                    |
| :------------------: | :---------------------------------------: |
|      grid_world      |                Grid World                 |
|       cartpole       | 倒立振り子 ※OpenAI GymのCartPole-v1を使用 |



## 設定ファイル

configディレクトリ直下に、YAML形式の設定ファイル(config.yaml)を置いてあります。この設定ファイルを変更することで、エージェントや環境に関する設定を変更できます。詳細は設定ファイル内のコメント("#"で始まる文字列)をご参照ください。



## 盤面定義ファイル

Grid World環境で使用する盤面(グリッド)を定義するExcelファイル(.xlsx)です。セルに罫線を引くだけで、盤面を記述できます。

盤面を記述する際は、以下のルールに従ってください。

- 盤面は矩形であること。
- 一番左上のマスはA1セルにあること。
- 盤面の周囲は罫線で囲うこと。
- 盤面の幅と高さを設定ファイルに記述すること。
- 盤面の内外問わず、セルに文字や関数を入れても良い。
- 罫線の太さや種類は何でも良い。



サンプルとして、"Grid.xlsx"というファイルを同梱しています。このファイルを参考に、盤面を定義してください。



## ライセンス

本ソフトは、MITライセンスです。ライセンスの制約のもとに、自己責任で自由にご利用いただけます。



## バグ報告

バグや不便な点がございましたら、お気軽に本リポジトリのIssueにご投稿ください。メールやLINEによる連絡もOKです。



## コントリビューティング

本リポジトリは自身の勉強を兼ねているため、すべてのコードを自分自身で実装する方針となっております。つきましては、プルリクエストなどは一切受け付けておりません。あしからずご了承ください。



## TODO

- 強化学習アルゴリズムの拡充
  - 動的計画法
  - 方策勾配法
  - etc.
- 深層強化学習アルゴリズムの実装
  - A3C
  - etc.
- 環境の拡充
  - OpenAI Gymとの更なる連携
  - シンプルなコンピュータゲーム(深層強化学習用)
    - OpenAI Gym Retroとの連携
    - オフライン環境上でプレイできる家庭用ゲームやPCゲーム
    - etc.
  - etc.
