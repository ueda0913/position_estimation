# rwp

- 単純にランダムウォーク。
- a小ほど出会う可能性は上がる
- r大ほど1度に同時に会うノードの数が増える
- p大ほど一度会ってから長い間同じノードと接する -> 汎化が進む
- 今後の方針としてはp大の状態でたくさんのノードと合う状態にすることでなんとかならないか

## vgg

- vggはmobilenetよりも複雑なモデルのため、表現力が高すぎてか学習気味？
- mobilenetよりも精度が出ない
- これをどうするかを含めて考えたい
      ### 2023-06-09-13
      vgg の最初の条件やり直し <- memoのnormalizeのやつ
      contact file: rwp_n10_a0500_r100_p10_s01.json
      filter file: なし
      lr=0.005
      ### 2023-06-10-11
      2023-06-09-13の結果が2023-06-09-14や2023-06-09-15よりも10％程度悪そうなので
      とりあえずvggでlr=0.01でやってみる
      contact file: rwp_n10_a0500_r100_p10_s01.json
      filter file: なし
      -> 75%くらい。少し上がった
      -> 学習を延長。3000まで -> nohup2.out
      ### 2023-06-12-11
      pが大きくなったらtrain_accは増えるのか？交換時間が不足していたのか確認
      vggでpretrainのlrは0.05から30epochごとに0.3倍していく
      trainはlr=0.01
      contact file: rwp_n10_a0500_r100_p40_s01.json
      -> まだ上がりそう。大体80%前後.pretrainは96.7%くらいまで行った
      -> 実はp10の時と変わらない説がある -> p10の時の学習を延長
      ### 2023-06-12-12
      accが90%ぐらいということはモデル合成の寄与が大きすぎて自前のデータに対する学習が足りていない？
      ということでlrをもっとあげてみる
      vggでpretrainのlrは0.05から30epochごとに0.3倍していく
      trainはlr=0.05
      contact file: rwp_n10_a0500_r100_p10_s01.json
      -> これもまだ上がりそう。ただし、train_accはむしろ0.01の方が良さそう。val_accはこっちの方がフラフラしてる
      ### 2023-06-12-22
      contact p=100にしてみる
      vggでpretrainのlrは0.05から30epochごとに0.3倍していく
      trainはlr=0.01
      contact file: rwp_n10_a0500_r100_p100_s01.json
      -> まだ上がるかも、80~82%くらい
      -> 長すぎるのか？忘れてしまうのかも...

## mobilenet

- 軽量だが、以前として精度が足りん

      ### 2023-06-09-14
      mobilenet
      contact file: rwp_n10_a0500_r100_p10_s01.json
      filter file: なし
      lr=0.01
      ### 2023-06-10-19
      mobilenetでlrを下げた時を考えてみる
      pretrainはlr=0.01
      trainのみlr=0.001
      contact file: rwp_n10_a0500_r100_p10_s01.json
      filter file: なし
      lrが小さすぎてtrainが進んでなくて、train_accが下がり続けてる
      ### 2023-06-10-22
      pが大きくなったらtrain_accは増えるのか？交換時間が不足していたのか確認
      mobilenetでpretrainのlrは0.01から30epochごとに0.3倍していく
      trainはlr=0.01
      contact file: rwp_n10_a0500_r100_p40_s01.json
      filter file: なし
      ### 2023-06-12-13
      上のやつが良さそうなので、もっと長くする
      pretrainのlrは0.05から30epochごとに0.3倍していく
      trainはlr=0.01
      contact file: rwp_n10_a0500_r100_p40_s01.json
      -> すごい性能が良かった
      ### 2023-06-24-10
      contact p=100にしてみる
      mobileでpretrainのlrは0.05から30epochごとに0.3倍していく
      trainはlr=0.01
      contact file: rwp_n10_a0500_r100_p100_s01.json
      -> めっちゃいい

## NON-iid

- データ分布に偏りがある場合

      ### mobilenet
      - pretrainのlrは0.05から30epochごとに0.3倍していく
      - trainはlr=0.01
      #### 2023-06-24-16
      contact file: rwp_n10_a0500_r100_p10_s01.json
      filter file: filter_r50_s01.pt
      #### 2023-06-24-21
      contact file: rwp_n10_a0500_r100_p40_s01.json
      filter file: filter_r50_s01.pt
      -> 1000epoch追加学習(2000+1000)
      -> 85%くらい
      #### 2023-06-25-00
      contact file: rwp_n10_a0500_r100_p100_s01.json
      filter file: filter_r50_s01.pt
      -> 1000epoch追加学習(2000+1000)
      -> 80から85%くらい
      ### vision transformer
      - pretrainのlrは0.05から30epochごとに0.3倍していく
      - trainはlr=0.01
      #### 2023-06-28-02
      contact file: rwp_n10_a0500_r100_p40_s01.json
      filter file: filter_r50_s01.pt
      -> 95%以上？めっちゃいい

      ### resnet
      - pretrainのlrは0.05から30epochごとに0.3倍していく
      - trainはlr=0.01
      #### 2023-06-28-03
      contact file: rwp_n10_a0500_r100_p40_s01.json
      filter file: filter_r50_s01.pt
      -> nohup2.out

# cse

- communityをnodeの数分だけ作って同じcommunity内でモデルの交換を行う
- c大ほど1つのcommunityに含まれるnode数が増え、1度に合うノードの種類が増える
- tt大ほどnodeのcommunity遷移に時間がかかり、nodeが誰とも会っていない時間が増える(いる？)
- tp大ほどnodeのcommunity遷移が起きやすい

# その他のコンタクトパターン

- rwp, cse以外

## 全結合

- 全ノードが全ノードと交換をずっと行う場合
      ### 2023-06-09-15
        vgg
        全てのノードが密に結合して交換し続けるとどうなるのか
        -> 予想はめっちゃ性能でる(従来型と同じ程度)
        lr=0.005
        実際は80%くらいの精度
        -> もっとやれば上がりそう、pretrainの改善も、後でやる

# fine-tuningとの併用

- 各ノードはfine-tuningを行う

      ## 2023-06-03-12
        contact file: rwp_n10_a0500_r100_p10_s01.json
        filter file: filter_r10_s01.pt
        mobilenet
        終わってた -> delete

# pre-trainオンリーの場合

- aiu
      ## 2023-06-09-12
        mobilenetを使った時に、pretrainのみを行うとどうなるか
        lr=0.01
        40epochぐらいで頭打ち。60%くらい
      ## 2023-06-10-09
        上の条件でlr=0.001とした場合
        60%ぐらいまでは行くが、100epochくらいは必要
        ぶれかたはあまり変わらないイメージ
      ## 2023-06-26-13
        mobilenetを使った
        pretrainのlrは0.05から30epochごとに0.3倍していく
