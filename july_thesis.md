# 7月論文用

- pretrain_lr = 0.01, スケジューラなし、pretrain_epoch = 100
- filter: r50_s01.pt
- rwp_n10_a0500_r100_p10_s01.pt
- optimizer はtrainとpretrainで揃える
- epoch = 3000, batch_size = 16

## 同期チェック

上田のデータで色々なseedの条件をつけずに実行
-> test.out と 2023-07-02-12
-> 何回やっても同じになることを確認し2023-07-02-13として残す
normalizeの引数が微妙に違う->2023-07-02-14(test.out)->変わってなさそう

GPU1と2で微妙に違う -> 1からデータディレクトリを引っ張ってきて実行する -> 2023-07-03-11 -> だめ
上田のデータでseed条件をつけて実行
すべてのモジュールのバージョンを固定して解決

## VGG

### 2023-07-03-21_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID -> nohup.out(完了)
- the average of the last 10 epoch: 0.8925574499629355
- the std of the last 10 epoch: 0.03178379255325632

### 2023-07-09-22

- GPU2
- SGDを使用。lr=0.01 IID -> nohup4.out(実行中)

## mobilenet

### 2023-07-07-00_gpu2

- GPU2
- Adamを使用。lr=0.01 IID
- 後でやり直し

### 2023-07-04-14_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID -> nohup2.out
- the average of the last 10 epoch: 0.8887620459599703
- the std of the last 10 epoch: 0.037707479233585646

### 2023-07-07-13_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID, pre_trainのみ
- the average of the last 10 epoch: 0.519518161601186
- the std of the last 10 epoch: 0.052939003996110616

### 2023-07-09-10_gpu2

- GPU2
- SGDを使用。lr=0.01 IID -> nohup3.out(実行中)

### 2023-07-07-09_gpu2

- GPU2
- SGDを使用。lr=0.01 IID, pre_trainのみ
-> nohup5.out(実行中)

## 残り

- Adam, VGG, pretrainのみ, IIDは未記述。
- Adam, mobile, pretrainのみ, IIDだけ未記述。
- それ以外は上に記述済み
