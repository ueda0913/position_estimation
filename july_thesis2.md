# 7月論文用(修正版)

- pretrain_lr = 0.01, スケジューラなし、pretrain_epoch = 100
- filter: r50_s01.pt
- rwp_n10_a0500_r100_p10_s01.pt
- optimizer はtrainとpretrainで揃える
- epoch = 3000, batch_size = 16
- ミスを修正後の内容

## 同期チェック

上田のデータで色々なseedの条件をつけずに実行
-> test.out と 2023-07-02-12
-> 何回やっても同じになることを確認し2023-07-02-13として残す
normalizeの引数が微妙に違う->2023-07-02-14(test.out)->変わってなさそう

GPU1と2で微妙に違う -> 1からデータディレクトリを引っ張ってきて実行する -> 2023-07-03-11 -> だめ
上田のデータでseed条件をつけて実行
すべてのモジュールのバージョンを固定して解決

## VGG

### 2023-07-10-17_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID
- the average of the last 10 epoch: 0.7089032258064516
- the std of the last 10 epoch: 0.037532665151064785

### 2023-07-10-17_gpu1

- GPU1
- SGDを使用。lr=0.01 IID
- the average of the last 10 epoch: 0.7950322580645162
- the std of the last 10 epoch: 0.019487003949491717

#### 2023-07-11-07_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID, pretrainのみ
- the average of the last 10 epoch: 0.43390322580645163
- the std of the last 10 epoch: 0.044508371970676336

#### 2023-07-11-07_gpu1

- GPU1
- SGDを使用。lr=0.01 IID, pretrainのみ
- the average of the last 10 epoch: 0.5736451612903226
- the std of the last 10 epoch: 0.022273925809018685

## mobilenet

### 2023-07-10-16_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID
- the average of the last 10 epoch: 0.7391290322580645
- the std of the last 10 epoch: 0.042421810949687694

### 2023-07-10-16_gpu1

- GPU1
- SGDを使用。lr=0.01 IID
- the average of the last 10 epoch: 0.8288064516129032
- the std of the last 10 epoch: 0.021171677865993038

### 2023-07-11-01_gpu2

- GPU2
- SGDを使用。lr=0.01 non-IID, pre_trainのみ
- the average of the last 10 epoch: 0.46419354838709675
- the std of the last 10 epoch: 0.05513499780208513

### 2023-07-11-10

- GPU2
- SGDを使用。lr=0.01 IID, pre_trainのみ
- the average of the last 10 epoch: 0.6031612903225807
- the std of the last 10 epoch: 0.028522662757069665
