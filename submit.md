# 使用したデータについて
## 画像分布
| ラベル(ノード番号) | トレーニングデータ数 | テストデー多数 |
| --- | --- | --- |
| 安田講堂(0) | 106 | 24 |
| 工2(1) | 159 | 36 |
| 工3(2)  | 101 | 24 |
| 工13(3)  | 97 | 22 |
| 工4(4)  | 119 | 26 |
| 工8(5)  | 133 | 31 |
| 工1(6)  | 121 | 28 |
| 工6(7)  | 169 | 40 |
| 列品館(8)  | 139 | 31 |
| 法文1(9)  | 205 | 48 |

## 各ノードへのトレーニング用のデータの分配
| ラベル(ノード番号) | トレーニングデータ数 |
| --- | --- |
| 安田講堂(0) | 108 |
| 工2(1) | 167 |
| 工3(2)  | 127 |
| 工13(3)  | 110 |
| 工4(4)  | 126 |
| 工8(5)  | 140 |
| 工1(6)  | 120 |
| 工6(7)  | 145 |
| 列品館(8)  | 125 |
| 法文1(9)  | 181 |

# パラメータ

- batch_size:20
- contact file: rwp_n10_a0500_r100_p10_s01.json
- filter file: filter_r50_s01.pt
- モデルとしてmobilenetを使用
- optimizerはSGDを使用
- pretrainのlrは0.05から30epochごとに0.3倍していき150epochまで
- trainはlr=0.01で2000epochまで
    ## 前処理
    - normalizeの値は各ノードのデータの平均を使用。実際にはtrainの値がノードごとに違う。
    ```
    test transform: Compose(
        CenterCrop(size=(224, 224))
        ConvertImageDtype()
        Normalize(mean=(0.4794929623603821, 0.4756014943122864, 0.4341307580471039), std=(0.23307910561561584, 0.236463725566864, 0.2503257989883423))
    )
    ```
    ```
    train transform: Compose(
        RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=warn)
        ConvertImageDtype()
        Normalize(mean=(0.4592202603816986, 0.44688066840171814, 0.409477174282074), std=(0.1909409612417221, 0.19420726597309113, 0.20837080478668213))
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    )
    ```
    ## モデルの最終層
    ```
    ├─Sequential: 1-2                                  [1, 10]                   --
    │    └─Dropout: 2-20                               [1, 1280]                 --
    │    └─Sequential: 2-21                            [1, 10]                   --
    │    │    └─Linear: 3-24                           [1, 256]                  327,936
    │    │    └─ReLU: 3-25                             [1, 256]                  --
    │    │    └─Linear: 3-26                           [1, 10]                   2,570
    ====================================================================================================
    Total params: 2,554,378
    Trainable params: 330,506
    Non-trainable params: 2,223,872
    Total mult-adds (Units.MEGABYTES): 299.86
    ====================================================================================================
    ```

# 結果
| ノード番号 | 初期状態の精度 | 最終状態の精度 |
| --- | --- | --- |
| 0 | 0.48036 | 0.85248 |
| 1 | 0.41883 | 0.81171 |
| 2 | 0.46701 | 0.76649 |
| 3 | 0.46850 | 0.85026 |
| 4 | 0.40400 | 0.75463 |
| 5 | 0.51964 | 0.86953 |
| 6 | 0.42031 | 0.66271 |
| 7 | 0.36027 | 0.81097 |
| 8 | 0.26612 | 0.76946 |
| 9 | 0.29429 | 0.85100 |
-  ## acc
    <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/acc.png" width="500">
-  ## 混同行列
    - ノードiはi番目のラベルのデータの50%を持つ
        ### ノード0
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node0.png" width="500"><br>
        ### ノード1
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node1.png" width="500"><br>
        ### ノード2
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node2.png" width="500"><br>
        ### ノード3
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node3.png" width="500"><br>
        ### ノード4
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node4.png" width="500"><br>
        ### ノード5
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node5.png" width="500"><br>
        ### ノード6
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node6.png" width="500"><br>
        ### ノード7
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node7.png" width="500"><br>
        ### ノード8
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node8.png" width="500"><br>
        ### ノード9
        <img src="../data-raid/static/WAFL_research/2023-06-24-16/images/normalized_confusion_matrix/normalized-cm-epoch2000-node9.png" width="500"><br>

# 事前学習モデルについて
## mobilenet_v2
ImageNet, COCO, ssd, DADA-seg
## resnet152
ImageNet, 
## vit
Imagenet, CIFAR-10, CIFAR-100, Oxford-IIIT Pets, Oxford Flowers-102, VTAB (19 tasks)
-> ImageNet, ImageNet-21k
## vgg
ILSVRC-2012 dataset (ImageNet)