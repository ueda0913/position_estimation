# WAFL_position_estimationのメモ

## IIDの結果

### 2023-09-20-07(lrの減少を早く)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.8408881199538639
- the std of the last 10 epoch: 0.007425158811673623
- the maxmize of the last 10 epoch: 0.8512110726643599
- the minimum of the last 10 epoch: 0.8166089965397924

### 2023-09-20-00(lrの減少を早く+ノードの交換の速度をup)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- the average of the last 10 epoch: 0.8566608996539793
- the std of the last 10 epoch: 0.006046113806215628
- the maxmize of the last 10 epoch: 0.8719723183391004
- the minimum of the last 10 epoch: 0.8408304498269896

### 2023-09-20-16(lrの減少を早く+過去の記憶使用)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- use_previous_memory
- the average of the last 10 epoch: 0.8334198385236448
- the std of the last 10 epoch: 0.006073233385436657
- the maxmize of the last 10 epoch: 0.8442906574394463
- the minimum of the last 10 epoch: 0.8166089965397924

### 2023-09-20-13(lrの減少を早く+ノードの交換の速度をup+過去の記憶使用)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- use_previous_memory
- the average of the last 10 epoch: 0.8517589388696655
- the std of the last 10 epoch: 0.007035703838325886
- the maxmize of the last 10 epoch: 0.8650519031141869
- the minimum of the last 10 epoch: 0.8373702422145328

ここまでが中間前。これ以降は、lrの減少を早く+ノードの交換の速度をupを基本に考える。

### 2023-10-22-20(lrを0.2に固定)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.2
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- the average of the last 10 epoch: 0.846885813148789
- the std of the last 10 epoch: 0.0032113719468406317
- the maxmize of the last 10 epoch: 0.8546712802768166
- the minimum of the last 10 epoch: 0.8373702422145328

### 2023-10-23-11(cos類似度を使用)

- vit
- epoch: 150->3000
- fl_coefficiency: cos類似度を使用
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- the average of the last 10 epoch: 0.8520761245674741
- the std of the last 10 epoch: 0.007110443823360739
- the maxmize of the last 10 epoch: 0.8650519031141869
- the minimum of the last 10 epoch: 0.8408304498269896

## Non-IIDの結果

### 2023-09-18-15(一番最初のやつ)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- log1.log
- the average of the last 10 epoch: 0.7619088811995386
- the std of the last 10 epoch: 0.03797441048346648
- the maxmize of the last 10 epoch: 0.8304498269896193
- the minimum of the last 10 epoch: 0.6816608996539792

### 2023-09-18-16(lrの減少を早く)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- log2.log
- the average of the last 10 epoch: 0.7658304498269896
- the std of the last 10 epoch: 0.02710190220965939
- the maxmize of the last 10 epoch: 0.8027681660899654
- the minimum of the last 10 epoch: 0.6920415224913494
->採用

### 2023-09-19-11(ノードの出会いを早く+lrの減少を早く)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- the average of the last 10 epoch: 0.7811130334486736
- the std of the last 10 epoch: 0.022525415041126918
- the maxmize of the last 10 epoch: 0.8269896193771626
- the minimum of the last 10 epoch: 0.726643598615917

### 2023-09-19-10(lrの減少を早く+過去の記憶使用)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- use_previous_memory
- the average of the last 10 epoch: 0.7807093425605536
- the std of the last 10 epoch: 0.026180020410580426
- the maxmize of the last 10 epoch: 0.8304498269896193
- the minimum of the last 10 epoch: 0.7370242214532872

### 2023-09-19-21(ノードの出会いを早く+lrの減少早く+過去の記憶使用)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- use_previous_memory
- the average of the last 10 epoch: 0.7814013840830449
- the std of the last 10 epoch: 0.025799082966628365
- the maxmize of the last 10 epoch: 0.8304498269896193
- the minimum of the last 10 epoch: 0.740484429065744

ここまでが中間前。これ以降は、lrの減少を早く+ノードの交換の速度をupを基本に考える。

### 2023-10-23-22(cos類似度を使用)

- vit
- epoch: 150->3000
- fl_coefficiency: cos類似度を使用
- 過去パラメータなし
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- the average of the last 10 epoch: 0.7756920415224914
- the std of the last 10 epoch: 0.024465974776845856
- the maxmize of the last 10 epoch: 0.8166089965397924
- the minimum of the last 10 epoch: 0.726643598615917

### どれだけ過去パラが使われているのかの検証ように10-25-10と10-25-11を回す

- p40: [2798, 2783, 2796, 2785, 2789, 2772, 2782, 2769, 2791, 2777, 2761, 2798]
- p10: [2683, 2699, 2713, 2672, 2695, 2713, 2663, 2680, 2692, 2669, 2710, 2733]
