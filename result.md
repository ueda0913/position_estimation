# WAFL_position_estimationのメモ

## IIDの結果

### 2023-09-14-15(最初の)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.8372837370242214
- the std of the last 10 epoch: 0.01707250711501161

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

### 2023-09-15-23(過去の記憶使用+普通の)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- use_previous_memory(連続変化)
- the average of the last 10 epoch: 0.8474625144175317
- the std of the last 10 epoch: 0.01623962493684984
- the maxmize of the last 10 epoch: 0.8685121107266436
- the minimum of the last 10 epoch: 0.7508650519031141

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

## Non-IIDの結果

### 2023-09-14-08

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- the average of the last 10 epoch: 0.5981833910034602
- the std of the last 10 epoch: 0.11724919125144062
- ミス発覚により無効

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
