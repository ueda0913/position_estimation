# WAFL_position_estimationのメモ

## IIDの結果

### 2023-09-14-15(lrを大きく)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.8372837370242214
- the std of the last 10 epoch: 0.01707250711501161

### 2023-09-15-23(過去の記憶使用)

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

### 2023-09-18-15

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- nohup.out
- log1.log

### 2023-09-18-16(lrの減少を早く)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- nohup2.out
- log2.log

### 2023-09-1(ノードの出会いを早く)

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p10_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- 延期

### 2023-09-15-00

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- use_previous_memory
- the average of the last 10 epoch: 0.7053921568627451
- the std of the last 10 epoch: 0.05903318991873559
- 無効
