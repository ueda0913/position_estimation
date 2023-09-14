# WAFL_position_estimationのメモ

## IIDの結果

### 2023-09-14-07

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"

### 2023-09-14-15

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"

## Non-IIDの結果

### 2023-09-14-08

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1

### 2023-09-14-18

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
