# WAFL_position_estimationのメモ

## IIDの結果

### vit_wafl_raw_iid

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.8347462514417532
- the std of the last 10 epoch: 0.027045538501666733

### mobile_wafl_raw_iid

- mobile
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch:
- the std of the last 10 epoch:

### resnet_wafl_raw_iid

- resnet
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch:
- the std of the last 10 epoch:

### 2023-09-14-15

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.8372837370242214
- the std of the last 10 epoch: 0.01707250711501161

### 2023-09-15-07

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- use_previous_memory
- -> 死んだので後回し

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

### 2023-09-14-18

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 1000,scheduler_rate = 0.5,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- filter:filter_rate = 70,filter_seed = 1
- the average of the last 10 epoch: 0.6782006920415224
- the std of the last 10 epoch: 0.09056063335043048

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
