# previous model aggregation

## base conditions

- vit
- epoch: 150->3000
- SGD(lr=0.05,momentum = 0.9,pretrain_lr = 0.05,pretrain_momentum = 0.9)
- scheduler(scheduler_step = 750,scheduler_rate = 0.3,pretrain_scheduler_step = 50,pretrain_scheduler_rate = 0.3)

## fl_coefficiencyの差異について(本題ではない、あまりに変わる場合のみ)

- p10かつnon-iidで、fl = 0.2にした場合の検証を2023-11-26-16(nohup.out)で行う

## non-iid
