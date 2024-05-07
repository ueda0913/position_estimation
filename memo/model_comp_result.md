# WAFLのモデル間比較の結果まとめ

## rwp

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
- the average of the last 10 epoch: 0.7770761245674741
- the std of the last 10 epoch: 0.020353366417513313

### resnet_wafl_raw_iid

- resnet
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.7577566320645905
- the std of the last 10 epoch: 0.034586067749729034

### vgg_wafl_raw_iid

- vgg
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "rwp_n12_a0500_r100_p40_s01.json"
- the average of the last 10 epoch: 0.7150230680507497
- the std of the last 10 epoch: 0.025544857099590403

## ringstar

### vit_wafl_raw_iid_ringstar

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_ringstar_n12.json"
- the average of the last 10 epoch: 0.8596597462514418
- the std of the last 10 epoch: 0.012725842208943463
- the maxmize of the last 10 epoch: 0.889273356401384
- the minimum of the last 10 epoch: 0.8096885813148789

### mobile_wafl_raw_iid_ringstar

- mobile
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_ringstar_n12.json"
- the average of the last 10 epoch: 0.7787773933102653
- the std of the last 10 epoch: 0.023731458741558516
- the maxmize of the last 10 epoch: 0.8235294117647058
- the minimum of the last 10 epoch: 0.6920415224913494

### resnet_wafl_raw_iid_ringstar

- resnet
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_ringstar_n12.json"
- the average of the last 10 epoch: 0.7677912341407152
- the std of the last 10 epoch: 0.027786127185047027
- the maxmize of the last 10 epoch: 0.8200692041522492
- the minimum of the last 10 epoch: 0.6782006920415224

### vgg_wafl_raw_iid_ringstar

- vgg
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_ringstar_n12.json"
- the average of the last 10 epoch: 0.7031430219146482
- the std of the last 10 epoch: 0.02683227649944828

## line

### vit_wafl_raw_iid_line

- vit
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_line_n12.json"
- the average of the last 10 epoch: 0.8364186851211073
- the std of the last 10 epoch: 0.012398801152536887
- the maxmize of the last 10 epoch: 0.8615916955017301
- the minimum of the last 10 epoch: 0.7923875432525952

### mobile_wafl_raw_iid_line

- mobile
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_line_n12.json"
- the average of the last 10 epoch: 0.7711937716262975
- the std of the last 10 epoch: 0.025290595300355507
- the maxmize of the last 10 epoch: 0.8235294117647058
- the minimum of the last 10 epoch: 0.698961937716263

### resnet_wafl_raw_iid_line

- resnet
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_line_n12.json"

### vgg_wafl_raw_iid_line

- vgg
- epoch: 150->3000
- fl_coefficiency = 0.1
- SGD(lr=0.005,momentum = 0.9,pretrain_lr = 0.005,pretrain_momentum = 0.9)
- schedulerなし
- contact_file = "static_line_n12.json"
- the average of the last 10 epoch: 0.702681660899654
- the std of the last 10 epoch: 0.028264923596512382
- the maxmize of the last 10 epoch: 0.7612456747404844
- the minimum of the last 10 epoch: 0.615916955017301
