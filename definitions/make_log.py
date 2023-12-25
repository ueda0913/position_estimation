import os

from torchinfo import summary


def initial_log(
    cur_dir,
    subset,
    batch_size,
    contact_file,
    is_use_noniid_filter,
    filter_file,
    testloader,
    trainloader,
    optimizers,
    use_scheduler,
    schedulers,
    pretrain_optimizers,
    pretrain_schedulers,
    use_pretrain_scheduler,
    use_previous_memory,
    use_cos_similarity,
    st_fl_coefficiency,
    sat_epoch,
    use_cos_similarity_previous_memory,
    st_fl_coefficiency_pm,
    is_pre_train_only,
    nets,
):
    with open(os.path.join(cur_dir, "log.txt"), "a") as f:
        for i in range(len(subset)):
            f.write(f"the number of data for training {i}-th node: {len(subset[i])}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"contact file: {contact_file}\n")
        if is_use_noniid_filter:
            f.write(f"filter file: {filter_file}\n")
        else:
            f.write("do not use fileter\n")
        f.write(f"test transform: {testloader.dataset.transform}\n")
        f.write(f"train transform: {trainloader[0].dataset.transform}\n")
        f.write(f"optimizer: {optimizers[0]}\n")
        if use_scheduler:
            f.write(f"training sheduler step: {schedulers[0].step_size}\n")
            f.write(f"training sheduler gamma: {schedulers[0].gamma}\n")
        f.write(f"pre-training optimizer: {pretrain_optimizers[0]}\n")
        if use_pretrain_scheduler:
            f.write(f"pre-training sheduler step: {pretrain_schedulers[0].step_size}\n")
            f.write(f"pre-training sheduler gamma: {pretrain_schedulers[0].gamma}\n")
        f.write(f"fl_coefficiency: {st_fl_coefficiency}\n")
        f.write(f"use previous memory: {use_previous_memory}\n")
        if use_previous_memory:
            f.write(f"fl_coefficiency_pm: {st_fl_coefficiency_pm}\n")
        f.write(f"use cosine similarity: {use_cos_similarity}\n")
        if use_cos_similarity:
            f.write(f"sat_epoch: {sat_epoch}\n")
        f.write(f"use cosine similarity_pm: {use_cos_similarity_previous_memory}\n")
        if use_cos_similarity_previous_memory:
            f.write(f"sat_epoch_pm: {sat_epoch}\n")
        f.write(f"pre_train_only: {is_pre_train_only}\n")
        f.write(f"net:\n {summary(nets[0], (1,3,224,224), verbose=False)}\n")


def initial_log_for_tuning(
    cur_dir,
    subset,
    batch_size,
    contact_file,
    is_use_noniid_filter,
    filter_file,
    testloader,
    trainloader,
    optimizer_name,
    use_scheduler,
    schedulers,
    pretrain_schedulers,
    use_pretrain_scheduler,
    use_previous_memory,
    use_cos_similarity,
    model_name,
    load_dir_name,
):
    with open(os.path.join(cur_dir, "log.txt"), "a") as f:
        for i in range(len(subset)):
            f.write(f"the number of data for training {i}-th node: {len(subset[i])}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"contact file: {contact_file}\n")
        if is_use_noniid_filter:
            f.write(f"filter file: {filter_file}\n")
        else:
            f.write("do not use fileter\n")
        f.write(f"test transform: {testloader.dataset.transform}\n")
        f.write(f"train transform: {trainloader[0].dataset.transform}\n")
        f.write(f"optimizer: {optimizer_name}\n")
        if use_scheduler:
            f.write(f"training sheduler step: {schedulers[0].step_size}\n")
            f.write(f"training sheduler gamma: {schedulers[0].gamma}\n")
        if use_pretrain_scheduler:
            f.write(f"pre-training sheduler step: {pretrain_schedulers[0].step_size}\n")
            f.write(f"pre-training sheduler gamma: {pretrain_schedulers[0].gamma}\n")
        f.write(f"use previous memory: {use_previous_memory}\n")
        f.write(f"use cosine similarity: {use_cos_similarity}\n")
        f.write(f"net: {model_name}\n")
        f.write(f"Loaded pre-trained data is in {load_dir_name}\n.")
