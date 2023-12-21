import copy
import math

import torch
import torch.nn as nn
from definitions.net import Net_vit_b16


def calc_cos_similarity(net1, net2):  # 入力は学習しているパラメータを取り出したstate_dict
    net1_params = net1.values()
    net1_param_list = [param for param in net1_params]
    net1_vector = torch.cat([param.view(-1) for param in net1_param_list], dim=0)

    net2_params = net2.values()
    net2_param_list = [param for param in net2_params]
    net2_vector = torch.cat([param.view(-1) for param in net2_param_list], dim=0)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos(net1_vector, net2_vector)
    return cos_similarity


def update_nets_vgg(nets, contact, fl_coefficiency):  # sはn番目の要素に複数のmodelがあるかで決まる
    n_node = len(contact)
    local_model = [{} for i in range(n_node)]
    recv_models = [[] for i in range(n_node)]
    for n in range(n_node):
        local_model[n] = nets[n].classifier[6].state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(nets[k].classifier[6].state_dict())

    # mixture of models
    for n in range(n_node):
        update_model = recv_models[n]
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                if local_model[n][key].dtype is torch.float32:
                    local_model[n][key] += (
                        update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                    )
                elif local_model[n][key].dtype is torch.int64:
                    pass
                else:
                    print(
                        key,
                        type(fl_coefficiency),
                        type(n_nbr),
                        local_model[n][key].dtype,
                        update_model[k][key].dtype,
                    )
                    exit(1)
    # update nets
    for n in range(n_node):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            nets[n].classifier[6].load_state_dict(local_model[n])


def update_nets_res(
    nets,
    contact,
    fl_coefficiency,
):  # sはn番目の要素に複数のmodelがあるかで決まる
    n_node = len(contact)
    local_model = [{} for i in range(n_node)]
    recv_models = [[] for i in range(n_node)]
    for n in range(n_node):
        local_model[n] = nets[n].fc.state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(nets[k].fc.state_dict())

    # mixture of models
    for n in range(n_node):
        update_model = recv_models[n]
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                if local_model[n][key].dtype is torch.float32:
                    local_model[n][key] += (
                        update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                    )
                elif local_model[n][key].dtype is torch.int64:
                    pass
                else:
                    print(
                        key,
                        type(fl_coefficiency),
                        type(n_nbr),
                        local_model[n][key].dtype,
                        update_model[k][key].dtype,
                    )
                    exit(1)
    # update nets
    for n in range(n_node):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            nets[n].fc.load_state_dict(local_model[n])


def update_nets_vit(
    nets, contact, use_cos_similarity, st_fl_coefficiency, epoch, sat_epoch
):  # sはn番目の要素に複数のmodelがあるかで決まる
    n_node = len(contact)
    local_model = [{} for i in range(n_node)]
    recv_models = [[] for i in range(n_node)]
    # update_models = [[] for _ in range(n_node)]
    for n in range(n_node):
        local_model[n] = nets[n].heads.state_dict()
    for n in range(n_node):
        nbr = contact[str(n)]  # the nodes n-th node contacted
        for k in nbr:
            recv_models[n].append(copy.deepcopy(local_model[k]))

    # mixture of models
    for n in range(n_node):
        update_model = [
            copy.deepcopy(local_model[0]) for _ in range(len(contact[str(n)]))
        ]
        # update_model = copy.deepcopy(recv_models[n])
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            if use_cos_similarity:
                # TODOこの与え方とかも考えないと不味そう
                # とりあえず最初はcos類似度に対し線形に変化し、その変化が徐々に小さくなるようにしていく
                cos_similarity = calc_cos_similarity(local_model[n], recv_models[n][k])
                epoch_rate = float(sat_epoch - epoch) / sat_epoch
                delta_fl = epoch_rate * st_fl_coefficiency
                fl_coefficiency = delta_fl * cos_similarity + (
                    st_fl_coefficiency - delta_fl
                )
                # fl_coefficiency = st_fl_coefficiency * (cos_similarity + 1) / 2
                if epoch % 500 == 499:
                    print(
                        f"cos_similarity between node-{n} and node-{contact[str(n)][k]} in epoch{epoch}: {cos_similarity}\n"
                    )
            else:
                fl_coefficiency = st_fl_coefficiency

            for key in update_model[k]:
                if local_model[n][key].dtype is torch.float32:
                    local_model[n][key] += (
                        update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                    )
                elif local_model[n][key].dtype is torch.int64:
                    pass
                else:
                    print(
                        key,
                        type(fl_coefficiency),
                        type(n_nbr),
                        local_model[n][key].dtype,
                        update_model[k][key].dtype,
                    )
                    exit(1)
    # update nets
    for n in range(n_node):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            nets[n].heads.load_state_dict(local_model[n])


def update_nets_mobile(
    nets,
    contact,
    fl_coefficiency,
):  # sはn番目の要素に複数のmodelがあるかで決まる
    n_node = len(contact)
    local_model = [{} for i in range(n_node)]
    recv_models = [[] for i in range(n_node)]
    for n in range(n_node):
        local_model[n] = nets[n].classifier[1].state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(nets[k].classifier[1].state_dict())

    # mixture of models
    for n in range(n_node):
        update_model = recv_models[n]
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                if local_model[n][key].dtype is torch.float32:
                    local_model[n][key] += (
                        update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                    )
                elif local_model[n][key].dtype is torch.int64:
                    pass
                else:
                    print(
                        key,
                        type(fl_coefficiency),
                        type(n_nbr),
                        local_model[n][key].dtype,
                        update_model[k][key].dtype,
                    )
                    exit(1)
    # update nets
    for n in range(n_node):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            nets[n].classifier[1].load_state_dict(local_model[n])


def model_exchange(
    nets, model_name, contact, use_cos_similarity, st_fl_coefficiency, epoch, sat_epoch
):  # cos similarity is now only for vit
    if model_name == "vgg19_bn":
        update_nets_vgg(nets, contact, st_fl_coefficiency)
    elif model_name == "resnet_152":
        update_nets_res(nets, contact, st_fl_coefficiency)
    elif model_name == "vit_b16":
        update_nets_vit(
            nets, contact, use_cos_similarity, st_fl_coefficiency, epoch, sat_epoch
        )
    elif model_name == "mobilenet_v2":
        update_nets_mobile(nets, contact, st_fl_coefficiency)


def model_exchange_with_former(
    former_contact,
    contact,
    former_nets,
    nets,
    counters,
    former_exchange_num,
    avg_former_exchange_num,
    model_name,
    use_cos_similarity_previous_memory,
    st_fl_coefficiency_pm,
):
    # countersはcontact_patternの変更から何epochだけ経過したか
    # former_netsにはcontact_patternが変化した時の変化前のnetsが入る
    for n in range(len(contact)):
        if model_name == "vit_b16":
            current_net = nets[n].heads.state_dict()
        elif model_name == "vgg19_bn":
            current_net = nets[n].classifier[6].state_dict()
        elif model_name == "resnet_152":
            current_net = nets[n].fc.state_dict()
        elif model_name == "mobilenet_v2":
            current_net = nets[n].classifier[1].state_dict()

        if former_contact[str(n)] != contact[str(n)]:
            former_contact[str(n)] = contact[str(n)]
            former_nets[n] = copy.deepcopy(current_net)
            avg_former_exchange_num[n] += counters[n]
            counters[n] = 0
            former_exchange_num[n] += 1

        # 連続交換回数による変動
        if counters[n] >= 10:  # この閾値と重み0.1は変更の余地あり
            ratio = st_fl_coefficiency_pm
        # elif counters[n] <= 5:
        #     ratio = 0
        else:
            d_x = (
                math.log(st_fl_coefficiency_pm * 2) - math.log(st_fl_coefficiency_pm)
            ) / 10
            x = math.log(st_fl_coefficiency_pm) + d_x * counters[n]
            ratio = math.pow(math.e, x) - st_fl_coefficiency_pm

        # 類似度による変動
        if use_cos_similarity_previous_memory:
            cos_sim_pm = calc_cos_similarity(current_net, former_nets[n])
            if cos_sim_pm > 0.8:
                ratio = 0

        for key in former_nets[n]:
            current_net[key] = (
                current_net[key] * (1 - ratio) + former_nets[n][key] * ratio
            )
        nets[n].heads.load_state_dict(current_net)

        counters[n] += 1
