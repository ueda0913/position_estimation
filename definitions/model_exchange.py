import torch
import torch.nn as nn


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
    nets,
    contact,
    use_cos_similarity,
    fl_coefficiency,
):  # sはn番目の要素に複数のmodelがあるかで決まる
    n_node = len(contact)
    local_model = [{} for i in range(n_node)]
    recv_models = [[] for i in range(n_node)]
    for n in range(n_node):
        local_model[n] = nets[n].heads.state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(nets[k].heads.state_dict())

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
            if use_cos_similarity:
                # TODOこの与え方とかも考えないと不味そう
                cos_similarity = calc_cos_similarity(
                    local_model[n],
                )
                fl_coefficiency = 0.5 * cos_similarity
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
    nets, model_name, contact, use_cos_similarity, st_fl_coefficiency
):  # cos similarity is now only for vit
    if model_name == "vgg19_bn":
        update_nets_vgg(nets, contact, st_fl_coefficiency)
    elif model_name == "resnet_152":
        update_nets_res(nets, contact, st_fl_coefficiency)
    elif model_name == "vit_b16":
        update_nets_vit(nets, contact, use_cos_similarity, st_fl_coefficiency)
    elif model_name == "mobilenet_v2":
        update_nets_mobile(nets, contact, st_fl_coefficiency)


def model_exchange_with_former_vit(
    former_contact, contact, former_nets, nets, counters, former_exchange_num
):
    # countersはcontact_patternの変更から何epochだけ経過したか
    # former_netsにはcontact_patternが変化した時の変化前のnetsが入る
    for n in range(len(contact)):
        if former_contact[str(n)] != contact[str(n)]:
            former_contact[str(n)] = contact[str(n)]
            former_nets[n] = nets[n].heads.state_dict()
            counters[n] = 0

        net_param = nets[n].heads.state_dict()
        ratio = 0.01 * counters[n]
        if counters[n] >= 10:  # この閾値と重み0.1は変更の余地あり
            ratio = 0.1
        for key in former_nets[n]:
            net_param[key] = net_param[key] * (1 - ratio) + former_nets[n][key] * ratio
        nets[n].heads.load_state_dict(net_param)
        former_exchange_num[n] += 1

        counters[n] += 1
