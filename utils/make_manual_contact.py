import math
import random

import numpy as np

n_time = 10000

n_node = 12
min_travel_speed = 3
max_travel_speed = 7
radio_range = 50

# pose_time = 10
# randomseed = 1

## meet all node at once
contact_list = []
for t in range(n_time):
    node_in_contact = {
        i: [] for i in range(n_node)
    }  # ある時刻においてノードiと通信可能なすべてのノードの番号をリストの中に入れている
    for i in range(n_node):
        node_in_contact[i] = []
        for j in range(n_node):
            if i != j:
                node_in_contact[i].append(j)
    # print(f't={t} : contacts={node_in_contact}')
    contact_list.append(node_in_contact)


import json

with open(
    f"../../data-raid/static/contact_pattern/meet_at_once_t{n_time:03d}.json", "w"
) as f:
    json.dump(contact_list, f, indent=4)
