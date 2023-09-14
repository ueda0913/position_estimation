import os
import pickle

from definitions.visualize import *

project_path = "../data-raid/static/WAFL_pos_estimation"

# change area
load_time_index = "2023-09-14-07"
# filename = "pre-train-only"


cur_dir = os.path.join(project_path, load_time_index)
with open(
    os.path.join(project_path, load_time_index, "params", "historys_data.pkl"),
    "rb",
) as f:
    historys = pickle.load(f)

evaluate_history(historys, cur_dir)
