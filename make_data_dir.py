import os
import random
import shutil

dir_name = "position_estimation_dataset"

# forループで一括処理するためのlist定義
build_list = [str(i) for i in range(0, 12)]
print(build_list)
validation_pct = 0.2  # どれだけ検証用に使うか
# dataset用のdirectoryの作成
"""
ディレクトリ構造
dataset
|-train
|  |-0
|  |...
|  |-9
|
|-val
   |-0
   |...
   |-9
"""

"""
raw data structure
data/raw_data
|-dir1
|  |-label1
|  | 
|  | 
|  |...
|-dir2
|  |-label1
|  |
|...

"""
os.makedirs(
    os.path.join(os.path.expanduser("~"), "data-raid/data", dir_name)
)  # homeディレクトリの指定はos.path.expanduser('~')
dataset_dir = os.path.join(os.path.expanduser("~"), "data-raid/data", dir_name)
print(dataset_dir)
os.makedirs(os.path.join(dataset_dir, "val"))
os.makedirs(os.path.join(dataset_dir, "train"))
for build_num in build_list:  # 0~11のディレクトリを作成
    os.makedirs(os.path.join(dataset_dir, "train", build_num))
    os.makedirs(os.path.join(dataset_dir, "val", build_num))

raw_data_container_path = os.path.join(
    os.path.expanduser("~"), "data-raid/data/raw_data"
)
files_and_dirs_in_container = os.listdir(raw_data_container_path)
raw_directories = [
    d
    for d in files_and_dirs_in_container
    if os.path.isdir(os.path.join(raw_data_container_path, d))
]  # raw data directory list
data_num = [0 for i in range(len(build_list))]

for raw_dir in raw_directories:
    for build_num in build_list:
        if not (
            build_num
            in os.listdir(
                os.path.join(
                    os.path.expanduser("~"), "data-raid/data/raw_data", raw_dir
                )
            )
        ):
            continue
        image_files = []
        image_dir = ""
        image_dir = os.path.join(
            os.path.expanduser("~"),
            "data-raid/data/raw_data",
            raw_dir,
            build_num,
        )
        num_files = len(
            [
                f
                for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
            ]
        )
        data_num[int(build_num)] += num_files
        print(image_dir)
        # imageをlistでもらう.
        # こうすることで、シャッフルのあとlistの1~nまでをvalに入れて...という処理で分割ができる.
        image_files = os.listdir(image_dir)

        random.shuffle(image_files)  # シャッフル!!!
        num_validation = int(len(image_files) * validation_pct)  # 検証用画像の枚数.
        # ファイルの移動. 1~num_validationまでをvalフォルダに入れる.
        for file in image_files[:num_validation]:
            src = os.path.join(image_dir, file)
            dst = os.path.join(dataset_dir, "val", build_num)
            shutil.move(src, dst)
        for file in image_files[num_validation:]:
            src = os.path.join(image_dir, file)
            dst = os.path.join(dataset_dir, "train", build_num)
            shutil.move(src, dst)

        os.rmdir(image_dir)
    os.rmdir(os.path.join(os.path.expanduser("~"), "data-raid/data/raw_data", raw_dir))

    print(data_num)
