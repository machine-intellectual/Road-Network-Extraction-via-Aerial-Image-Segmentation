import os

import cv2
import shutil

input_ds_path = "datasets/maxtar/helene/tiles"
subsets = ["pre", "post"]

def index_files_by_quadkey(files):
    files_dict = {}
    for file in files:
        (name, ext) = file.split(".")
        parts = name.split("_")
        quadkey = parts[0]
        date = parts[1]
        file_id = parts[2]
        (y, x) = (parts[-2], parts[-1])
        files_dict.setdefault(quadkey, []).append(file)
    return files_dict

for subset in subsets:
    all_files = [name for name in os.listdir(f"{input_ds_path}/{subset}") if name.endswith(".png")]
    files_index = index_files_by_quadkey(all_files)
    for key, files in files_index.items():
        key_path = f"datasets/maxtar/helene/quadkey/{key}/{subset}"
        os.makedirs(key_path, exist_ok=True)

        for file in files:
            (name, ext) = file.split(".")
            parts = name.split("_")
            quadkey = parts[0]
            date = parts[1]
            file_id = parts[2]
            (y, x) = (parts[-2], parts[-1])

            save_path = f"{key_path}/grid_{y}_{x}_{date}_{file_id}.{ext}"
            shutil.copy(f"{input_ds_path}/{subset}/{file}", save_path)
