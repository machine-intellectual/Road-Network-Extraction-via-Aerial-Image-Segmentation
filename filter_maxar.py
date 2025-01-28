import os
import shutil

import cv2
import numpy as np
import torch
from torch import nn, tensor
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

id2label = {1: 'road'}
label2id = {v: k for k, v in id2label.items()}

root_dir = "datasets/maxtar/helene/quadkey"
target_dir = "datasets/maxar/helene/quadkey"

stats = {}

def get_directories_with_pre_post(parent_dirs):
    for parent_dir in parent_dirs:
        # Check if the parent_dir exists and is actually a directory
        if os.path.isdir(parent_dir):
            # Check if 'pre' and 'post' subdirectories exist within the parent directory
            pre_dir = os.path.join(parent_dir, 'pre')
            post_dir = os.path.join(parent_dir, 'post')

            if os.path.isdir(pre_dir) and os.path.isdir(post_dir):
                total = 0
                num_images = 0
                for dir in [f"{pre_dir}/label", f"{post_dir}/label"]:
                    files = os.listdir(dir)
                    for file in files:
                        file_path = os.path.join(dir, file)
                        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        total = total + np.sum(image)
                        num_images = num_images + 1
                stats[parent_dir] = total / num_images

# Example usage

parent_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]

get_directories_with_pre_post(parent_dirs)
all_dirs = list(stats.keys())
all_dirs.sort(key=lambda x: -stats[x])
print(all_dirs[:10])

for d in all_dirs[:20]:
    shutil.copytree(d, d.replace("maxtar", "maxar"), dirs_exist_ok=True)