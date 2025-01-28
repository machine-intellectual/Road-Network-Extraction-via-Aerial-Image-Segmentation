import os
import sys

import cv2
import numpy as np
import torch
from torch import nn, tensor
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

id2label = {1: 'road'}
label2id = {v: k for k, v in id2label.items()}

root_dir = "datasets/maxar/helene/quadkey"
checkpoint = "models/orthoimagery_twin_cities_pos_b3_ma_mixed"
image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b3", do_reduce_labels=False)
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU


def compute_mask(logits_tensor):
    # Convert logits to probabilities
    pred_labels = torch.sigmoid(logits_tensor)

    threshold = 0.5
    pred_labels[pred_labels > threshold] = 1
    pred_labels[pred_labels <= threshold] = 0

    pred_labels = pred_labels.detach().cpu().numpy()

    return pred_labels


def normalize_histogram(image):
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return output

def label_image(image):
    image = normalize_histogram(image)
    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=512,
        mode="bilinear",
        align_corners=False,
    ).squeeze(dim=1)
    prediction: tensor = compute_mask(upsampled_logits[0])
    return prediction


def get_directories_with_pre_post(parent_dirs):
    for parent_dir in parent_dirs:
        # Check if the parent_dir exists and is actually a directory
        if os.path.isdir(parent_dir):
            # Check if 'pre' and 'post' subdirectories exist within the parent directory
            pre_dir = os.path.join(parent_dir, 'pre')
            post_dir = os.path.join(parent_dir, 'post')

            if os.path.isdir(pre_dir) and os.path.isdir(post_dir):
                for dir in [pre_dir, post_dir]:
                    files = os.listdir(dir)
                    for file in files:
                        if "label" in file:
                            continue
                        file_path = os.path.join(dir, file)
                        label_path = file_path.replace("visual", "label")
                        label_path = label_path.replace(dir, f"{dir}/label1")
                        if os.path.isfile(label_path):
                            continue

                        try:
                            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                            if image is None:
                                print("What did I read ", file_path)
                                continue

                            label: np.ndarray = label_image(image)
                            label = label.astype(np.uint8)
                            label[label > 0] = 255
                            os.makedirs(f"{dir}/label1", exist_ok=True)
                            cv2.imwrite(label_path, label)
                            print(label_path)
                        except:
                            print(label_path, sys.exc_info()[1])



# Example usage

parent_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  # Replace with the actual list of directories

get_directories_with_pre_post(parent_dirs)
