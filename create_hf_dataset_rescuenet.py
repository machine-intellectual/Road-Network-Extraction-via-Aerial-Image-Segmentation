import os

import cv2
import numpy as np
import rasterio
from huggingface_hub import login

from datasets import Features, Array3D, Dataset, DatasetDict, Array2D, concatenate_datasets, Image, DatasetInfo
from datasets.features.image import objects_to_list_of_image_dicts

ds_path = "datasets/rescuenet"
image_size = 1500
hf_token = "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"


def load_pre_split():
    data = {}
    empty = 0
    clear = 0
    blocked = 0
    for dir in [name for name in os.listdir(ds_path)]:
        if not os.path.isdir(os.path.join(ds_path, dir)):
            continue

        split = "validation"
        if "train" in dir:
            split = "train"
        elif "test" in dir:
            split = "test"

        images = {}
        labels = {}
        d = labels if "label" in dir else images

        for f in os.listdir(f"{ds_path}/{dir}"):
            (name, ext) = f.split(".")
            if "_" in name:
                name = name.split("_")[0]
            image = cv2.imread(f"{ds_path}/{dir}/{f}", cv2.IMREAD_UNCHANGED)

            if "label" in dir:
                image[~np.isin(image, [7, 8])] = 0
                if not np.any(image):
                    empty = empty + 1
                    if split in data and "image" in data[split] and name in data[split]["image"]:
                        del data[split]["image"][name]
                    continue
                if np.any(image == 7):
                    clear = clear + 1
                if np.any(image == 8):
                    blocked = blocked + 1

            elif split in data and "label" in data[split] and name not in data[split]["label"]:
                continue

            s_m = np.argmin(image.shape[:1])
            if s_m == 0:
                new_size = (int(image.shape[1] * (512 / image.shape[0])), 512)
            else:
                new_size = (512, int(image.shape[0] * (512 / image.shape[1])))

            resized_down = cv2.resize(image, new_size, interpolation=(cv2.INTER_NEAREST_EXACT if "label" in dir else cv2.INTER_LINEAR))

            if s_m == 0:
                off = (resized_down.shape[1] - 512) // 2
                resized_down = resized_down[:, off:-off]
            else:
                off = (resized_down.shape[0] - 512) // 2
                resized_down = resized_down[off:-off, :]


            d[name] = resized_down

        data.setdefault(split, {})["label" if "label" in dir else "image"] = d
        print(f"{split} has {len(d)} {dir}, clear {clear}, blocked {blocked}")
    return data

features = Features({
    # "image": Array3D(dtype="uint8", shape=(image_size, image_size, 3)),  # Adjust shape to match your images
    # "label": Array2D(dtype="uint8", shape=(image_size, image_size))   # Same shape for segmentation masks
    "image": Image(),  # Adjust shape to match your images
    "label": Image()  # Same shape for segmentation masks
})
info = DatasetInfo(description="Digital Orthoimagery, Twin Cities, Minnesota, Spring 2016, 1-ft Resolution "
                   ,homepage="https://gisdata.mn.gov/dataset/us-mn-state-metc-base-metro-orthos-2016",
                   features=features
                   )

save = False
data_dict = {}
batch_size = 250
data = load_pre_split()
for split in data:
    images_list = []
    labels_list = []

    for k in data[split]["image"]:
        images_list.append(data[split]["image"][k])
        labels_list.append(data[split]["label"][k])

    # 2. Combine images and labels dictionaries
    datasets = []
    for i in range(0, len(images_list), batch_size):

        images = {"image": objects_to_list_of_image_dicts(images_list[i:i + batch_size])}
        labels = {"label": objects_to_list_of_image_dicts(labels_list[i:i + batch_size])}
        ds = Dataset.from_dict({**images, **labels}, info=info)
        datasets.append(ds)

    data_dict[split] = concatenate_datasets(datasets)

# If you have a train/test tiles, create a DatasetDict
dataset_dict = DatasetDict(data_dict)  # Add "test": test_dataset if applicable

if save:
    # 4. Log in to Hugging Face
    login(token=hf_token)

    # 5. Push the Dataset to the Hugging Face Hub
    repo_id = "asolodin/rescuenet-roads"  # Replace with your repo ID
    os.makedirs(f"datasets/{repo_id}", exist_ok=True)
    dataset_dict.save_to_disk(f"datasets/{repo_id}")

    dataset_dict.push_to_hub(repo_id)

