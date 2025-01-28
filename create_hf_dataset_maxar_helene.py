import os

import cv2
import numpy as np
import rasterio
from huggingface_hub import login

from datasets import Features, Array3D, Dataset, DatasetDict, Array2D, concatenate_datasets, Image, DatasetInfo
from datasets.features.image import objects_to_list_of_image_dicts

ds_path = "datasets/maxtar/helene"
target_path = "datasets/maxtar/helene/tiles"
image_size = 1500
hf_token = "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"
np.random.seed(42)


def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        # Read all bands
        array = src.read()

        # Transpose from (bands, height, width) to (height, width, bands)
        array = np.transpose(array, (1, 2, 0))


        # Normalize image to [0,1] range
        #image = image / 255.0
        #image = np.clip(image, 0, 1)

        # Binarize mask
        #mask = (mask > 0).astype(np.float32)

        return array


def is_more_than_10_percent_black(image):
    # Create a boolean mask where pixels are black ([0, 0, 0])
    black_pixels = np.all(image == [0, 0, 0], axis=-1)

    # Count the number of black pixels
    num_black_pixels = np.sum(black_pixels)

    # Calculate the total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of black pixels
    black_percentage = (num_black_pixels / total_pixels) * 100

    # Check if more than 10% of the pixels are black
    return black_percentage > 10

def split_maxar(subset):
    all_files = [name for name in os.listdir(f"{ds_path}/{subset}") if name.endswith(".tif")]

    for file in all_files:
        (name, ext) = file.split(".")
        image = load_geotiff(f"{ds_path}/{subset}/{file}")
        height, width = image.shape[:2]
        new_width = width // 2
        new_height = height // 2
        image = cv2.resize(image, (new_width, new_height))

        for j in range(0, image.shape[0], 512):
            j_str = str(j//512)
            if len(j_str) == 1:
                j_str = f"0{j_str}"
            for i in range(0, image.shape[1], 512):
                i_str = str(i//512)
                if len(i_str) == 1:
                    i_str = f"0{i_str}"
                tile = image[j:j+512, i:i+512]
                if is_more_than_10_percent_black(tile):
                    continue

                parts = name.split("_")
                quadkey = parts[0]
                date = parts[1]
                file_id = parts[2]

                key_path = f"{ds_path}/quadkey/{quadkey}/{subset}"
                os.makedirs(key_path, exist_ok=True)

                save_path = f"{key_path}/grid_{j_str}_{i_str}_{date}_{file_id}.png"
                cv2.imwrite(save_path, tile)


def load_pre_split():
    data = {}
    for dir in [name for name in os.listdir(ds_path)]:
        split = "validation"
        if "train" in dir:
            split = "train"
        elif "test" in dir:
            split = "test"

        images = {}
        labels = {}
        d = labels if "_labels" in dir else images

        for f in os.listdir(f"{ds_path}/{dir}"):
            (name, ext) = f.split(".")
            image = cv2.imread(f"{ds_path}/{dir}/{f}", cv2.IMREAD_UNCHANGED)
            for i in range(0, 1500, 500):
                d[f"{name}_{i/500*3}"] = image[i:i+500, :500]
                d[f"{name}_{i/500*3+1}"] = image[i:i+500, 500:1000]
                d[f"{name}_{i/500*3+2}"] = image[i:i+500, 1000:1500]

        data.setdefault(split, {})["label" if "_labels" in dir else "image"] = d
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

split_maxar("pre")
split_maxar("post")
data_dict = {}
batch_size = 250
data = load_maxar()
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


# 4. Log in to Hugging Face
login(token=hf_token)

# 5. Push the Dataset to the Hugging Face Hub
repo_id = "asolodin/orthoimagery_twin_cities_pos"  # Replace with your repo ID
os.makedirs(f"datasets/{repo_id}", exist_ok=True)
dataset_dict.save_to_disk(f"datasets/{repo_id}")

dataset_dict.push_to_hub(repo_id)

