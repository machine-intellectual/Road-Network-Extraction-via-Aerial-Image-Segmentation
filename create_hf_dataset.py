import copy
import os

import cv2
import numpy as np
import rasterio
from huggingface_hub import login
from matplotlib import pyplot as plt

from datasets import Features, Array3D, Dataset, DatasetDict, Array2D, concatenate_datasets, Image, DatasetInfo
from datasets.features.image import objects_to_list_of_image_dicts

ds_path = "datasets/Hibbing_Aerial_Roads"
target_path = "datasets/hibbing"
image_size = 1500
hf_token = "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"


def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        # Read all bands
        array = src.read()

        # Get nodata value
        nodata = src.nodata or 32767  # Use 32767 as default nodata value if none specified

        # Convert to float32 for processing
        #array = array.astype(np.float32)

        # Handle nodata values
        array[array == nodata] = 0

        # Transpose from (bands, height, width) to (height, width, bands)
        array = np.transpose(array, (1, 2, 0))

        # Split into image and mask
        image = array[:, :, :3]
        mask = array[:, :, 3:4]  # Keep the mask as (H,W,1)

        mask = np.squeeze(mask, axis=2)

        # Normalize image to [0,1] range
        #image = image / 255.0
        #image = np.clip(image, 0, 1)

        # Binarize mask
        #mask = (mask > 0).astype(np.float32)
        mask[mask == 255] = 0
        mask[mask == 1] = 255

        return image.astype(np.uint8), mask.astype(np.uint8)

def load_eli():
    data = {}
    all_files = [name for name in os.listdir(ds_path, ) if name.endswith(".TIF")]
    np.random.seed(42)
    #np.random.shuffle(all_files)
    all_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    train, val, test = np.split(np.array(all_files), [5000, 6000])
    train = all_files[:5000]
    data_dict = {"train": train}#, "validation": val, "test": test}

    for split, files in data_dict.items():
        empty = 0
        images = {}
        labels = {}
        for file in files:
            (name, ext) = file.split(".")
            try:
                i, m = load_geotiff(f"{ds_path}/{file}")
                if i.shape[0] != 512 or i.shape[1] != 512:
                    print("wrong shape ", file)
                    continue
                if np.all(i == 255):
                    #i = np.zeros_like(i)
                    continue
                if split != "test" and not np.any(m):
                    empty = empty + 1
                    if np.random.random() < 0.9:
                        continue

                images[name] = fix_nodata(f"{ds_path}/{file}", i, m)
                labels[name] = erode_mask(m)
            except:
                print("could not read ", file)
        print(f"Set {split} has {len(images)} images, {empty} empty")
        data.setdefault(split, {})["image"] = images
        data.setdefault(split, {})["label"] = labels
    return data

def erode_mask(m):
    if np.any(m):

        # Create a structuring element (kernel) for erosion
        kernel_size = 13 # Size of the kernel (adjust as needed)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply erosion to shrink the white areas
        shrunken_mask = cv2.erode(m, kernel, iterations=1)

        #fig, axes = plt.subplots(1, 2)
        #axes[0].imshow(m, cmap='gray', vmin=0, vmax=1)
        #axes[0].axis('off')
        #axes[1].imshow(shrunken_mask, cmap='gray', vmin=0, vmax=1)
        #axes[1].axis('off')
        #plt.show()  # Redraw the plot

        #input("Press Enter to continue...")
        return shrunken_mask
    return m

def fix_nodata(path, image, label):
    image_copy = copy.deepcopy(image)
    pixel_sum = np.sum(image_copy, axis=-1)
    white_pixels = pixel_sum > 250 * 3
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_pixels.astype(np.uint8))
    for i in range(1, num_labels):  # Start at 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 10000:
            # If the area of the connected component is larger than min_area, set these pixels to black
            image_copy[labels == i] = [0, 0, 0]
            if label is not None:
                label[labels == i] = 0
    if not np.any(image_copy):
        print("no image??? ", path)
    return image_copy

def load_pre_split():
    data = {}
    for dir in ["test_labels", "train_labels", "val_labels", "test", "train", "val"]:
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
                if "_labels" in dir:
                    d[f"{name}_{i / 500 * 3}"] = image[i:i + 500, :500]
                    d[f"{name}_{i / 500 * 3 + 1}"] = image[i:i + 500, 500:1000]
                    d[f"{name}_{i / 500 * 3 + 2}"] = image[i:i + 500, 1000:1500]
                else:
                    key = f"{name}_{i/500*3}"
                    d[key] = fix_nodata(f"{dir}/{f}", image[i:i+500, :500], data[split]["label"].get(key, None))
                    key = f"{name}_{i/500*3+1}"
                    d[key] = fix_nodata(f"{dir}/{f}", image[i:i+500, 500:1000], data[split]["label"].get(key, None))
                    key = f"{name}_{i/500*3+2}"
                    d[key] = fix_nodata(f"{dir}/{f}", image[i:i+500, 1000:1500], data[split]["label"].get(key, None))

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

save = True
save_as_images = False
data_dict = {}
batch_size = 250
data = load_eli()

if save_as_images:
    for split in data:
        images_list = []
        labels_list = []

        image_path = f"{target_path}/{split}/image"
        label_path = f"{target_path}/{split}/label"
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        for k in data[split]["image"]:
            #if np.any(data[split]["image"][k]):
            cv2.imwrite(f"{image_path}/{k}.png", data[split]["image"][k])
            cv2.imwrite(f"{label_path}/{k}.png", data[split]["label"][k])
            #else:
            #    print(f"dropping {split}/{k}")


if save:
    for split in data:
        images_list = []
        labels_list = []

        if split == "test":
            for k in data[split]["image"]:
                images_list.append(data[split]["image"][k])
                labels_list.append(data[split]["label"][k])

        else:
            for k in data[split]["image"]:
                if np.any(data[split]["image"][k]):
                    images_list.append(data[split]["image"][k])
                    labels_list.append(data[split]["label"][k])
                else:
                    print(f"dropping {split}/{k}")
                    #assert not np.any(data[split]["label"][k])

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
    repo_id = "asolodin/hibbing_roads_train_demo"  # Replace with your repo ID
    os.makedirs(f"datasets/{repo_id}", exist_ok=True)
    dataset_dict.save_to_disk(f"datasets/{repo_id}")

    dataset_dict.push_to_hub(repo_id)

