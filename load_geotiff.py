import os
import logging
import rasterio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split




def prepare_dataset(data_dir, image_size=(128, 128)):
    """Prepare dataset for training"""
    images = []
    masks = []
    file_paths = []
    data_dir = Path(data_dir)

    # Get all TIF files (case insensitive)
    tif_files = list(data_dir.glob('**/*.[Tt][Ii][Ff]')) + \
                list(data_dir.glob('**/*.[Tt][Ii][Ff][Ff]'))


    # Process files with progress tracking
    for i, file_path in enumerate(tif_files):
        if i % 100 == 0:
            logger.info(f"Processing file {i + 1}/{len(tif_files)}")

        img, mask = load_geotiff(str(file_path))
        if img is not None and mask is not None:
            # Resize image and mask
            img = tf.image.resize(img, image_size)
            mask = tf.image.resize(mask, image_size, method='nearest')

            # Ensure float32 dtype
            img = tf.cast(img, tf.float32)
            mask = tf.cast(mask, tf.float32)

            images.append(img)
            masks.append(mask)
            file_paths.append(file_path)

    if not images:
        raise ValueError(f"No valid images found in {data_dir}")

    # Convert lists to numpy arrays
    images_array = np.array(images)
    masks_array = np.array(masks)

    logger.info(f"Final dataset shape: {images_array.shape}, masks shape: {masks_array.shape}")
    logger.info(f"Number of images with masks: {len(images)}")

    return images_array, masks_array, file_paths


def visualize_samples(images, file_paths, title, num_samples=16, save_path='/app/plots/'):
    """Visualize a grid of sample images"""
    plt.close('all')

    # Create a figure with a grid of subplots
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig = plt.figure(figsize=(20, 20))

    # Randomly select samples
    indices = np.random.choice(len(images), num_samples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)

        # Get the image and filename
        img = images[idx]
        filename = Path(file_paths[idx]).name

        # Calculate some statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)

        # Display the image
        plt.imshow(img)
        plt.title(f'Title: {title}\n' +
                  f'File: {filename}\n' +
                  f'Range: [{min_val:.3f}, {max_val:.3f}]')
        plt.axis('off')

    plt.tight_layout()

    plt.show()