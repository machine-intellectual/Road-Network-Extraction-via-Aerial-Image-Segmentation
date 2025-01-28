import os

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.dtypes import StringDType
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import gdown

id2label = {1: 'road'}
label2id = {v: k for k, v in id2label.items()}

ds_directory = "demo_ds"
checkpoint = "asolodin/segformer-hibbing-b3-demo"
proc_checkpoint = "nvidia/mit-b3"

print("Loading model, please wait...")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
image_processor = AutoImageProcessor.from_pretrained(proc_checkpoint, do_reduce_labels=False)
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
print("Done, using ")
if torch.cuda.is_available():
    model = model.cuda()
    print("CUDA")
else:
    print("CPU")

# Requires a dataset from https://drive.google.com/file/d/1Oqmc_-8OksAeD690gLxMbZYPzLusy2YW/view?usp=drive_link
class AerialRoadDetector:
    def __init__(self, directory, image_size=(512, 512)):
        self.event = None
        self.directory = directory
        self.image_size = image_size
        image_files = [f for f in os.listdir(f"{directory}/image") if f.endswith(('png')) and "image" in f]
        mask_files = [f for f in os.listdir(f"{directory}/label") if f.endswith(('png')) and "image" in f]
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        mask_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

        self.current_index = (10, 58)  # Index of the first image in the current batch
        self.map_shape = (19, 74)

        self.win_size = (6, 9)
        self.main_win_size = (6, 6)
        self.mask_scale = 0.5

        self.overlay_preds = False

        self.image_map, self.mask_map = self.create_image_map(image_files, mask_files)
        self.full_image = np.zeros((self.map_shape[0] * self.image_size[0], self.map_shape[1] * self.image_size[1], 3), dtype=np.uint8)
        self.full_mask = self.load_mask()
        self.pred_mask = np.ones_like(self.full_mask)

        #self.create_mask_image()

        # Set up the plot for displaying images
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.patch.set_facecolor('black')
        self.ax.axis('off')  # Turn off axis

        self.img_buffer = np.zeros((self.win_size[0] * self.image_size[0],
                                    self.win_size[1] * self.image_size[1], 3), dtype=np.uint8)

        self.mask_win_size = (
            int(self.main_win_size[0] * self.mask_scale), int(self.main_win_size[1] * self.mask_scale))
        self.pred_win_size = self.mask_win_size

        # Load and display the first set of images
        self.update_display()

        # Connect the key press events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def compute_mask(self, logits_tensor):
        # Convert logits to probabilities
        pred_labels = torch.sigmoid(logits_tensor)

        threshold = 0.5
        pred_labels[pred_labels > threshold] = 1
        pred_labels[pred_labels <= threshold] = 0

        pred_labels = pred_labels.detach().cpu().numpy()

        return pred_labels

    def normalize_histogram(self, image):
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return output

    def filter_mask(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all contours and remove the small ones
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 64:
                # If the contour area is less than 64, set it to 0 (remove the blob)
                cv2.drawContours(mask, [contour], -1, (0), thickness=cv2.FILLED)
        return mask

    def label_image(self, image):
        image = self.normalize_histogram(image)
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
        prediction: np.ndarray = self.compute_mask(upsampled_logits[0])
        prediction[prediction >= 1] = 255
        prediction = self.filter_mask(prediction.astype(np.uint8))
        return prediction

    def create_image_map(self, image_files, mask_files):
        image_row_size = 156
        image_map = np.empty(self.map_shape, dtype=StringDType())
        mask_map = np.empty(self.map_shape, dtype=StringDType())

        current_index = 0
        start_offset = 1248
        for j in range(self.map_shape[0]):
            for i in range(self.map_shape[1]):
                current_image_num = j * image_row_size + i + start_offset
                while True:
                    file_name = image_files[current_index]
                    image_num = int(''.join(filter(str.isdigit, file_name)))
                    if image_num >= current_image_num:
                        break
                    current_index += 1

                if image_num > current_image_num:
                    image_map[j, i] = "blank512.png"
                    mask_map[j, i] = "blank512.png"
                else:
                    image_map[j, i] = file_name
                    mask_map[j, i] = mask_files[current_index]
                    current_index += 1

        image_map = np.flip(image_map, 0)
        mask_map = np.flip(mask_map, 0)
        return image_map, mask_map

    def load_mask(self):
        scaled_size = (int(self.image_size[0] * self.mask_scale), int(self.image_size[1] * self.mask_scale))
        image = np.zeros((self.map_shape[0] * scaled_size[0],
                          self.map_shape[1] * scaled_size[1]), dtype=np.uint8)
        for j in range(self.map_shape[0]):
            j1 = j * scaled_size[0]
            j2 = (j + 1) * scaled_size[0]
            for i in range(self.map_shape[1]):
                image_path = os.path.join(f"{self.directory}/label", self.mask_map[j, i])
                img = Image.open(image_path)
                img = img.resize(scaled_size, Image.Resampling.LANCZOS)
                image[j1:j2, i * scaled_size[1]:(i + 1) * scaled_size[1]] = img
        return image

    def create_mask_image(self):
        image = np.zeros((self.map_shape[0] * self.image_size[0], self.map_shape[1] * self.image_size[1]),
                         dtype=np.uint8)
        for j in range(self.map_shape[0]):
            j1 = j * self.image_size[0]
            j2 = (j + 1) * self.image_size[0]
            for i in range(self.map_shape[1]):
                image_path = os.path.join(f"{self.directory}/label", self.mask_map[j, i])
                img = Image.open(image_path)
                image[j1:j2, i * self.image_size[1]:(i + 1) * self.image_size[1]] = img
        cv2.imwrite(f"{self.directory}/full_mask.png", image)

        for j in range(self.map_shape[0]):
            j1 = j * self.image_size[0]
            j2 = (j + 1) * self.image_size[0]
            for i in range(self.map_shape[1]):
                image_path = os.path.join(f"{self.directory}/image", self.image_map[j, i])
                img = Image.open(image_path)
                print("labeling ", image_path)
                label = self.label_image(img)
                image[j1:j2, i * self.image_size[1]:(i + 1) * self.image_size[1]] = label
        cv2.imwrite(f"{self.directory}/pred_mask.png", image)

    def draw_images(self, start_index):
        for j in range(self.main_win_size[0]):
            j1 = j * self.image_size[0]
            j2 = (j + 1) * self.image_size[0]
            for i in range(self.main_win_size[1]):
                i1 = i * self.image_size[1]
                i2 = i1 + self.image_size[1]

                j1_img = (start_index[0] + j) * self.image_size[0]
                j2_img = j1_img + self.image_size[0]
                i1_img = (start_index[1] + i) * self.image_size[1]
                i2_img = i1_img + self.image_size[1]

                if not np.any(self.full_image[j1_img:j2_img, i1_img:i2_img]):
                    image_path = os.path.join(f"{self.directory}/image",
                                              self.image_map[start_index[0] + j, start_index[1] + i])
                    self.full_image[j1_img:j2_img, i1_img:i2_img] = Image.open(image_path)
                self.img_buffer[j1:j2, i1:i2] = self.full_image[j1_img:j2_img, i1_img:i2_img]

    def draw_masks(self, start_index):
        mask_size = (int(self.image_size[0] * self.mask_scale), int(self.image_size[1] * self.mask_scale))
        j1_buf = 0
        j2_buf = j1_buf + self.mask_win_size[0] * self.image_size[0]
        j1_mask = start_index[0] * mask_size[0]
        j2_mask = j1_mask + (self.mask_win_size[0] * self.image_size[0])
        i1_buf = self.main_win_size[1] * self.image_size[1]
        i2_buf = i1_buf + self.mask_win_size[1] * self.image_size[1]
        i1_mask = start_index[1] * mask_size[1]
        i2_mask = i1_mask + (self.mask_win_size[1] * self.image_size[1])

        mask_image = np.repeat(
            self.full_mask[j1_mask:j2_mask, i1_mask:i2_mask][:, :, np.newaxis], 3, axis=2)

        if self.overlay_preds:
            pred_mask = self.pred_mask[j1_mask:j2_mask, i1_mask:i2_mask]
            mask_image[:, :, 2][np.logical_and(pred_mask == 0, mask_image[:, :, 1] == 255)] = 0

        self.img_buffer[j1_buf:j2_buf, i1_buf:i2_buf] = mask_image
        cv2.rectangle(self.img_buffer, (i1_buf, j1_buf), (i2_buf - 1, j2_buf - 1), color=(255, 0, 0), thickness=2)

    def draw_preds(self, start_index):
        mask_size = (int(self.image_size[0] * self.mask_scale), int(self.image_size[1] * self.mask_scale))
        j1_buf = self.mask_win_size[0] * self.image_size[0]
        j2_buf = j1_buf + self.mask_win_size[0] * self.image_size[0]
        j1_mask = start_index[0] * mask_size[0]
        j2_mask = j1_mask + (self.mask_win_size[0] * self.image_size[0])
        i1_buf = self.main_win_size[1] * self.image_size[1]
        i2_buf = i1_buf + self.mask_win_size[1] * self.image_size[1]
        i1_mask = start_index[1] * mask_size[1]
        i2_mask = i1_mask + (self.mask_win_size[1] * self.image_size[1])

        # populate preds if necessary
        for j in range(self.main_win_size[0]):
            for i in range(self.main_win_size[1]):
                j_pred = j1_mask + j * mask_size[0]
                i_pred = i1_mask + i * mask_size[1]
                if self.pred_mask[j_pred, i_pred] == 1:
                    pred_mask = self.label_image(self.img_buffer[j * self.image_size[0]:(j + 1) * self.image_size[0],
                                                 i * self.image_size[1]:(i + 1) * self.image_size[1]])
                    cv2.resize(pred_mask, mask_size,
                               self.pred_mask[j_pred:j_pred + mask_size[0], i_pred:i_pred + mask_size[1]],
                               interpolation=cv2.INTER_LINEAR)

        self.img_buffer[j1_buf:j2_buf, i1_buf:i2_buf] = np.repeat(
            self.pred_mask[j1_mask:j2_mask, i1_mask:i2_mask][:, :, np.newaxis], 3, axis=2)
        cv2.rectangle(self.img_buffer, (i1_buf, j1_buf), (i2_buf - 1, j2_buf - 1), color=(0, 255, 0), thickness=2)

    def update_display(self):
        self.img_buffer.fill(0)
        self.draw_images(self.current_index)
        self.draw_preds(self.current_index)
        self.draw_masks(self.current_index)

        # Display the combined image
        self.ax.clear()  # Clear the previous image
        self.ax.imshow(self.img_buffer)
        self.ax.axis('off')  # Hide the axis
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.draw()  # Redraw the plot

    def on_key_press(self, event):
        self.event = event
        if event.key == 'right':
            if self.current_index[1] + 1 < (self.map_shape[1] - self.win_size[1] + 1):
                self.current_index = (self.current_index[0], self.current_index[1] + 1)
                self.update_display()
        elif event.key == 'left':
            if self.current_index[1] - 1 >= 0:
                self.current_index = (self.current_index[0], self.current_index[1] - 1)
                self.update_display()
        elif event.key == 'up':
            if self.current_index[0] - 1 >= 0:
                self.current_index = (self.current_index[0] - 1, self.current_index[1])
                self.update_display()
        elif event.key == 'down':
            if self.current_index[0] + 1 < (self.map_shape[0] - self.win_size[0] + 1):
                self.current_index = (self.current_index[0] + 1, self.current_index[1])
                self.update_display()
        elif event.key == 'o':
            self.overlay_preds = not self.overlay_preds
            self.update_display()

    def on_click(self, event):
        i, j = event.xdata, event.ydata

        if i is not None and j is not None and 0 <= i < self.image_size[1] * self.main_win_size[1] and \
                0 <= j < self.image_size[0] * self.main_win_size[0]:
            i_tile = tuple(map(int, divmod(i, self.image_size[1])))
            j_tile = tuple(map(int, divmod(j, self.image_size[0])))

            radius = 60
            i_center = (i_tile[0] + self.current_index[1]) * self.image_size[1] + i_tile[1]
            j_center = (j_tile[0] + self.current_index[0]) * self.image_size[0] + j_tile[1]
            if np.all(self.full_image[j_center, i_center] != [255, 0, 0]):
                cv2.circle(self.full_image, (i_center, j_center), radius, (255, 0, 0), thickness=cv2.FILLED)
                mask_size = (int(self.image_size[0] * self.mask_scale), int(self.image_size[1] * self.mask_scale))
                # erase the corresponding predictions
                for j1 in range(
                        (j_tile[0] + self.current_index[0] - (1 if j_tile[1] - radius < 0 else 0)) * mask_size[0],
                        (j_tile[0] + self.current_index[0] + (2 if j_tile[1] + radius > self.image_size[0] else 1)) * mask_size[0],
                        mask_size[0]):
                    for i1 in range(
                            (i_tile[0] + self.current_index[1] - (1 if i_tile[1] - radius < 0 else 0)) * mask_size[1],
                            (i_tile[0] + self.current_index[1] + (2 if i_tile[1] + radius > self.image_size[1] else 1)) * mask_size[1],
                            mask_size[1]):
                        self.pred_mask[j1, i1] = 1

            #else:
            #    for j1 in range(
            #            (j_tile[0] + self.current_index[0] - 1 if j_tile[1] - radius < 0 else 0) * self.image_size[0],
            #            (j_tile[0] + self.current_index[0] + 1 if j_tile[1] + radius > self.image_size[0] else 0) * self.image_size[0]):
            #        j2 = j1 + self.image_size[0]
            #        for i1 in range(
            #                (i_tile[0] + self.current_index[1] - 1 if i_tile[1] - radius < 0 else 0) * self.image_size[1],
            #                (i_tile[0] + self.current_index[1] + 1 if i_tile[1] + radius > self.image_size[1] else 0) * self.image_size[1]):
            #            i2 = i1 + self.image_size[1]
            #            self.full_image[j1:j2, i1:i2].fill(0)

            self.update_display()


    def start(self):
        plt.show()


def main():
    if not os.path.isdir(ds_directory):
        url = 'https://drive.google.com/uc?id=1Oqmc_-8OksAeD690gLxMbZYPzLusy2YW'
        output = 'ds.zip'
        print("Downloading dataset, please wait...")
        gdown.download(url, output, quiet=False)
        import zipfile
        print("Done. Extracting dataset, please wait...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
            print("Done.")

    print("Loading images, please wait...")
    app = AerialRoadDetector(ds_directory)
    app.start()


if __name__ == '__main__':
    main()
