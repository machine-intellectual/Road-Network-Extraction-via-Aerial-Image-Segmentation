import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from datasets import load_dataset, load_from_disk
from transformers import AutoImageProcessor
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from PIL import Image

repo_id = "asolodin/massachusetts_roads_dataset"
ds = load_from_disk(f"datasets/{repo_id}")
train_ds = ds["train"]
val_ds = ds["validation"]

id2label = {1: 'road'}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

checkpoint = "blah"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=False)

def train_transforms(example_batch):
    l = len(example_batch["image"])
    rot = np.random.choice([0, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270], l)
    images = [x.transpose(r) for r, x in zip(rot, example_batch["image"])]
    labels = [x.transpose(r) for r, x in zip(rot, example_batch["label"])]
    inputs = image_processor(images, labels)
    for label_arr in inputs.data['labels']:
        label_arr[np.where(label_arr > 0)] = 1
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    for label_arr in inputs.data['labels']:
        label_arr[np.where(label_arr > 0)] = 1

    return inputs

def split_images(batch):

    return batch

val_ds = val_ds.map(split_images)

val_ds.set_transform(val_transforms)
train_ds.set_transform(train_transforms)

def calculate_class_weights(labels):
    # Count the total number of pixels in the positive and negative classes
    total_pixels = labels.numel()
    pos_pixels = labels.sum()
    neg_pixels = total_pixels - pos_pixels

    # Calculate weights
    pos_weight = neg_pixels / (pos_pixels + 1e-6)  # Adding a small value to avoid division by zero
    neg_weight = pos_pixels / (neg_pixels + 1e-6)
    return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)


pos_weight = calculate_class_weights(torch.tensor([val_ds[i]["labels"] for i in range(len(val_ds))]))

def compute_iou_f1_batch(preds_batch, labels_batch):
    iou_scores = []
    f1_scores = []

    for preds, labels in zip(preds_batch, labels_batch):
        # Flatten the arrays
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()

        # IoU Calculation
        intersection = np.sum((preds_flat == 1) & (labels_flat == 1))
        union = np.sum(preds_flat) + np.sum(labels_flat) - intersection

        # Avoid division by zero
        iou = intersection / union if union > 0 else 0.0
        iou_scores.append(iou)

        # F1 Score Calculation
        TP = intersection
        FP = np.sum((preds_flat == 1) & (labels_flat == 0))
        FN = np.sum((preds_flat == 0) & (labels_flat == 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1_score)

    # Average the scores
    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)

    return {"avg_iou": avg_iou, "avg_f1": avg_f1}

def compute_mask(logits_tensor):
    # Convert logits to probabilities
    pred_labels = torch.sigmoid(logits_tensor)

    threshold = 0.5
    pred_labels[pred_labels > threshold] = 1.0
    pred_labels[pred_labels <= threshold] = 0

    pred_labels = pred_labels.detach().cpu().numpy()
    return pred_labels

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(dim=1)

        # Convert logits to probabilities
        pred_labels = compute_mask(logits_tensor)
        return compute_iou_f1_batch(pred_labels, labels)


model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="segformer-ma-roads",
    learning_rate=6e-5,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=4,
    eval_steps=4,
    logging_steps=1,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
)

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels, **kwargs):
        upsampled_logits = nn.functional.interpolate(
            logits.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.loss_fn(upsampled_logits.squeeze(1), labels.float())

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    compute_loss_func=WeightedBCEWithLogitsLoss(pos_weight=pos_weight[1])
)

trainer.train()