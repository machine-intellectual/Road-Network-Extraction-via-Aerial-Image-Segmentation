# Road Network Extraction via Aerial Image Segmentation

## Important Notice
This repository contains code from an academic group project completed for CSCI 5561 Computer Vision at the University of Minnesota Twin Cities (Fall 2024). The code is shared for portfolio and demonstration purposes only. All rights are reserved and the code is not licensed for reuse, as it represents collaborative work with shared ownership.

### Permissions & Usage
- This code is provided for reference and demonstration only
- No license is granted for reuse or redistribution
- All contributors retain their rights to their work
- Please contact all contributors for any usage requests

## Project Overview

This project explores computer vision approaches to extract road networks from aerial imagery, focusing on two key applications:
1. Disaster recovery - identifying usable roadways after natural disasters
2. Road detection in underdeveloped areas - detecting less defined roads like gravel/dirt paths

## Team Contributions

### Kyle Anthes
- Wrote custom load dataset script to read in custom GeoTIFF dataset, calculate summary statistics, and visualize outlier samples  
- Implemented ResUNet notebook with custom evaluation metrics

### Eli Schlossberg
- Gathered and curated data into training dataset using a number of geoprocessing techniques
- Utilized DiffusionSat to generate a novel dataset of disaster data from the existing road network data

### Andrei Solodin  
- Generated HuggingFace dataset using the raw dataset
- Created a notebook to train and evaluate SegFormer on various datasets
- Created a demo application to demonstrate potential usage of fine-tuned model for real-time road/damage detection

### Jack Swanberg
- Implemented U-Net model in PyTorch, trained on dataset and recorded metrics
- Implemented road-network extraction, taking model output road masks and converting to road network graphs

## Technical Details

### Models Explored
We implemented and evaluated three prominent models for semantic segmentation:
- UNet
- Segformer  
- ResUNet

### Training Data
The models were trained on a combination of:
- Aerial imagery from the National Aerial Imagery Program (NAIP)
- Road masks derived from Minnesota Department of Transportation (MnDoT) road centerlines
- St. Louis County Aerial Imagery Program data (1m resolution RGB)
- OpenStreetMap road centerlines 
- Massachusetts Roads Dataset for additional training samples

### Dataset Split
- Train: 2814 samples
- Validation: 60 samples  
- Test: 608 samples

### Model Performance

Model results on test set:

| Model | mIOU | F1 Score |
|-------|------|-----------|
| SegFormer-b0 | 0.41 | 0.56 |
| SegFormer-b3 | 0.53 | 0.67 |
| UNet | 0.553 | 0.675 |
| ResUNet | 0.499 | 0.665 |

## Key Features

- Road network extraction from aerial imagery
- Real-time road damage detection demo application
- Road network graph generation from segmentation masks
- Support for both urban and rural road detection

## Key Findings
- Models demonstrated better performance on urban/suburban roads compared to rural roads
- Data augmentation experiments highlighted the importance of normalizing training images
- Real-time inference proved possible even with limited computational resources
- Results suggest potential applications in disaster response and infrastructure mapping

## Course Information
- Course: CSCI 5561 Computer Vision
- University: University of Minnesota Twin Cities  
- Semester: Fall 2024