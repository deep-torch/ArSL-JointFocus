# ArSL Recognition

This project is focused on recognizing Arabic Sign Language (ArSL) using deep learning models. It contains a full pipeline for downloading, processing data, training, and evaluating two types of models: a baseline convolutional neural network (CNN) and a pretrained EfficientNet model.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Signer Dependency](#signer-dependency)
- [Usage](#usage)
- [Citing the KArSL Dataset](#citing-the-karsl-dataset)
- [TODO](#todo)
- [Acknowledgments](#acknowledgments)

## Project Overview

The ArSL Recognition project implements two neural network models to perform sign language classification to 502 different class. The project provides scripts to:
- Download and extract a dataset of sign language gestures.
- Preprocess the data by extracting and selecting frames from videos.
- Train and evaluate models based on different architectures.

The baseline model is a custom-built CNN combined with an LSTM for temporal processing, while the pretrained model uses EfficientNet for feature extraction.

## Dataset

KArSL is the largest video dataset for Word-Level Arabic sign language (ArSL). The database consists of 502 isolated sign words collected using Microsoft Kinect V2. Each sign of the database is performed by three professional signers. The data is split into train and test sets, and processed to ensure that the length of each video is consistent.

### Data Download and Preparation

To download the dataset, the `download_data.py` script is used. This downloads and extracts data from Google Drive using specified file IDs. Afterward, the `process_data.py` script processes the frames of each video, retaining only key frames to reduce computation load during training.


## How It Works

This project contains several Python scripts and a Jupyter notebook for running various parts of the pipeline.

- **Models**: 
  - `baseline_model.py`: Defines a CNN + LSTM model. The CNN extracts features from the video frames, and the LSTM processes the temporal sequence of the video. The final output is a classification over 502 gestures.
  - `pretrained_model.py`: Uses EfficientNet as a feature extractor followed by an LSTM to process the sequence of frames and classify gestures.

- **Data Processing**: 
  - `data.py`: Contains the `KArSL` dataset class that loads the frames from the video, applies the necessary transformations (like resizing and normalization), and ensures all sequences are the same length.
  - `download_data.py`: Downloads the dataset from Google Drive.
  - `process_data.py`: Processes the video frames by selecting every 10th frame to reduce redundancy.

- **Training**: 
  - `train.py`: This script trains the model on the dataset. It uses PyTorch for model training and includes options for resuming training from checkpoints.
  
- **Utilities**:
  - `utils.py`: Utility functions for saving/loading checkpoints and determining the available device (GPU or CPU).

### Signer Dependency
- **Dependent Training**: The model is trained on a subset of the signers and evaluated on the same group of signers. This scenario is less challenging because the model has already seen similar gestures from the same individuals.
- **Independent Training**: The model is trained on a different subset of signers and evaluated on a new signer that the model has never seen before. This simulates real-life conditions where the model must generalize to new signers.


## Usage

You can either use the provided Jupyter notebook `arsl_usage_example.ipynb` for an interactive demonstration or run the scripts directly from the command line.

### Example Commands:

- Download and extract the dataset:
    ```bash
    !python -m arsl.download_data --files_ids [FILE_IDS] --extract_dir [EXTRACT_PATH]
    ```

- Preprocess the dataset:
    ```bash
    !python -m arsl.process_data --root_dir [DATASET_PATH]
    ```

- Train the model:
    ```bash
    !python -m arsl.train --root_dir [DATASET_PATH] --labels_path [LABELS_FILE_PATH] --checkpoints_dir [CHECKPOINTS_DIR] --model_type [baseline/pretrained] --training_mode [dependent/independent]
    ```

## Citing the KArSL Dataset

This project uses the [KArSL: Arabic Sign Language Database](https://dl.acm.org/doi/10.1145/3423420#:~:text=Signs%20in%20KArSL%20database%20are,language%20recognition%20using%20this%20database).

```bibtex
@article{sidig2021karsl,  
    title={KArSL: Arabic Sign Language Database},  
    author={Sidig, Ala Addin I and Luqman, Hamzah and Mahmoud, Sabri and Mohandes, Mohamed},  
    journal={ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},  
    volume={20},  
    number={1},  
    pages={1--19},  
    year={2021},  
    publisher={ACM New York, NY, USA}  
}
```

## TODO

- Implement an **attention mechanism** on joints to improve model performance in signer-independent training.


## Acknowledgments

This project was created by [Alaa Hassoun](https://github.com/alaaHassoun86) and [Joshua Mohammed](https://github.com/JoshuaMohammed) as a final project for DeepTorch Training. All the work, including the code, dataset processing, and model implementation, was done solely by the two of us.
