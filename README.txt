# Pneumonia Detection in Chest X-ray Images Using Ensemble Deep Learning Models: Replication and Enhancement with CBAM Attention

## 📌 Project Overview

Pneumonia is a respiratory infection that affects the lungs and can cause inflammation and fluid accumulation in the air sacs. Chest X-ray imaging is commonly used to support the assessment of pneumonia. However, interpreting X-ray images can be challenging because of variations in image appearance and similarities between normal and abnormal lung patterns.

This project implements a deep learning-based approach for detecting pneumonia from chest X-ray images. It is based on the research paper:

**"Pneumonia Detection in Chest X-ray Images Using an Ensemble of Deep Learning Models" – Kundu et al. (2021), PLOS ONE.**

The project implements multiple Convolutional Neural Network (CNN) architectures, evaluates a probability-based ensemble approach, and introduces a research enhancement using the **Convolutional Block Attention Module (CBAM)** integrated into DenseNet-121.

The objective is to investigate whether attention-based feature refinement can improve the classification of chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

This work is a modified replication and experimental extension of the reference study, rather than an exact reproduction of every experiment in the original paper.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Study and implement the methodology proposed in the reference research paper.
2. Prepare and preprocess the chest X-ray dataset for deep learning.
3. Implement and train ResNet-18, GoogLeNet, and DenseNet-121.
4. Evaluate a probability-based ensemble of the baseline models.
5. Enhance DenseNet-121 using the Convolutional Block Attention Module (CBAM).
6. Compare baseline and enhanced models using multiple evaluation metrics.
7. Analyse model performance using confusion matrices, ROC curves, and classification metrics.
8. Evaluate computational efficiency using inference latency, throughput, and GPU memory consumption.
9. Apply paired statistical analysis to compare baseline and enhanced model predictions.

---

## 📂 Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

**Dataset source:**  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains chest X-ray images belonging to two classes:

- NORMAL – X-ray images of normal lungs.
- PNEUMONIA – X-ray images showing pneumonia-related abnormalities.

### Dataset Distribution

| Dataset partition | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Original training set | 1,341 | 3,875 | 5,216 |
| Original validation set | 8 | 8 | 16 |
| Official test set | 234 | 390 | 624 |

Since the original validation set contained only 16 images, a stratified validation split was created from the original training data.

The resulting experimental split was:

| Partition | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,140 | 3,293 | 4,433 |
| Validation | 201 | 582 | 783 |
| Official test | 234 | 390 | 624 |

The class labels used in the implementation are:

- NORMAL = 0
- PNEUMONIA = 1

The dataset is imbalanced, with more pneumonia images than normal images. Therefore, class-wise evaluation metrics are important when interpreting the results.

---

## ⚙️ Technologies and Tools Used

The project was implemented using the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development and execution environment |
| PyTorch | Deep learning framework |
| Torchvision | Pretrained CNN architectures and image transformations |
| NumPy | Numerical operations |
| Pandas | Data handling and analysis |
| Matplotlib | Visualization of training results |
| Scikit-learn | Classification metrics and evaluation |
| Kaggle API | Dataset download |
| GitHub | Project version control and source-code sharing |

The experiments were executed using GPU acceleration in Google Colab.

---

## 🧹 Data Preprocessing

Before training the models, the chest X-ray images were processed using the following steps:

1. Images were loaded from the dataset directories.
2. Images were resized to **224 × 224 pixels**.
3. Grayscale images were converted into three-channel images to match the input requirements of the pretrained CNN architectures.
4. Images were converted into PyTorch tensors.
5. ImageNet mean and standard deviation normalization were applied.
6. Random horizontal flipping was used as a training augmentation.
7. A fixed random seed was used to support repeatability of the data split.

These preprocessing steps provide a consistent input format for all the baseline models and the CBAM-enhanced model.

---

## 🧠 Baseline Deep Learning Models

Three CNN architectures were implemented using transfer learning.

### 1. ResNet-18

ResNet-18 is a residual convolutional neural network that uses skip connections to support the training of deeper networks.

Residual connections help information and gradients flow through the network, reducing the difficulty of optimization.

In this project, ResNet-18 was adapted for binary classification of chest X-ray images.

### 2. GoogLeNet

GoogLeNet is a convolutional neural network based on the Inception architecture.

It uses multiple convolutional operations with different filter sizes within its Inception modules. This allows the network to learn features at different spatial scales.

GoogLeNet was adapted to classify chest X-ray images into NORMAL and PNEUMONIA categories.

### 3. DenseNet-121

DenseNet-121 uses dense connections between layers, allowing each layer to receive feature maps from earlier layers within a dense block.

This encourages feature reuse and improves information and gradient propagation.

DenseNet-121 was implemented as a baseline model and later selected as the backbone for the CBAM enhancement.

Each baseline model was trained for 30 epochs.

---

## 🔗 Ensemble Learning

Ensemble learning combines predictions from multiple models to obtain a final prediction.

In this project, predictions from ResNet-18, GoogLeNet, and DenseNet-121 were used in an ensemble evaluation pipeline inspired by the reference study.

The project also evaluated a simple probability-averaging ensemble.

The ensemble approach was investigated to understand whether combining predictions from multiple CNN architectures could provide better classification performance than individual models.

The ensemble results are reported separately from the individual model results.

---

## 🚀 Research Enhancement: CBAM-DenseNet121

### Motivation

The baseline DenseNet-121 model learns useful visual features from chest X-ray images. However, the model does not explicitly use the CBAM attention mechanism implemented in this project.

To investigate whether the model could benefit from attention-based feature refinement, a **Convolutional Block Attention Module (CBAM)** was integrated into DenseNet-121.

### What is CBAM?

CBAM is an attention mechanism that applies two sequential attention operations:

**1. Channel Attention**

Channel attention identifies which feature channels are more important for the classification task.

It uses global average pooling and global max pooling to generate channel-wise attention information. The resulting attention weights are applied to recalibrate the input feature maps.

**2. Spatial Attention**

Spatial attention identifies important spatial locations within the feature maps.

It uses channel-wise average and maximum information, followed by convolution and sigmoid activation, to generate spatial attention weights.

### CBAM Integration

The CBAM module was integrated after the DenseNet-121 feature extraction and ReLU operation, before adaptive average pooling and the final classifier.

The processing sequence is:

Input Image → DenseNet-121 Feature Extractor → ReLU → Channel Attention → Spatial Attention → Adaptive Average Pooling → Fully Connected Classifier → Output

The enhanced model was fine-tuned for 15 epochs using Adam with an initial learning rate of 0.00001 and a StepLR learning-rate scheduler.

### Expected Benefit

The purpose of this enhancement is to investigate whether channel and spatial attention can help the model emphasize informative features in chest X-ray images.

The improvement is evaluated experimentally rather than assumed in advance.

---

## 📊 Experimental Results

All models were evaluated on the official test partition containing 624 images.

### Baseline Model Comparison

| Model | Test Accuracy | ROC-AUC | NORMAL Recall | PNEUMONIA Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 78.37% | 0.9435 | 42.74% | 99.74% | 0.7246 |
| GoogLeNet | 83.01% | 0.9312 | 54.70% | 100.00% | 0.7938 |
| DenseNet-121 | 74.84% | 0.8984 | 33.33% | 99.74% | 0.6652 |
| Probability-average ensemble | 78.37% | 0.9331 | 42.74% | 99.74% | 0.7246 |

### CBAM Enhancement Results

| Metric | DenseNet-121 | CBAM-DenseNet121 |
|---|---:|---:|
| Test Accuracy | 74.84% | 80.13% |
| ROC-AUC | 0.8984 | 0.9044 |
| NORMAL Recall | 33.33% | 47.44% |
| PNEUMONIA Recall | 99.74% | 99.74% |
| Macro F1-score | 0.6652 | 0.7521 |

The CBAM-enhanced model improved the DenseNet-121 baseline on the reported accuracy, ROC-AUC, NORMAL recall, and macro F1-score metrics.

However, CBAM-DenseNet121 did not outperform GoogLeNet in overall test accuracy or macro F1-score.

### Statistical Analysis

McNemar's exact test was applied to paired predictions from DenseNet-121 and CBAM-DenseNet121 on the same test images.

- Baseline correct, CBAM incorrect: 3 images
- Baseline incorrect, CBAM correct: 36 images
- Exact two-sided p-value: 3.608874976634979e-08

This test indicates a statistically significant difference in the paired prediction outcomes under the implemented experiment. It does not establish clinical validity or prove that the enhancement will generalize to other datasets.

### Computational Efficiency

The inference benchmark recorded the following measurements:

| Metric | DenseNet-121 | CBAM-DenseNet121 |
|---|---:|---:|
| Inference latency per image | 3.136 ms | 3.287 ms |
| Throughput | 318.921 images/s | 304.205 images/s |
| Peak allocated GPU memory | 8765.591 MB | 8765.591 MB |

The CBAM-enhanced model required slightly more inference time per image and achieved lower throughput in the recorded benchmark.

---

## 📈 Visualizations

The project includes visualizations generated during model training and evaluation, including:

- Dataset class distribution.
- Representative NORMAL and PNEUMONIA chest X-ray images.
- Training and validation accuracy curves.
- Training and validation loss curves.
- Confusion matrices for individual models.
- ROC curves for individual models.
- Ensemble evaluation results.
- CBAM-DenseNet121 confusion matrix.
- CBAM-DenseNet121 learning curves.
- CBAM-DenseNet121 ROC curve.
- Baseline and enhanced model comparison.

These visualizations help analyse model learning behaviour, classification performance, and the effect of the attention-based enhancement.

---

## 🗂️ Project Structure

The repository is organized to preserve the notebook, supporting outputs, and project documentation.

```text
pneumonia-detection-cbam/
│
├── Pneumonia_Project_Consolidated_ReportReady.ipynb
│
├── figures/
│   ├── 01_dataset_distribution.png
│   ├── 02_sample_xrays.png
│   ├── resnet18_confusion_matrix.png
│   ├── googlenet_confusion_matrix.png
│   ├── densenet121_confusion_matrix.png
│   ├── ensemble_confusion_matrix.png
│   ├── cbam_densenet121_confusion_matrix.png
│   └── other training and evaluation plots
│
├── outputs/
│   └── notebook_text_outputs.txt
│
├── README.md
│
└── requirements.txt
```

*The folder and file names above are suggested for organizing the uploaded files. Keep the actual names of your files when uploading them.*

---

## ▶️ How to Run the Project

The project was developed in Google Colab.

### Step 1: Clone the Repository

```bash
git clone https://github.com/jerinklaudia/pneumonia-detection-cbam.git
```

### Step 2: Open the Notebook

Open the `.ipynb` notebook using Google Colab or Jupyter Notebook.

### Step 3: Download the Dataset

Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Configure the dataset path in the notebook according to the location where the dataset is extracted.

### Step 4: Install Dependencies

The main libraries used in the project include:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### Step 5: Execute the Notebook

Run the notebook cells in their intended order to perform:

1. Environment setup.
2. Dataset loading.
3. Image preprocessing.
4. Baseline model training.
5. Ensemble evaluation.
6. CBAM integration and fine-tuning.
7. Model evaluation and visualization.

**Note:** Re-running the complete notebook requires access to the dataset and a suitable computational environment. Previously saved notebook outputs can be viewed without retraining the models.

---

## ⚠️ Limitations

The following limitations should be considered when interpreting the results:

- The project uses a single publicly available chest X-ray dataset.
- The dataset contains an unequal distribution of NORMAL and PNEUMONIA images.
- The complete five-fold cross-validation and additional dataset experiments from the reference paper were not reproduced.
- The test set was evaluated repeatedly during project development and should not be considered a completely untouched final evaluation set.
- Performance may differ on images collected from other hospitals, equipment, and patient populations.
- The experiments do not establish clinical validity or demonstrate readiness for medical use.

The project is an academic research implementation and is not intended to replace professional medical diagnosis.

---

## 🔮 Future Enhancements

Possible directions for future work include:

- Evaluating the models on independent external chest X-ray datasets.
- Performing repeated experiments with different random seeds.
- Exploring alternative attention mechanisms.
- Investigating class imbalance handling techniques.
- Applying explainability methods such as Grad-CAM to visualize model attention.
- Improving probability calibration and decision-threshold selection.
- Optimizing inference speed and computational resource usage.
- Evaluating the approach with more extensive cross-validation.

---

## 📚 Reference Paper

Kundu, R., Santosh, K. C., et al. (2021). **Pneumonia detection in chest X-ray images using an ensemble of deep learning models.** *PLOS ONE, 16*(9), e0256630.

https://doi.org/10.1371/journal.pone.0256630

**Dataset:**  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Reference Implementation:**  
https://github.com/Rohit-Kundu/Ensemble-Pneumonia-Detection

---

## 📄 Disclaimer

This repository is intended for educational and research purposes only. The models developed in this project are experimental and have not been clinically validated. They must not be used as a substitute for professional medical advice, diagnosis, or treatment.
