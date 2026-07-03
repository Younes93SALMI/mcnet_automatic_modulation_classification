# MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification

TensorFlow/Keras implementation of the CNN architecture proposed in:

> Thien Huynh-The, Cam-Hao Hua, Quoc-Viet Pham, and Dong-Seong Kim,
> **MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification**,
> IEEE Communications Letters, vol. 24, no. 4, pp. 811–815, 2020. :contentReference[oaicite:0]{index=0}

This repository reproduces the original **MCNet** architecture for **Automatic Modulation Classification (AMC)** using the **DeepSig RadioML 2018.01A** dataset.

---

# Overview

Automatic Modulation Classification (AMC) is a fundamental component of modern wireless communication systems, including cognitive radio, spectrum sensing, 5G/6G networks, and intelligent receivers.

MCNet is a lightweight convolutional neural network specifically designed for robust modulation recognition under challenging wireless channel impairments. The architecture combines:

- asymmetric convolution kernels,
- multi-branch feature extraction,
- residual (skip) connections,
- efficient feature aggregation,

allowing accurate modulation classification while maintaining a relatively small number of trainable parameters. :contentReference[oaicite:1]{index=1}

---

# Repository Structure

```
MCNET_AN_EFFICIENT_CNN_ARCHITECTURE_FOR_ROBUST_AUTOMATIC_MODULATION_CLASSIFICATION
│
├── datasets/
│   └── GOLD_XYZ_OSC.0001_1024.hdf5
│
├── DeepLearning/
│   ├── __init__.py
│   └── mcnet.py
│
├── results/
│   ├── mcnet_rml2018_best.keras
│   └── mcnet_training_log.csv
│
├── mcnet_an_efficient_cnn_architecture_for_robust_automatic_modulation_classification.ipynb
│
└── README.md
```

---

# Implemented Features

- TensorFlow/Keras implementation of MCNet
- DeepSig RadioML 2018.01A support
- Efficient HDF5 dataset loading
- GPU training
- Mini-batch streaming using TensorFlow Dataset API
- Model checkpointing
- Learning-rate scheduling
- Training log generation
- Classification report
- Confusion matrix
- Accuracy versus SNR evaluation

---

# Dataset

The implementation uses the **DeepSig RadioML 2018.01A** dataset.

Dataset characteristics:

| Property | Value |
|-----------|------:|
| Modulations | 24 |
| SNR values | 26 |
| SNR range | -20 dB to +30 dB |
| Frames per modulation-SNR pair | 4096 |
| Samples per frame | 1024 |
| Input shape | (1024,2) |
| Total frames | 2,555,904 |

Each IQ frame

```
(1024,2)
```

is converted into the MCNet input format

```
(2,1024,1)
```

before entering the neural network.

---

# Network Architecture

The implemented network follows the original MCNet architecture described in the paper:

- Initial 3×7 convolution
- Pre-block using asymmetric 3×1 and 1×3 convolutions
- Six M-blocks
- Multi-scale skip connections
- Feature concatenation
- Global average pooling
- Fully connected classifier
- Softmax output layer

The architecture exploits asymmetric convolutions to reduce computational complexity while preserving the discriminative capability of learned features. :contentReference[oaicite:2]{index=2}

---

# Training Configuration

The implementation follows the experimental configuration reported in the original paper.

| Parameter | Value |
|-----------|------:|
| Optimizer | SGD |
| Momentum | 0.9 |
| Initial learning rate | 0.01 |
| Batch size | 128 |
| Epochs | 60 |
| Train/Test split | 80% / 20% |

The learning rate is reduced after 30 epochs, reproducing the training strategy adopted by the authors. :contentReference[oaicite:3]{index=3}

---

# Running the Notebook

Launch

```
mcnet_an_efficient_cnn_architecture_for_robust_automatic_modulation_classification.ipynb
```

The notebook performs the complete pipeline:

1. Load the RadioML2018 dataset
2. Preprocess IQ samples
3. Split the dataset into training and testing subsets
4. Build the MCNet architecture
5. Train the network
6. Save the trained model
7. Evaluate the classifier
8. Generate the confusion matrix
9. Compute the classification report
10. Plot the accuracy versus SNR

---

# Results

The notebook generates:

- Training accuracy
- Validation accuracy
- Test accuracy
- Learning curves
- Classification report
- Confusion matrix
- Accuracy versus SNR
- Saved TensorFlow model
- Training log

---

# Requirements

- Python 3.11
- TensorFlow 2.17
- NumPy 1.26
- h5py
- scikit-learn
- matplotlib
- CUDA-compatible NVIDIA GPU (recommended)

---

# Citation

If you use this implementation in your research, please cite both the original MCNet paper and the DeepSig RadioML 2018.01A dataset.

```bibtex
@article{HuynhThe2020MCNet,
  title={MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification},
  author={Huynh-The, Thien and Hua, Cam-Hao and Pham, Quoc-Viet and Kim, Dong-Seong},
  journal={IEEE Communications Letters},
  volume={24},
  number={4},
  pages={811--815},
  year={2020},
  publisher={IEEE}
}
```

---

# Acknowledgements

This implementation reproduces the architecture proposed by Huynh-The *et al.* and uses the publicly available **DeepSig RadioML 2018.01A** dataset for research and educational purposes.

---

# License

This repository is intended for research and educational purposes. Users are responsible for complying with the licenses of the original MCNet publication and the DeepSig RadioML dataset.
