# Datasets

The **DeepSig RadioML2018.01A** dataset is **not included** in this repository due to its large size.

Before running the notebook, please download the required dataset and place the files in this directory.

## Required files

```text
datasets/
├── GOLD_XYZ_OSC.0001_1024.hdf5
```

Alternatively, you may keep the compressed archive:

```text
datasets/
├── GOLD_XYZ_OSC.0001_1024.hdf5
└── GOLD_XYZ_OSC.0001_1024.hdf5.zip
```

---

## Dataset

This implementation uses the **DeepSig RadioML2018.01A** dataset introduced in:

> T. J. O'Shea, T. Roy, and T. C. Clancy,
> *Over-the-Air Deep Learning Based Radio Signal Classification*,
> IEEE Journal of Selected Topics in Signal Processing, 2018.

The dataset is also used for evaluating the MCNet architecture presented in:

> Thien Huynh-The, Cam-Hao Hua, Quoc-Viet Pham, and Dong-Seong Kim,
> *MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification*,
> IEEE Communications Letters, 2020.

The dataset can be downloaded from the official DeepSig repository or other publicly available mirrors.

Example:

```
https://www.kaggle.com/datasets/pinxau1000/radioml2018?resource=download
```

---

## Dataset Characteristics

- **24 modulation classes**
- **26 SNR values**
- SNR range from **−20 dB** to **+30 dB**
- **4096 frames** for each modulation-SNR combination
- **1024 complex IQ samples** per frame
- Input frame shape:

```text
(1024, 2)
```

- Total number of frames:

```text
2,555,904
```

---

## Directory Structure

After downloading the dataset, your directory should look like:

```text
datasets/
├── README.md
├── GOLD_XYZ_OSC.0001_1024.hdf5
└── GOLD_XYZ_OSC.0001_1024.hdf5.zip    (optional)
```

---

## Notes

- Do **not** rename the dataset file.
- The notebook expects the dataset to be located inside the `datasets/` directory.
- The implementation reads the dataset directly from the HDF5 file.
- No additional metadata files are required, since the modulation labels and SNR values are stored within the HDF5 dataset (`X`, `Y`, and `Z`).
- During preprocessing, each IQ frame is converted from

```text
(1024, 2)
```

to the MCNet input format

```text
(2, 1024, 1)
```

before being fed into the neural network.
