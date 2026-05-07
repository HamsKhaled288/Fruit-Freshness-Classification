# Fruit Freshness Detection 🍎🍌🍊

A complete end-to-end Computer Vision project for binary image classification: **Fresh vs Rotten** fruits, built for an Image Processing university final project.

## Project Description

This project trains two deep learning models to classify fruit images as either fresh or rotten:
1. **CNN** — A custom 4-block convolutional network
2. **Transfer Learning** — MobileNetV2 with feature extraction + fine-tuning

Both models are evaluated and compared. A Flask web app allows live predictions via image upload.

---

## Folder Structure

```
project/
├── data/
│   ├── raw/               ← original dataset (fresh/ and rotten/)
│   └── split/
│       ├── train/fresh | rotten
│       ├── val/fresh   | rotten
│       └── test/fresh  | rotten
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_model_cnn.ipynb
│   ├── 03_model_transfer.ipynb
│   └── 04_evaluation.ipynb
├── models/
│   ├── CNN_model.h5
│   └── transfer_model.h5
├── results/
│   └── plots/             ← all saved figures
├── app/
│   ├── app.py
│   ├── templates/index.html
│   └── static/style.css
├── split_dataset.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Place your raw images under:
- `data/raw/fresh/`   — all fresh fruit images
- `data/raw/rotten/`  — all rotten fruit images

### 3. Split the Dataset

```bash
python split_dataset.py
```

This creates the `data/split/` folder with 70% train / 15% val / 15% test.

---

## Running the Notebooks

Run notebooks **in order** from the `notebooks/` folder:

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01_preprocessing.ipynb` | Corrupt image check, class balance, data generators, augmentation |
| 2 | `02_model_cnn.ipynb` | Train CNN, save model, plot curves |
| 3 | `03_model_transfer.ipynb` | MobileNetV2 feature extraction + fine-tuning |
| 4 | `04_evaluation.ipynb` | Load both models, metric comparison, confusion matrices, ROC curves |

```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

---

## Flask Web App

Launch the prediction web app:

```bash
python app/app.py
```

Then open your browser at **http://localhost:5000**.

Upload any fruit image and the app will return whether it is **Fresh** or **Rotten** with a confidence score.

---

## Results

After training you will find in `results/plots/`:

- `class_balance.png` — dataset class distribution
- `augmented_samples.png` — sample augmented images
- `CNN_accuracy.png` / `CNN_loss.png`
- `transfer_accuracy.png` / `transfer_loss.png`
- `CNN_confusion.png` / `transfer_confusion.png`
- `CNN_roc.png` / `transfer_roc.png`

And `results/comparison_table.csv` — a side-by-side comparison of both models.
