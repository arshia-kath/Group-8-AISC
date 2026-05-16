# CLAUDE.md — Group 8 AISC: Deepfake Detection System

## Project Overview

A deepfake detection pipeline built on **FaceForensics++ C23** (face-swap videos).
The system extracts handcrafted features from face crops and trains a logistic regression ensemble to classify faces as real (0) or fake (1).

Three detection modules:
- **Module 1** — EAR blink analysis (`main.py` + `src/preprocessing/`)
- **Module 2** — JPEG compression artifact score (`artifact_module.py`)
- **Module 3** — FFT frequency anomaly + Laplacian texture score → ensemble (`src/freq_analysis/` + `ensemble.py`)

---

## Run Order

```bash
# Step 1 — extract face crops from the FF++ C23 videos already on disk
python inspect_dataset.py

# Step 2 — extract features, train logistic regression, evaluate
python ensemble.py
```

`main.py` is a standalone demo for Module 1 (video loading + face detection). It is independent of the Steps 1/2 pipeline above.

---

## File Structure

```
Group-8-AISC/
│
├── inspect_dataset.py       Step 1 — extract face crops from FF++ C23 videos
├── ensemble.py              Step 2 — feature extraction, training, evaluation
├── artifact_module.py       Module 2 — JPEG recompression artifact scorer
├── main.py                  Module 1 demo — video loading + face detection
├── download_data.py         Helper — Kaggle dataset downloader (kagglehub)
│
├── src/
│   ├── preprocessing/       Module 1 helpers
│   │   ├── face_detector.py     Haar cascade face crop
│   │   ├── frame_extracter.py   Frame generator from cv2.VideoCapture
│   │   └── video_loader.py      cv2.VideoCapture wrapper
│   │
│   └── freq_analysis/       Module 3 feature extractors
│       ├── anomaly_scorer.py    fft_anomaly_score() — 0-1 FFT score
│       ├── fft_extractor.py     FFT primitives (grayscale, log-mag, mask)
│       ├── frequency_analyzer.py  Batch scoring + visualise_spectrum()
│       ├── texture_scorer.py    laplacian_score() — sharpness/texture score
│       └── utils.py             load_face_image(), resize_to_square()
│
├── data/
│   ├── FaceForensics++_C23/   Source videos — DO NOT MODIFY
│   │   ├── original/          1000 real YouTube face videos (.mp4)
│   │   ├── Deepfakes/         1000 autoencoder face-swap videos (.mp4)
│   │   ├── Face2Face/         1000 expression-transfer videos (.mp4)
│   │   ├── FaceSwap/          1000 geometry face-swap videos (.mp4)
│   │   └── csv/               FF++ metadata CSVs
│   ├── real/frames/           Extracted real face crops (224x224 JPEGs)
│   ├── fake/frames/           Extracted fake face crops (224x224 JPEGs)
│   ├── manifest.csv           Image list: file_path, label, video_id, source
│   ├── module3_features.csv   Per-image features: ear, artifact, fft, laplacian
│   ├── plots/                 roc_curve.png, precision_recall.png
│   └── visualizations/        FFT spectrum side-by-sides, artifact examples
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Key Configuration

**`inspect_dataset.py`** — controls dataset size and quality:
```python
TARGET_PER_CLASS = 200   # face crops to extract per class
FRAMES_PER_VIDEO = 4     # frames sampled from each video
MIN_BRIGHTNESS   = 40    # reject frames darker than this
MIN_FACE_FRAC    = 0.04  # reject if face < 4% of frame area
REAL_SRC = "data/FaceForensics++_C23/original"
FAKE_SRC = "data/FaceForensics++_C23/Deepfakes"
```

To switch manipulation type, change `FAKE_SRC` to one of:
`Deepfakes` / `Face2Face` / `FaceSwap` / `NeuralTextures` / `DeepFakeDetection`

**`ensemble.py`** — controls training:
```python
# train_ensemble() uses GroupShuffleSplit(test_size=0.20)
# grouped by video_id — no identity leakage between train/val
# LogisticRegression(class_weight="balanced")
```

---

## Features Used

| Feature | Source | Signal on FF++ C23 |
|---|---|---|
| `ear` | Module 1 stub (0.5) | None — constant until integrated |
| `artifact` | JPEG recompression delta | Very weak (Δ ≈ 0.002) |
| `fft` | FFT peripheral energy | Weak (Δ ≈ 0.012) |
| `laplacian` | Laplacian variance / 3000 | Moderate (Δ ≈ 0.06) |

All features are in [0, 1]. StandardScaler is applied before LogisticRegression so the model learns the correct direction and magnitude for each feature.

---

## Manifest Format

`data/manifest.csv` columns:
- `file_path` — relative path to the JPEG face crop
- `label` — `0` = real, `1` = fake
- `video_id` — e.g. `real_000`, `fake_042` — used for GroupShuffleSplit
- `source_dataset` — `FaceForensics++_C23/original` or `.../Deepfakes`

---

## Known Limitations

**FF++ C23 is deliberately hard.** The C23 H.264 quality setting smooths out the GAN/autoencoder generation artifacts that JPEG and FFT scores are designed to detect. Expected AUC for handcrafted features on C23 is 0.50–0.70.

**EAR is stubbed.** `ear_score = 0.5` in `extract_all_features()` until Module 1 is integrated. The logistic regression correctly assigns it zero weight.

**Face detection fallback.** Frames where the Haar cascade finds no face pass the `MIN_BRIGHTNESS` and `MIN_FACE_FRAC` quality checks but may still be non-ideal crops. These are skipped (not saved) rather than using center-crop fallback.

---

## Environment

- Python 3.13, Windows 11
- Virtual environment: `.venv/` (run `.\.venv\Scripts\activate` before any `pip` command)
- Key dependencies: `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`, `datasets` (HuggingFace), `kagglehub`

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Module Integration Status

| Module | Status |
|---|---|
| Module 1 — EAR blink detection | Preprocessing helpers done; scoring stub in ensemble |
| Module 2 — JPEG artifact | Complete (`artifact_module.py`) |
| Module 3 — FFT + texture ensemble | Complete (`ensemble.py`) |
| Module 3 — video-level split | Complete (GroupShuffleSplit on video_id) |
