
# 🎓 MMU FYP Student Dashboard (Dr Subar Edition)

A Streamlit-based **Student FYP Dashboard** project that provides an end-to-end workflow for:
- loading datasets
- training ML/DL models
- evaluating results (Accuracy / Precision / Recall / F1)
- comparing deployment feasibility (estimated latency + energy)
- exporting deployable artifacts (e.g., `.joblib`, `.tflite`, zipped packages)

This dashboard is generated and set up using a single installer script: `install_fyp.sh`.

---

## About the Dashboard

The dashboard is designed for FYP students to run repeatable experiments through a consistent pipeline. After installation, it creates a ready-to-run project folder (`Student_FYP/`) containing:
- Streamlit app entry (`app.py`)
- Training / Display / Download pages (`pages/`)
- Core training & evaluation pipeline (`core/`)
- Example datasets (`data/`)
- Deployment build outputs (`deployment_builds/`)
- Python virtual environment (`venv/`)

---

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- Linux/macOS terminal (recommended)

> ⚠️ IMPORTANT: **Do NOT run the installer using `sudo`.** Run as a normal user.

---

## Installation

### 1) Create the installer file
Create a file named `install_fyp.sh`:

```bash
nano install_fyp.sh
````


### 2) Run the installer (no sudo)

```bash
bash install_fyp.sh
```

---

## Interactive Setup

During installation, you will be asked:

1. **Project Title**

* Example: `IoT Application for Tracking Farm Animals`

2. **Custom Algorithms (optional, comma-separated)**

* Example: `GRU, BiLSTM, XGBoost-Tiny`
* Leave blank if you do not have custom algorithms.

After that, the installer will generate `Student_FYP/` and install all required Python packages into `venv/`.

---

## Launch the Dashboard

After you see **"✅ INSTALLATION COMPLETE"**, run:

```bash
cd Student_FYP
source venv/bin/activate
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## How to Use

### 1) 🧪 Training Lab

Use this page to:

* select dataset
* choose algorithms to train (classic ML + deep learning)
* set parameters (epochs, batch size, estimators, etc.)
* start training and auto-evaluation

### 2) 📊 Display Page

Use this page to:

* view charts for Accuracy / Precision / Recall / F1
* compare algorithms
* see edge profiling results (estimated **Latency (ms)** and **Energy (mJ)**)

### 3) 📥 Download Page

Use this page to:

* auto-select the best model (typically highest F1)
* download the packaged deployment artifacts from `deployment_builds/`

---

## Output Artifacts

After training, outputs are typically saved into:

* `results/metrics/` (evaluation metrics, summaries)
* `results/checkpoints/` (saved models/checkpoints)
* `deployment_builds/` (final deployment artifacts)

Common deployment artifacts:

* `<Algorithm>.joblib` (sklearn models)
* `<Algorithm>.tflite` (TensorFlow Lite models, if exported)
* `<Algorithm>_Deploy.zip` (packaged deployment bundle)

---

## Project Structure (Generated)

```text
Student_FYP/
├── app.py
├── requirements.txt
├── venv/
├── data/
│   └── *.csv
├── pages/
│   ├── 1_🧪_Training_Lab.py
│   ├── 2_📊_Display_Page.py
│   └── 3_📥_Download_Page.py
├── core/
│   ├── config_user.py
│   ├── registry.py
│   ├── trainer.py
│   ├── profiler.py
│   └── packager.py
├── results/
│   ├── metrics/
│   └── checkpoints/
└── deployment_builds/
    ├── README.txt
    ├── <Algorithm>.joblib
    ├── <Algorithm>.tflite
    └── <Algorithm>_Deploy.zip
```

---

## Troubleshooting

### Streamlit command not found

Make sure you activated the environment:

```bash
source venv/bin/activate
```

### Port already in use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

### Installation failed / missing packages

Re-run the installer (no sudo):

```bash
bash install_fyp.sh
```


