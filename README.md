# 🧠 Train Your Model

## Overview

**Train Your Model** is a no-code machine learning platform built with Streamlit that empowers users to train, tune, and interpret ML models entirely through a browser-based UI — no Python expertise required. Users upload a CSV or Excel dataset, apply a rich suite of feature engineering transformations, select from a wide range of classification or regression algorithms, perform automated hyperparameter tuning, evaluate model performance, and download the trained model artifact. SHAP-based model explainability is built in to surface feature importance and individual prediction reasoning.

## Key Features

- **No-code interface** — Upload data and train models through a guided, tab-based UI
- **Extensive feature engineering** — Log, polynomial, binning, interaction, power (Yeo-Johnson), sqrt, exponential, frequency encoding, target encoding, TF-IDF, robust scaling, rank transformation, and more
- **Dimensionality reduction** — PCA, Kernel PCA, Truncated SVD, t-SNE, UMAP, and LDA
- **Multiple imputation strategies** — Mean, median, KNN, and iterative imputation
- **15+ ML algorithms** — Logistic/Linear Regression, Random Forest, SVM, KNN, Decision Tree, Gradient Boosting, AdaBoost, XGBoost, Naive Bayes, LDA, QDA, and ensemble stacking
- **Automated hyperparameter tuning** — GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV, and Optuna
- **Model evaluation** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix (classification); R², MAE, RMSE, MSE (regression)
- **SHAP explainability** — Summary plots, feature importance bar charts, force plots, and decision plots; falls back to permutation importance when SHAP is unavailable
- **Model export** — Download trained models as versioned `.pkl` files

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.x |
| **UI Framework** | Streamlit, streamlit-lottie |
| **ML / Modeling** | scikit-learn, XGBoost |
| **Hyperparameter Tuning** | Optuna |
| **Explainability** | SHAP |
| **Dimensionality Reduction** | scikit-learn (PCA, t-SNE, LDA), umap-learn |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib |
| **Serialization** | pickle |

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/vanix056/Train-Your-Model.git
cd Train-Your-Model

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r src/requirements.txt
```

## Usage

```bash
# Launch the Streamlit application
cd src
streamlit run app.py
```

The app opens in your default browser at `http://localhost:8501`.

### Workflow

1. **Data Upload** — Upload a `.csv` or `.xlsx` file and preview the first rows
2. **Feature Engineering** — Apply transformations, encoding, and dimensionality reduction
3. **Model Training & Tuning** — Select problem type (Classification / Regression), configure preprocessing, choose a model, optionally enable hyperparameter tuning, and click **Train Model**
4. **Evaluate** — Review metrics displayed after training
5. **Explain** — Click **Explain Model** to generate SHAP visualizations
6. **Export** — Download the trained model as a versioned `.pkl` file

## Project Structure

```
Train-Your-Model/
├── src/
│   ├── app.py               # Entry point — launches the Streamlit app
│   ├── functionality.py     # Main GUI logic: tabs, feature engineering, training, SHAP
│   ├── util.py              # Core ML utilities: preprocessing, training, evaluation, tuning, SHAP
│   └── requirements.txt     # Python dependencies
├── data/
│   ├── heart.csv            # Sample heart disease dataset
│   ├── diabetes.csv         # Sample diabetes dataset
│   └── diabetes excel.xlsx  # Excel version of diabetes dataset
└── trained_model/           # Directory where trained .pkl models are saved
```

## Model Architecture

The platform supports two problem types with the following algorithm catalogue:

**Classification**
- Logistic Regression, Random Forest, SVM (SVC / LinearSVC / NuSVC), Extra Trees, AdaBoost, Gradient Boosting, XGBoost, Decision Tree, K-Nearest Neighbors, LDA, QDA, Gaussian NB, Bernoulli NB, Stacking Classifier

**Regression**
- Linear Regression, Random Forest, Decision Tree, K-Nearest Neighbors, Gradient Boosting, XGBoost, Stacking Regressor

All models are wrapped in optional scikit-learn `Pipeline` objects and serialized with `pickle` upon training completion.

## Dataset

The repository ships with two sample datasets to get started immediately:

| Dataset | File | Task | Target |
|---|---|---|---|
| Heart Disease | `data/heart.csv` | Binary Classification | `target` (0 / 1) |
| Diabetes | `data/diabetes.csv` | Binary Classification | `Outcome` (0 / 1) |

Any CSV or Excel file can be uploaded at runtime. The expected format is a tabular dataset with a clearly labeled target column.

## Training

```bash
# Start the app and use the UI to train
streamlit run src/app.py
```

Training steps performed internally:
1. Data upload and column sanitization
2. Optional feature engineering transformations
3. Train/test split (80/20, `random_state=42`)
4. Numeric imputation + scaling / categorical imputation + one-hot encoding
5. Optional hyperparameter search
6. Model fit and serialization to `trained_model/<name>_v<version>.pkl`

## Evaluation Metrics

| Problem Type | Metrics |
|---|---|
| **Classification** | Accuracy, Precision (weighted), Recall (weighted), F1 Score (weighted), Confusion Matrix |
| **Regression** | R², MAE, RMSE, MSE |

## Configuration

No environment variables are required to run the app locally. The following paths are resolved automatically at runtime:

| Path | Description |
|---|---|
| `src/` | Working directory for source modules |
| `trained_model/` | Output directory for serialized model files (created automatically) |

Hyperparameter grids are editable in-app via a JSON text area. Default grids are provided per algorithm and can be modified before tuning.

## UI Features

- **Animated header** powered by Lottie
- **Tabbed layout**: Data Upload → Feature Engineering → Model Training & Tuning
- **Interactive column selectors** for all transformation options
- **Progress bar** during model training
- **In-app SHAP visualizations**: summary plot, bar chart, force plot, decision plot
- **One-click model download** as a binary `.pkl` file

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`)
2. Keep changes focused and well-tested
3. Follow existing code style and module structure
4. Open a pull request with a clear description of the change and its motivation

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed and maintained by **[vanix056](https://github.com/vanix056)**.  
Contributions and feedback are welcome via GitHub Issues and Pull Requests.
