# Financial Inclusion in Africa: MSc AI Project

## Academic Context

I built this project as part of my **MSc Artificial Intelligence** programme, specifically for the course unit **CSA 802: Systems and Data Integration**. The assignment required me to deliver an end-to-end practical data mining solution: acquiring or simulating a dataset, cleaning and exploring it, applying dimension reduction where appropriate, selecting a justified algorithm, and evaluating the resulting model. Beyond the analytics workflow, I also needed to produce an interactive interface that demonstrates how the insights can be deployed.

## Project Overview

To meet those objectives, I analysed the **Financial Inclusion in Africa** dataset from Kaggle. My goal was to predict bank account ownership across East African populations and surface the socio-economic factors that drive inclusion. The deliverables include:

- A reproducible data mining notebook (developed in Google Colab)
- A comprehensive written report (`improved_datamining_report.md`)
- A Streamlit web application (`app.py`) for live predictions and stakeholder-friendly storytelling
- Persisted model artefacts to support deployment (`models/` directory)

## Learning Outcomes

This project allowed me to practise and demonstrate:

- Integrating heterogeneous survey data into a clean analytical dataset
- Executing a structured CRISP-DM-style pipeline (ingestion → cleaning → EDA → modelling → deployment)
- Performing Principal Component Analysis to assess dimensionality and information redundancy
- Selecting **XGBoost** to address non-linear relationships, class imbalance, and interpretability needs
- Translating machine learning outputs into an interactive application suitable for policy and decision makers

## Repository Structure

```
Financial Inclusion in Africa Dataset Analysis and Algorithm Selection/
│
├── models/                          # Model artefacts exported from Colab
│   ├── xgb_model.pkl               # Trained XGBoost classifier
│   ├── scaler.pkl                  # StandardScaler fitted on training data
│   ├── label_encoders.pkl          # LabelEncoder for the target variable
│   ├── feature_names.pkl           # Ordered feature names for inference
│   ├── pca.pkl                     # PCA estimator (90% variance)
│   ├── onehot_columns.pkl          # One-hot encoded column names
│   ├── categorical_cols.pkl        # Original categorical column list
│   └── numerical_cols.pkl          # Numerical column list
│
├── data/
│   └── archive.zip                 # Kaggle dataset download (Train/Test/Metadata)
│
├── app.py                          # Streamlit user interface
├── improved_datamining_report.md   # Written analysis and algorithm justification
├── requirements.txt                # Python dependencies for the app
├── SETUP_GUIDE.md                  # Step-by-step instructions to place model files
└── README.md                       # You are here
```

## How to Reproduce My Environment

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Retrieve model artefacts from Google Colab

After running the notebook end-to-end in Google Colab, I used the built-in file browser to download all `.pkl` files. These files are required by the Streamlit application and must live in `models/`.

**Files to copy into `models/`:**
- `xgb_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_names.pkl`
- `pca.pkl`
- `onehot_columns.pkl`
- `categorical_cols.pkl`
- `numerical_cols.pkl`

If you need a refresher, `SETUP_GUIDE.md` documents two Colab download workflows (individual files or zipped export).

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The interface launches on http://localhost:8501/. From there you can explore what-if scenarios and examine the probability breakdown, feature importance, and confidence gauges that I designed for stakeholders.

## Data Mining Workflow (What I Implemented)

1. **Dataset acquisition & understanding** – Imported Kaggle’s survey-based dataset (23,524 records, mixed categorical & numerical features).
2. **Data cleaning** – Validated data types, handled duplicates (none detected), confirmed there were no missing values, and retained plausible outliers after contextual evaluation.
3. **Exploratory Data Analysis** – Conducted descriptive statistics, visualised distributions, and analysed cross-tabulated inclusion rates by country, education, job type, gender, and location. Documented key insights:
   - Education level: 3.9% (no education) → 57.0% (vocational)
   - Job type: 2.1% (no income) → 77.5% (government employment)
   - Cellphone access: 1.7% (no access) → 18.4% (has access)
4. **Dimension reduction** – Applied PCA after standardisation; 24 principal components explain ~90% variance, confirming redundancy but also the need for non-linear models.
5. **Algorithm selection** – Justified XGBoost over alternatives (Logistic Regression, Random Forest, SVM, Neural Networks, KNN) given the strong interactions, sparsity, and class imbalance (86% “No” vs 14% “Yes”).
6. **Model training & evaluation** – Tuned XGBoost with imbalance-aware parameters, benchmarked against Logistic Regression, and reported metrics (ROC AUC 0.85, F1 0.65, Accuracy 0.88). Performed stratified 5-fold cross-validation to validate stability.
7. **Interpretability** – Generated feature importance rankings and probability breakdowns. The Streamlit app surfaces these insights via interactive charts and explanation panels.
8. **Deployment artefacts** – Serialized the model and preprocessing steps so the app can run without rebuilding the pipeline.

## Streamlit Application Features

- **Guided input sidebar** – Organised into thematic blocks (geography, demographics, family structure, education/employment, technology access).
- **Modern UI theme** – Custom CSS and Plotly visuals built around the primary colour `#1aa1e5`, with accessible contrast using black and white accents.
- **Prediction dashboard** – Displays the classification, confidence score, probability gauge, and key contributing factors in real time.
- **Stakeholder storytelling** – Welcome screen summarises model performance, key insights, and usage steps for policy or NGO stakeholders.
- **Explainability widgets** – Interactive bar charts for class probabilities, model metrics, and feature importance help translate ML output into actionable intelligence.

## Model Performance Snapshot

| Metric        | Score |
|---------------|-------|
| ROC AUC       | ~0.85 |
| F1 Score      | ~0.65 |
| Accuracy      | ~0.88 |
| Precision     | ~0.72 |
| Recall        | ~0.58 |

These results confirm that XGBoost handles the complex interactions and imbalance far better than the Logistic Regression baseline that I initially implemented for comparison.

## Reflections & Future Work

- **Data integration**: The assignment strengthened my ability to integrate survey data with minimal preprocessing assumptions while maintaining domain context.
- **Fairness considerations**: Future iterations could incorporate bias audits or uplift modelling to explore equitable policy interventions.
- **Temporal analysis**: If longitudinal data becomes available, I would extend the pipeline to monitor inclusion trends and policy impact over time.
- **AutoML comparison**: Benchmarking against AutoML frameworks could further validate the selected hyperparameters and highlight opportunities for improvement.

## Dataset Reference

- **Dataset:** [Financial Inclusion in Africa (Kaggle)](https://www.kaggle.com/datasets/gauravduttakiit/financial-inclusion-in-africa)
- **Records:** 23,524 respondents from Kenya, Rwanda, Tanzania, and Uganda
- **Target Variable:** `bank_account` (Yes/No)

## Acknowledgements

I would like to thank the CSA 802 teaching team for providing the project brief and evaluation rubric that guided this work. The Kaggle community’s open dataset also made it possible to explore financial inclusion in a reproducible way.

## Author

**[Your Name]**  
MSc Artificial Intelligence – CSA 802 Systems and Data Integration  
Financial Inclusion in Africa – Data Mining Project

