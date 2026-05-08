# Developer Survey ML Salary Estimator

This project builds an interpretable salary estimator for software developers using the Stack Overflow Developer Survey 2025. It filters the survey to US professional developers, engineers a compact set of experience, role, language, cloud, framework, and workplace features, trains a Ridge regression model on log salary, and serves predictions through a Streamlit app with SHAP explanations.

The Streamlit UI is branded as **DevSalary** and lets a user enter a developer profile, view an estimated annual salary, compare it with the survey distribution, and inspect which features pushed the estimate up or down.

## Project Structure

```text
.
|-- app.py                         # Streamlit salary estimator UI
|-- requirements.txt               # Python dependencies
|-- shap_analysis.ipynb            # SHAP explanation notebook
|-- data/
|   |-- survey_results_public.csv  # Raw Stack Overflow survey data
|   `-- survey_cleaned.csv         # Cleaned modeling dataset
|-- preprocessing/
|   |-- datasetshape.py            # Quick raw dataset inspection script
|   `-- preprocess_data.ipynb      # Data cleaning and feature engineering
|-- training/
|   |-- train.ipynb                # Model training and evaluation
|   `-- models/
|       |-- model.pkl              # Trained Ridge regression model
|       |-- scaler.pkl             # StandardScaler fitted on training data
|       |-- feature_names.pkl      # Feature order used by the model
|       |-- X_test_sc.csv          # Scaled test features
|       `-- y_test.csv             # Test target values
`-- reports/
    |-- shap_bar.png               # Global SHAP feature importance
    |-- shap_beeswarm.png          # SHAP beeswarm plot
    `-- shap_waterfall.png         # Example individual explanation
```

## What the Project Does

1. Loads the raw Stack Overflow Developer Survey 2025 public dataset.
2. Filters to professional developers in the United States with USD compensation.
3. Removes salary outliers by keeping annual salaries in the `$20,000` to `$400,000` range.
4. Encodes survey fields into numeric model features.
5. Adds engineered features such as squared experience, cloud breadth, high-value language count, web framework count, and interaction terms.
6. Trains a Ridge regression model to predict `log_salary`.
7. Converts predictions back to dollars with `exp(predicted_log_salary)`.
8. Uses SHAP to explain each prediction and generate global explanation plots.
9. Serves the model through a Streamlit UI.

## Dataset

The raw dataset used by this project is:

- `data/survey_results_public.csv`
- Shape: `49,191` rows x `172` columns

The cleaned modeling dataset is:

- `data/survey_cleaned.csv`
- Shape: `4,149` rows x `38` columns

Filtering steps from the preprocessing notebook:

- Professional developers only: `37,467` rows
- US respondents only: `5,531` rows
- USD currency only: `4,453` rows
- Salary range `$20k-$400k`: `4,149` rows

## Features

The final model uses `36` input features.

Core profile features:

- `years_code`
- `work_exp`
- `log_work_exp`
- `education`
- `org_size`
- `remote_work`
- `is_manager`

Role flags:

- `is_fullstack`
- `is_backend`
- `is_frontend`
- `is_ml`
- `is_devops`
- `is_mobile`

Language flags:

- `lang_python`
- `lang_rust`
- `lang_go`
- `lang_typescript`
- `lang_kotlin`
- `lang_scala`
- `lang_swift`

Cloud and framework flags:

- `cloud_aws`
- `cloud_azure`
- `cloud_gcp`
- `web_react`
- `web_nextjs`
- `web_fastapi`
- `web_django`
- `uses_ai_tools`

Engineered features:

- `work_exp_sq`
- `n_cloud`
- `n_hv_langs`
- `n_webframes`
- `exp_x_edu`
- `exp_x_orgsize`
- `manager_x_exp`
- `ml_x_cloud`

## Model

The training notebook uses:

- Model: `sklearn.linear_model.Ridge`
- Regularization: `alpha=10`
- Target: `log_salary`
- Split: `80%` train, `20%` test
- Scaling: `StandardScaler` fitted on the training set only
- Test rows: `830`
- Train rows: `3,319`

Saved artifacts:

- `training/models/model.pkl`
- `training/models/scaler.pkl`
- `training/models/feature_names.pkl`
- `training/models/X_test_sc.csv`
- `training/models/y_test.csv`

## Evaluation

Reported results from `training/train.ipynb`:

```text
MAE  : $45,107
MAPE : 33.9%
R2   : 0.283
```

Cross-validation:

```text
5-fold CV R2: 0.334 +/- 0.055
Folds: [0.262, 0.278, 0.356, 0.410, 0.362]
```

Top coefficients by absolute impact include:

- `log_work_exp`
- `org_size`
- `exp_x_edu`
- `work_exp_sq`
- `education`
- `cloud_aws`
- `is_fullstack`
- `remote_work`
- `years_code`
- `is_backend`
- `lang_go`

## Explainability

The project uses `shap.LinearExplainer`, which is appropriate for the Ridge regression model. SHAP values are computed on the scaled feature space and used in two places:

- `shap_analysis.ipynb` for global and example explanation plots.
- `app.py` for live per-profile salary impact explanations.

Generated reports:

- `reports/shap_bar.png`
- `reports/shap_beeswarm.png`
- `reports/shap_waterfall.png`

The notebook includes a sanity check confirming:

```text
baseline + SHAP sum = model prediction
```

## Streamlit App

Run the app from the project root:

```bash
streamlit run app.py
```

The app provides:

- Sidebar profile inputs for experience, education, company size, remote work, role, languages, cloud platforms, frameworks, and AI tool usage.
- Estimated annual salary in USD.
- Estimated gross monthly salary.
- Percentile against cleaned survey respondents.
- Difference from the survey median.
- SHAP-based feature impact chart.
- Salary distribution chart.
- Complete feature contribution table.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project depends on:

- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `matplotlib`
- `streamlit`

See `requirements.txt` for version constraints.

## Reproducing the Workflow

Run the notebooks in this order:

1. `preprocessing/preprocess_data.ipynb`
   - Loads the raw survey CSV.
   - Filters rows.
   - Cleans and encodes features.
   - Saves `data/survey_cleaned.csv`.

2. `training/train.ipynb`
   - Loads the cleaned data.
   - Splits and scales features.
   - Trains Ridge regression.
   - Evaluates the model.
   - Saves model artifacts to `training/models/`.

3. `shap_analysis.ipynb`
   - Loads the trained model and test data.
   - Computes SHAP values.
   - Saves plots to `reports/`.

4. `app.py`
   - Loads saved artifacts.
   - Serves predictions and explanations through Streamlit.


## Dataset Download link:
   https://survey.stackoverflow.co/

## Important Notes

- The model is trained only on US respondents with annual compensation between `$20,000` and `$400,000`.
- Predictions are estimates from survey data, not salary guarantees or financial advice.
- The app currently uses absolute Windows paths in `app.py` when loading data and model artifacts. If the project is moved to another machine or folder, update those paths or replace them with relative paths.
- The model has moderate explanatory power (`R2` around `0.28` on the test set), so it is best used for exploration and interpretability rather than exact salary prediction.
