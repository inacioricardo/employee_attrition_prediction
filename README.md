# Employee Attrition Prediction

A data science project for analyzing and predicting employee attrition using machine learning techniques.

## Project Overview

This project analyzes HR employee attrition data to identify patterns and build predictive models that can help organizations understand and reduce employee turnover.

## Dataset

**Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
**License:** DbCL (Database Contents License) v1.0  
**Attribution:** Fictional dataset created by IBM Data Scientists

The dataset `WA_Fn-UseC_-HR-Employee-Attrition.csv` contains employee information including:
- Demographics (Age, Gender, etc.)
- Job-related factors (Department, Job Role, etc.)
- Compensation (Monthly Income, Salary Hike, etc.)
- Work-life factors (Work-Life Balance, Job Satisfaction, etc.)
- Attrition status (Target variable)

See [data/DATA_ATTRIBUTION.md](data/DATA_ATTRIBUTION.md) for complete dataset information and proper citation.

## Project Structure

```
employee_attrition_prediction/
│
├── data/
│   ├── raw/                    # Original, immutable data
│   └── processed/              # Cleaned and processed data
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Data preprocessing functions
│   ├── feature_engineering.py # Feature creation functions
│   └── model.py               # Model training and evaluation
│
├── tests/                      # Unit tests
│   └── __init__.py
│
├── models/                     # Trained models
│
├── outputs/                    # Generated outputs
│   └── figures/               # Visualizations and plots
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── .env.example              # Environment variables template
└── README.md                 # This file
```


## Using Python Scripts in Power BI

To use the preprocessing and feature engineering scripts in Power BI, you must configure Power BI to use the Python executable from your virtual environment (venv):

1. **Locate your venv Python executable:**
   - On Windows, after creating your venv, the path will be:
     ```
     <your_project_path>\venv\Scripts\python.exe
     ```

2. **Configure Power BI to use this Python:**
   - Open Power BI Desktop
   - Go to **File > Options and settings > Options**
   - Under **Global > Python scripting**, set the **Detected Python home directories** to your venv's `python.exe` path above

3. **Install all required packages in your venv:**
   - Activate your venv in a terminal:
     ```powershell
     .\venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```

4. **Use Python scripts in Power BI:**
   - In Power BI, use the **Python script** data source and write code that imports your preprocessing and feature engineering modules from the `src` folder.
   - Example Power BI script:
     ```python
     import sys
     sys.path.append(r'<your_project_path>\\src')
     from preprocessing import DataPreprocessor
     from feature_engineering import FeatureEngineer
     import pandas as pd
     # Load your data and run pipeline as needed
     ```

**Note:** Any changes to your Python code or dependencies require you to restart Power BI or refresh the Python environment.



1. **Exploratory Data Analysis (EDA)**
   - Data loading and inspection
   - Missing value analysis
   - Univariate and bivariate analysis
   - Correlation analysis
   - Visualization of key patterns

2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling
   - Train-test split

3. **Feature Engineering**
   - Create derived features
   - Select relevant features
   - Handle class imbalance

4. **Model Development**
   - Train multiple models (Logistic Regression, Random Forest, XGBoost, etc.)
   - Hyperparameter tuning
   - Cross-validation

5. **Model Evaluation**
   - Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Feature importance analysis
   - Model interpretation with SHAP

6. **Deployment Preparation**
   - Save best model
   - Create prediction pipeline
   - Documentation

## Key Technologies

- **Python 3.11** (required)
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Model Interpretation**: SHAP
- **Notebooks**: Jupyter

## Next Steps

1. Run the exploratory data analysis notebook
2. Develop preprocessing pipeline
3. Engineer relevant features
4. Train and evaluate models
5. Interpret results and generate insights

## Contributing

This is a personal data science project. Feel free to fork and adapt for your needs.

## License

- **Code License:** MIT License (see [LICENSE](LICENSE) file)
- **Dataset License:** DbCL v1.0 - Database Contents License
  - Dataset Source: [Kaggle - IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
  - Created by IBM Data Scientists (Fictional dataset)
  - See [data/DATA_ATTRIBUTION.md](data/DATA_ATTRIBUTION.md) for complete attribution

---

**Created**: November 2025
