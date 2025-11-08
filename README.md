# Employee Attrition Prediction

A data science project for analyzing and predicting employee attrition using machine learning techniques.

## Project Overview

This project analyzes HR employee attrition data to identify patterns and build predictive models that can help organizations understand and reduce employee turnover. The project is structured with modular Python scripts for data processing and machine learning, along with comprehensive Jupyter notebooks for analysis and visualization.

## Dataset

**Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
**License:** DbCL (Database Contents License) v1.0  
**Attribution:** Fictional dataset created by IBM Data Scientists

The dataset `WA_Fn-UseC_-HR-Employee-Attrition.csv` contains 1,470 employee records with 35 features including:
- **Demographics**: Age, Gender, Marital Status
- **Job Details**: Department, Job Role, Job Level, Years at Company
- **Performance**: Performance Rating, Job Satisfaction, Work-Life Balance
- **Compensation**: Monthly Income, Stock Option Level
- **Career**: Years in Current Role, Years with Current Manager
- **Other**: Distance from Home, Overtime, Travel Frequency

**Important Data Notes:**
- **EducationField**: Contains descriptive values (e.g., "Life Sciences", "Medical", "Technical Degree")
- **Education**: Contains numeric codes (1-5) representing education levels
- The project uses EducationField for meaningful feature interpretation in visualizations

**Target Variable**: Attrition (Yes/No) - whether an employee has left the company.

See [data/DATA_ATTRIBUTION.md](data/DATA_ATTRIBUTION.md) for complete dataset information and proper citation.

## Project Structure

```
employee_attrition_prediction/
├── data/
│   ├── raw/                           # Original dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv)
│   ├── processed/                     # Cleaned and preprocessed data
│   └── DATA_ATTRIBUTION.md           # Dataset license and attribution
├── notebooks/
│   ├── 01_attrition_modeling.ipynb   # Initial exploratory analysis
│   └── 02_attrition_modeling.ipynb   # Complete modeling workflow
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── load_data.py                  # Data loading utilities
│   ├── preprocessing.py              # Data preprocessing and encoding pipeline
│   ├── imbalance_handling.py         # Class imbalance handling with SMOTE
│   ├── train_model.py               # Model training and evaluation
│   └── __pycache__/                 # Python bytecode cache
├── models/                           # Directory for saved model artifacts
├── outputs/
│   └── figures/                      # Generated plots and visualizations
├── tests/
│   └── __init__.py                  # Test package initialization
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                 # This file
```

## Quick Start

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd employee_attrition_prediction
   pip install -r requirements.txt
   ```

2. **Run the complete workflow:**
   - Open `notebooks/02_attrition_modeling.ipynb` in Jupyter
   - Run all cells to execute the full pipeline from data loading to visualization

## Modular Components

### Core Scripts

- **`src/load_data.py`**: Centralized data loading functionality
  - `load_attrition_data()`: Loads and returns the HR dataset
  
- **`src/preprocessing.py`**: Data preprocessing and encoding
  - `one_hot_encode_columns()`: One-hot encoding for categorical variables
  - `binary_encode_columns()`: Binary encoding for high-cardinality categories
  - `preprocess_pipeline()`: Complete preprocessing workflow

- **`src/imbalance_handling.py`**: Class imbalance handling
  - `handle_imbalance()`: Applies SMOTE for balanced training data

- **`src/train_model.py`**: Model training and evaluation
  - `train_logistic_regression()`: Trains and evaluates logistic regression model

### Notebook Workflow

**`notebooks/02_attrition_modeling.ipynb`** provides a complete end-to-end workflow:

1. **Data Loading & Exploration**
   - Load dataset using modular script
   - Display basic statistics and structure

2. **Data Preprocessing**
   - Handle categorical encoding (one-hot for EducationField, binary for others)
   - Feature scaling and normalization
   - Creates meaningful column names for visualization

3. **Class Imbalance Handling**
   - Apply SMOTE to balance training data
   - Maintains test set distribution

4. **Model Training**
   - Train logistic regression model
   - Cross-validation and performance metrics

5. **Feature Analysis & Visualization**
   - Feature importance analysis with meaningful names
   - Correlation heatmap showing relationship strengths
   - Business-interpretable visualizations

## Key Features

- **Modular Design**: Reusable components for preprocessing, modeling, and evaluation
- **Meaningful Visualizations**: Feature names like "EducationField_Life Sciences" instead of generic codes
- **Class Imbalance Handling**: SMOTE implementation for balanced training
- **Business Interpretation**: Clear, descriptive feature names for stakeholder communication
- **Complete Workflow**: End-to-end pipeline from raw data to insights

## Analysis Workflow
## Analysis Workflow

The project follows a systematic approach to employee attrition prediction:

1. **Data Loading & Exploration**
   - Load HR dataset with 1,470 employee records
   - Examine data structure, missing values, and distributions
   - Understand categorical vs numerical features

2. **Data Preprocessing**
   - **Categorical Encoding**: 
     - One-hot encoding for EducationField (creates descriptive columns)
     - Binary encoding for other categorical variables
   - **Feature Scaling**: StandardScaler for numerical features
   - **Data Splitting**: Train-test split for model validation

3. **Class Imbalance Handling**
   - Apply SMOTE (Synthetic Minority Oversampling Technique)
   - Balance the training dataset while preserving test set distribution
   - Improve model performance on minority class (attrition cases)

4. **Model Training & Evaluation**
   - Train logistic regression model with balanced data
   - Evaluate performance using accuracy, precision, recall, F1-score
   - Cross-validation for robust performance estimation

5. **Feature Analysis & Insights**
   - **Feature Importance**: Identify key predictors of attrition
   - **Correlation Analysis**: Understand relationships between features
   - **Business Interpretation**: Generate actionable insights for HR teams

6. **Visualization & Reporting**
   - Feature importance plots with meaningful names
   - Correlation heatmaps showing feature relationships
   - Business-friendly visualizations for stakeholder communication

## Key Technologies

- **Python 3.x** - Primary programming language
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebooks

## Results & Insights

The analysis provides insights into key factors driving employee attrition, including:
- Job satisfaction and work-life balance impact
- Compensation and career development factors
- Demographic and role-specific patterns
- Actionable recommendations for HR interventions

## Future Enhancements

- Additional model algorithms (Random Forest, XGBoost)
- Advanced feature engineering techniques
- Model deployment pipeline
- Real-time prediction capabilities
- Interactive dashboard development

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
