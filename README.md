# Employee Attrition Prediction

A data science project for analyzing and predicting employee attrition using machine learning techniques.

## Project Overview

This project analyses HR employee attrition data to identify patterns and build predictive models that can help organisations understand and reduce employee turnover. The project is structured with modular Python scripts for data processing and machine learning, along with comprehensive Jupyter notebooks for analysis and visualisation.

## Dataset

**Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
**License:** DbCL (Database Contents License) v1.0  
**Attribution:** Fictional dataset created by IBM Data Scientists

The dataset `WA_Fn-UseC_-HR-Employee-Attrition.csv` contains 1,470 employee records with 35 features including:
- **Demographics**: Age, Gender, Marital Status  
- **Education**: Education Level (1-5 ordinal), Education Field (6 categories)
- **Job Details**: Department (3 categories), Job Role (9 categories), Job Level, Business Travel
- **Performance**: Performance Rating, Job Satisfaction, Work-Life Balance ratings
- **Compensation**: Monthly Income, Stock Option Level, Salary Hike Percentage
- **Career**: Years in Current Role, Years with Current Manager, Total Working Years
- **Work Environment**: Environment Satisfaction, Overtime, Distance from Home

**Target Variable**: Attrition (Yes/No) - whether an employee has left the company (16.1% attrition rate).

See [data/DATA_ATTRIBUTION.md](data/DATA_ATTRIBUTION.md) for complete dataset information and proper citation.

## Project Structure

```
employee_attrition_prediction/
├── data/
│   ├── raw/                           # Original dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv)
│   ├── processed/                     # Cleaned and preprocessed data
│   └── DATA_ATTRIBUTION.md           # Dataset license and attribution
├── notebooks/
│   └── 01_attrition_modeling.ipynb   # Complete modeling workflow
├── src/
│   ├── __init__.py                   # Package initialisation
│   ├── load_data.py                  # Data loading utilities
│   ├── preprocessing.py              # Data preprocessing and encoding pipeline
│   ├── imbalance_handling.py         # Class imbalance handling with SMOTE
│   ├── train_model.py               # Model training and evaluation
│   └── __pycache__/                 # Python bytecode cache
├── models/                           # Directory for saved model artifacts
├── outputs/
│   └── figures/                      # Generated plots and visualisations
├── requirements.txt           # Python dependencies
├── Makefile                   # Development automation tasks
├── CHANGELOG.md               # Version history and changes
├── CONTRIBUTING.md            # Development guidelines
├── DEPLOYMENT.md              # Deployment checklist and guide
├── LICENSE                    # MIT License
└── README.md                 # This file
```

## Quick Start

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd employee_attrition_prediction
   
   # Create and activate virtual environment (recommended)
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run the analysis:**
   - **Jupyter Notebook**: Open `notebooks/01_attrition_modeling.ipynb` in Jupyter and run all cells
   - **Direct Python**: Import and use the modular functions in your scripts

3. **Using the package functions:**
   ```python
   from src import load_attrition_data, preprocess_attrition_data, complete_analysis_workflow
   
   # Load and process data
   df = load_attrition_data('path/to/data.csv')
   df_processed = preprocess_attrition_data(df)
   
   # Run complete analysis
   results = complete_analysis_workflow(X, y, visualise=True)
   ```

> **💡 Why Virtual Environment?**  
> Using a virtual environment isolates this project's dependencies from your system Python, preventing version conflicts and ensuring reproducible results. It's considered a Python best practice for all projects.

## Modular Components

### Core Scripts

- **`src/load_data.py`**: Centralised data loading functionality
  - `load_attrition_data()`: Loads and returns the HR dataset
  
- **`src/preprocessing.py`**: Comprehensive data preprocessing and encoding
  - `preprocess_attrition_data()`: Complete preprocessing pipeline with proper categorical encoding
  - `one_hot_encode_columns()`: One-hot encoding for categorical text variables  
  - `binary_encode_columns()`: Binary encoding for Yes/No type variables
  - **Categorical Features**: BusinessTravel, Department, EducationField, JobRole, MaritalStatus (24 one-hot features)
  - **Ordinal Features**: Education (1-5), satisfaction ratings (1-4), job levels maintained as numeric
  - **Feature Engineering**: Creates 51 processed features from 35 original features

- **`src/imbalance_handling.py`**: Class imbalance handling
  - `handle_imbalance()`: Applies SMOTE for balanced training data

- **`src/train_model.py`**: Model training and evaluation
  - `train_logistic_regression()`: Trains and evaluates logistic regression model

### Notebook Workflow

**`notebooks/01_attrition_modeling.ipynb`** provides a complete end-to-end workflow:

1. **Data Loading & Exploration**
   - Load dataset using modular script
   - Display basic statistics and structure

2. **Data Preprocessing**
   - **Categorical Encoding Strategy**:
     - **One-Hot Encoding**: BusinessTravel (3), Department (3), EducationField (6), JobRole (9), MaritalStatus (3)
     - **Binary Encoding**: Attrition, OverTime, Gender (Yes/No, Male/Female types)
     - **Ordinal Preservation**: Education (1-5), satisfaction ratings, job levels kept as numeric
   - **Feature Engineering**: Transforms 35 features into 51 processed features
   - **Data Quality**: Removes constant columns, handles missing values
   - Creates meaningful column names for business interpretation

3. **Class Imbalance Handling**
   - Apply SMOTE to balance training data
   - Maintains test set distribution

4. **Model Training**
   - Train logistic regression model
   - Cross-validation and performance metrics

5. **Feature Analysis & Visualisation**
   - Feature importance analysis with meaningful names
   - Correlation heatmap showing relationship strengths
   - Business-interpretable visualisations

## Key Features

- **Comprehensive Categorical Encoding**: Proper handling of all categorical variables
  - **24 One-Hot Features**: BusinessTravel, Department, EducationField, JobRole, MaritalStatus
  - **Ordinal Features**: Education levels and satisfaction ratings preserved as numeric
  - **Binary Features**: Yes/No and gender variables appropriately encoded
- **Enhanced Feature Engineering**: 35 → 51 features with meaningful business names
- **Class Imbalance Handling**: SMOTE implementation for balanced training (16.1% → 50% attrition)
- **Interactive Analysis**: Complete Jupyter notebook workflow for exploration and visualisation
- **Modular Design**: Reusable components for preprocessing, modeling, and evaluation
- **Business Interpretation**: Clear, descriptive feature names for stakeholder communication
- **Professional Structure**: Complete package with comprehensive documentation

## Analysis Workflow
## Analysis Workflow

The project follows a systematic approach to employee attrition prediction:

1. **Data Loading & Exploration**
   - Load HR dataset with 1,470 employee records
   - Examine data structure, missing values, and distributions
   - Understand categorical vs numerical features

2. **Data Preprocessing**
   - **Comprehensive Categorical Encoding**:
     - **One-Hot Encoding**: 5 categorical text variables → 24 binary features
       - BusinessTravel: Non-Travel, Travel_Frequently, Travel_Rarely
       - Department: Human Resources, Research & Development, Sales  
       - EducationField: 6 fields including Life Sciences, Medical, Marketing
       - JobRole: 9 roles from Sales Executive to Research Director
       - MaritalStatus: Divorced, Married, Single
     - **Binary Encoding**: Yes/No and Male/Female variables (3 features)
     - **Ordinal Preservation**: Education (1-5), satisfaction ratings maintained as numeric
   - **Feature Scaling**: StandardScaler for continuous numerical features
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

6. **Visualisation & Reporting**
   - Feature importance plots with meaningful names
   - Correlation heatmaps showing feature relationships
   - Business-friendly visualisations for stakeholder communication

## Key Technologies

- **Python 3.x** - Primary programming language
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebooks

## Results & Insights

The analysis provides comprehensive insights into key factors driving employee attrition:

**Feature Engineering Results:**
- **Original Dataset**: 1,470 employees × 35 features
- **Processed Dataset**: 1,470 employees × 51 features  
- **Categorical Encoding**: 5 text variables → 24 meaningful binary features
- **Target Distribution**: 16.1% attrition rate (237 of 1,470 employees)

**Key Analytical Capabilities:**
- **Education Analysis**: Both education level (1-5 ordinal) and field (6 categories)
- **Role-Specific Insights**: 9 distinct job roles with detailed attrition patterns
- **Department Comparison**: Human Resources, Research & Development, Sales analysis
- **Work-Life Factors**: Business travel, overtime, and satisfaction correlations
- **Career Progression**: Job levels, tenure, and promotion impact on retention

**Business Value:**
- Comprehensive categorical feature set for detailed segmentation analysis
- Meaningful feature names enable direct stakeholder communication
- Balanced dataset preparation improves minority class (attrition) prediction
- Modular code structure supports production deployment and maintenance

## Future Enhancements

- **Advanced Models**: Random Forest, XGBoost, Neural Networks for improved prediction
- **Feature Engineering**: Interaction features, polynomial features, advanced transformations  
- **Model Deployment**: REST API, containerization, cloud deployment pipeline
- **Real-time Prediction**: Streaming data processing and live prediction capabilities
- **Interactive Dashboard**: Web-based visualization and prediction interface
- **A/B Testing Framework**: Model comparison and performance monitoring
- **Automated Retraining**: ML pipeline with data drift detection and model updates

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
