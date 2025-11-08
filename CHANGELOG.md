# Changelog

All notable changes to the Employee Attrition Prediction project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-08

### Added
- Initial project setup with modular architecture
- Data loading utilities (`src/load_data.py`)
- Comprehensive preprocessing pipeline (`src/preprocessing.py`)
- Class imbalance handling with SMOTE (`src/imbalance_handling.py`)
- Model training and visualisation utilities (`src/train_model.py`)
- Complete end-to-end Jupyter notebook (`notebooks/01_attrition_modeling.ipynb`)
- Enhanced package structure with proper `__init__.py`
- Comprehensive README with setup instructions
- Requirements file with exact version specifications
- Project structure documentation
- Development automation with Makefile
- Professional documentation (CONTRIBUTING.md, DEPLOYMENT.md)

### Features
- Load and preprocess IBM HR Analytics Employee Attrition dataset
- Handle categorical encoding with meaningful feature names
- Address class imbalance using SMOTE oversampling
- Train logistic regression model for attrition prediction
- Generate feature importance and correlation visualisations
- Modular, reusable code architecture
- Professional package structure for easy imports

### Technical Details
- Python 3.11 compatibility
- Jupyter notebook environment
- scikit-learn, pandas, numpy, matplotlib, seaborn integration
- Proper error handling and logging
- Clean separation of concerns across modules