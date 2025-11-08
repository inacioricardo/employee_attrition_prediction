# Contributing to Employee Attrition Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/inacioricardo/employee_attrition_prediction.git
   cd employee_attrition_prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup**
   ```bash
   python run_analysis.py --help
   ```

## Code Structure

- `src/` - Core package modules
- `notebooks/` - Jupyter notebooks for analysis
- `data/` - Data files (not tracked in git)
- `models/` - Saved model artifacts
- `outputs/` - Generated outputs and visualisations

## Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

## Making Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes
3. Update documentation if needed
4. Commit with clear message: `git commit -m "Add feature description"`
5. Push branch: `git push origin feature-name`
6. Create pull request

## Documentation

- Update README.md for significant changes
- Add docstrings for new functions
- Update CHANGELOG.md with your changes
- Include examples in docstrings where helpful

## Questions?

Feel free to open an issue for questions or discussion about contributions.