# 🎓 Demand Forecasting: Comparing SVM, Machine Learning and Deep Learning Models

This repository contains the Final Project (TCC) developed for the MBA in Data Science and Analytics at USP/ESALQ.

## 🎯 Objective

The goal of this project is to compare the predictive performance of three approaches — **Support Vector Machines (SVM)**, **XGBoost Model**, and **Deep Learning architectures** — applied to demand forecasting using real operational data from the food industry.

## 🧱 Project Workflow

The project is structured into three main steps:

1. **Data Extraction**  
   - Load structured tables from raw sources (CSV, Excel, internal systems).
   - Merge, format, and save raw datasets for processing.

2. **Data Preprocessing**  
   - Clean and transform the data.
   - Feature engineering and preparation for modeling.

3. **Modeling & Evaluation**  
   - Apply and compare multiple algorithms:
     - Support Vector Regression (SVR)
     - XGBoost
     - Deep Learning (MLP, LSTM)
   - Evaluate performance using metrics:
     - MAE, RMSE, R², MAPE
   - Generate final comparison charts and insights.

## 📁 Project Structure

```
tcc-demand-forecasting/
│
├── notebooks/
│   ├── 01_extraction.ipynb         # Raw data extraction
│   ├── 02_preprocessing.ipynb      # Data cleaning and transformation
│   └── 03_model_comparison.ipynb   # Model training and evaluation
│
├── data/
│   ├── raw/                        # Raw input data
│   └── processed/                  # Cleaned data for modeling
│
├── outputs/
│   ├── metrics_table.csv           # Results of model evaluations
│   └── charts/                     # Graphs, residual plots, etc.
```

## ⚙️ Tools and Libraries

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- keras / tensorflow

## 📝 Notes

- This repository contains **synthetic or anonymized** data to comply with confidentiality requirements.
- The modeling phase focuses on performance and business applicability.

## 👤 Author

**Jorge Luiz Fumagalli**  
MBA in Data Science & Analytics - USP/ESALQ  
📧 jorgefumagalli@yahoo.com.br  
🔗 [LinkedIn]((https://www.linkedin.com/in/jorge-fumagalli-bb8975121/))
