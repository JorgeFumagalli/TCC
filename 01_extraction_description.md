# 📥 Data Extraction Notebook - Macroeconomic and Credit Risk Series (BCB/SGS)

This notebook/script performs the extraction of time series data from the Central Bank of Brazil (SGS API), with the goal of consolidating macroeconomic indicators and credit-related variables for further demand forecasting modeling.

## ✅ Extracted Series
- SELIC (monthly %)
- IBC-Br (with and without seasonal adjustment)
- IPCA (inflation index)
- Credit card default rates (total, revolving, installment)
- Income commitment and family indebtedness
- PIB real (real GDP, quarterly)
- Credit-to-GDP ratio (constructed manually from monthly credit volume and quarterly GDP)

The script performs:
1. Download of data via API (with fallback URLs)
2. Transformation to monthly frequency
3. Reconstruction of derived metrics (e.g., credit/GDP)
4. Saving results to `.csv` and `.xlsx` formats

📁 Output files:
- `macro_credito_bcb_com_credito_pib_reconstruido.csv`
- `macro_credito_bcb_com_credito_pib_reconstruido.xlsx`

🛠️ Technologies:
- Python, pandas, matplotlib, requests

🔍 This file should be placed in `/notebooks/01_extraction.ipynb` or as a script in `/scripts/01_extraction.py`.

✍️ Author: Jorge Luiz Fumagalli