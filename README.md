# STOCK-MARKET-PREDICTION-ANALYSIS
this project is based on different stocks for the course data science to see the change in trend of stock market.
We worked in a group of 3 in this project. I was assigned to do the data visualization and trends of stock while others cleaned the webscrapped data and applied prediction models. For this project we used LSTM model to predict our closing price for the stocks. Our model showed 98-99 percent of accuracy.

# Market Data EDA & Forecasting (AAPL, MSFT, TSLA, SBUX, BTC-USD)

A reproducible end-to-end workflow for:
1) **Ingest**: download daily data via `yfinance`
2) **Clean**: type fixes, duplicates, simple outlier handling
3) **Explore**: line charts, histograms, KDEs, boxplots, correlation heatmaps
4) **Model**: 
   - LSTM on **AAPL** (closing price sequence)
   - Linear Regression on **BTC-USD** (Close ~ Open, High, Low, Volume)
5) **Evaluate**: RMSE; plus a simple direction-of-change classification demo

> **Note**  
> This project is for learning/analysis only, **not trading advice**.

---

## ✨ Features

- Download & persist CSVs for: `AAPL`, `MSFT`, `TSLA`, `SBUX`, `BTC-USD`
- Consistent cleaning: date parsing, numeric casts, duplicate removal
- Simple outlier rule (mean ± 3σ or quantiles in places) for quick visuals
- Correlation + distribution plots, dividend/split scatter demos
- LSTM baseline on AAPL with train/val split and RMSE
- Linear model baseline on BTC-USD with RMSE & R²
- Saves cleaned CSVs for downstream work

---

## 📦 Environment

- Python 3.10+ recommended
- Key libs: `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` (for the LSTM)

**requirements.txt (minimal)**
yfinance
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow

yaml
Copy code

---

## 📁 Project structure

.
├─ data/
│ ├─ raw/ # optional: original pulls (yfinance)
│ └─ clean/ # cleaned CSVs saved here
├─ notebooks/
│ └─ eda_and_models.ipynb
├─ scripts/
│ ├─ fetch_and_save.py # yfinance pulls -> data/raw
│ ├─ clean_pipeline.py # cleaning, typing, outliers -> data/clean
│ └─ visualize_and_model.py
├─ README.md
├─ requirements.txt
└─ .gitignore

markdown
Copy code

**.gitignore**
python
pycache/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

data
data/raw/
data/clean/

misc
.DS_Store

yaml
Copy code

> If you’re working entirely in a single notebook, keep the structure simple and still save artifacts under `data/`.

---
## 📊 Reproducing the notebook results

Run cells in order. Plots include:

Trends: closing price time series

Distributions: hist/KDE, boxplot

Relationships: correlation heatmap; scatter vs. dividends/splits (categorical hue)

LSTM metrics: training progress + RMSE

Confusion matrix (toy): direction-of-change (increase vs decrease)

Note on seaborn warnings
When using categorical hues (e.g., Dividends, Stock Splits), ensure your palette length matches the number of unique categories; otherwise seaborn will warn and cycle colors.

## 🔧 Extending this repo

Add walk-forward CV for time series (rolling windows)

Use returns (log returns) instead of prices

Add exogenous features: moving averages, RSI/MACD, macro indicators

Compare more models: ARIMA/SARIMAX, Prophet, XGBoost/LightGBM, CNN-LSTM hybrids

Serve predictions via a simple FastAPI endpoint

## 🙌 Acknowledgements

Data via yfinance (Yahoo Finance)

Plots via matplotlib / seaborn

Models via scikit-learn and tensorflow
## 🚀 Quickstart

1) **Install deps**
```bash
pip install -r requirements.txt
Fetch & save

In notebook or script:

python
Copy code
import yfinance as yf
import pandas as pd

tickers = ["AAPL","MSFT","TSLA","SBUX","BTC-USD"]
for t in tickers:
    df = yf.Ticker(t).history(period="1d", start="2013-01-01", end="2024-01-07")
    df.to_csv(f"data/raw/{t}.csv")
Clean

Parse dates, cast numerics, remove duplicates

Rename Open→open_price, Close→close_price

Add average_price = (open_price + close_price)/2

(Simple) outlier handling for visual clarity

Outputs:

bash
Copy code
data/clean/cleanedAAPLstock.csv
data/clean/cleanedMSFTstock.csv
data/clean/cleanedTSLAstock.csv
data/clean/cleanedSBUXstock.csv
data/clean/cleanedBTC-USDstock.csv
Explore

Line charts: closing price over time

Histograms/KDEs for distribution

Boxplots for spread/outliers

Heatmaps for correlations

Modeling

LSTM (AAPL)
Univariate (closing price)

Windowed sequences (default time_step=100)

Train/validation split with train_test_split

Metrics: RMSE on train/val

Linear Regression (BTC-USD)
Features: open_price, High, Low, Volume

Target: close_price

Metrics: RMSE, R²

