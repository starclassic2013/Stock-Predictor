# Stock-Predictor
Stock Predictor used for AAPL as an example. results yield unacceptable precision and predictability.
# AAPL Next‑Day Direction Classifier

A small, single‑file Python project that downloads five years of Apple (AAPL) daily price data, engineers common technical features, trains a **logistic regression** classifier to predict whether the **next trading day’s return** will be positive (UP=1) or not (DOWN=0), and prints evaluation metrics plus a naive strategy backtest.

> ⚠️ Educational use only. This is **not** investment advice. Past performance does not guarantee future results.

---

## Contents

* [How it works](#how-it-works)
* [Setup](#setup)
* [Run](#run)
* [What you’ll see](#what-youll-see)
* [Features & labels](#features--labels)
* [Modeling choices](#modeling-choices)
* [Backtest logic](#backtest-logic)
* [Reproducibility tips](#reproducibility-tips)
* [Common issues](#common-issues)
* [Extending the script](#extending-the-script)
* [License](#license)

---

## How it works

1. **Download data**: Pulls 5 years of AAPL daily candles with `yfinance`.
2. **Clean columns**: Flattens MultiIndex columns if present and prefers `Adj Close` when available.
3. **Engineer features**: SMA(5), SMA(10), momentum over 3 days, 10‑day volatility (std of returns), and RSI(14).
4. **Create label**: `Target = 1` if **tomorrow’s** return > 0, else `0`.
5. **Train/test split (time‑aware)**: Last 20% of rows held out for testing (no shuffling).
6. **Pipeline**: `StandardScaler` → `LogisticRegression(max_iter=1000)`.
7. **Evaluate**: Accuracy, precision, recall, confusion matrix, full classification report.
8. **Toy strategy**: Enters long if model predicted UP **yesterday** (shifted by 1 to avoid lookahead) and compares mean daily return to buy‑and‑hold over the test window.
9. **Latest prediction**: Retrains on all data and prints the model’s predicted direction for the next day.

---

## Setup

### Requirements

* Python 3.10+
* Packages: `pandas`, `numpy`, `yfinance`, `scikit-learn`

### Install

```bash
# (Optional) create & activate a virtual environment
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

# Install deps
pip install pandas numpy yfinance scikit-learn
```

> If you already have a working data‑science environment (e.g., Anaconda), you can skip the venv lines and just `pip install` the packages.

---

## Run

Save the script as `aapl_nextday_classifier.py`, then run:

```bash
python aapl_nextday_classifier.py
```

The script prints metrics and the model’s **next‑day direction** prediction (1=UP, 0=DOWN) to stdout.

---

## What you’ll see

* **Metrics** for the out‑of‑sample (last 20%) period:

  * Accuracy, precision, recall
  * Confusion matrix
  * Detailed classification report (per‑class precision/recall/F1)
* **Strategy vs Buy\&Hold**:

  * Mean daily return of the naive signal‑following strategy
  * Mean daily return of buy‑and‑hold over the same test window
* **Latest prediction** for the next trading day.

> Note: Returns are **not** compounded here—means are simple averages for quick comparison.

---

## Features & labels

**Label**

* `Target` — 1 if **next day’s** percent change in price is > 0, else 0.

**Engineered features** (all derived from `Adj Close` when present, else `Close`):

* `SMA5`, `SMA10` — Simple moving averages (5 & 10 days).
* `Mom3` — 3‑day momentum (pct change over 3 days).
* `Vol10` — 10‑day rolling std dev of daily returns.
* `RSI14` — 14‑day Relative Strength Index (simple rolling mean version).
* `Return` — Same‑day pct change, included as a feature.

Rolling windows induce NaNs at the start; the script drops them before training.

---

## Modeling choices

* **Time‑series split**: Uses the first 80% for training and the last 20% for testing to avoid leakage.
* **Scaling**: Standardization improves optimization for linear models.
* **Classifier**: Logistic Regression is fast, interpretable, and a solid baseline.

Potential improvements:

* Add cross‑validation with `TimeSeriesSplit`.
* Try non‑linear models (e.g., Gradient Boosting, RandomForest, XGBoost).
* Hyperparameter tuning and feature selection.
* Add more robust RSI (Wilder’s smoothing) and additional features (MACD, Bollinger Bands, volume‑based signals).

---

## Backtest logic

* Prediction for **day t** is used as the **signal for day t+1** (`shift(1)`).
* The toy strategy’s daily return is `signal_t * Return_t` on the test set.
* This avoids lookahead bias but **does not** include costs, slippage, or risk controls.

To turn this into a proper backtest, add:

* Transaction costs & slippage
* Position sizing and risk limits
* Portfolio accounting and compounding

---

## Reproducibility tips

* `yfinance` can revise historical data; results may shift slightly over time.
* Fix a date range (e.g., start/end) for stable datasets.
* Pin package versions in `requirements.txt` for consistent behavior.

Example `requirements.txt`:

```
pandas>=2.0
numpy>=1.25
yfinance>=0.2
scikit-learn>=1.4
```

---

## Common issues

* **No `Adj Close` column**: The script falls back to `Close` automatically.
* **Convergence warnings**: Increase `max_iter` or scale features (already done here).
* **All‑nan features**: Ensure enough rows for rolling windows (min 14+ days for RSI).
* **Zeros in RSI denom**: A small epsilon (`1e-9`) prevents division‑by‑zero.

---

## Extending the script

* Swap ticker(s) and retrain (e.g., `"MSFT"`, `"SPY"`).
* Parameterize window lengths via CLI args or a config file.
* Log metrics to a CSV, and save the trained pipeline with `joblib`.
* Add plotting (price, indicators, signals) with `matplotlib`.
* Build a simple Streamlit app for interactive exploration.

---

## License

MIT (or choose your preferred OSS license).
