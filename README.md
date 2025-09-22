# Trend Insight: Stock Forecasting & Portfolio Optimization


## Project Overview 📈

This project leverages Long Short-Term Memory (LSTM) networks to predict stock prices and construct optimized investment portfolios. By analyzing historical stock data and various technical indicators, the model forecasts future stock prices and evaluates the accuracy of these predictions. The ultimate goal is to create a risk-minimized portfolio that achieves a desired return, providing a data-driven approach to investment strategy.

---

## Features ✨

* **Data Collection**: Fetches historical stock data from Yahoo Finance.
* **Data Preprocessing**: Cleans, normalizes, and reshapes data for LSTM model input.
* **Feature Engineering**: Utilizes technical indicators like **RSI, VWAP, EMA, and MACD** to enhance predictive accuracy.
* **Model Training**: Trains an LSTM model on historical data to forecast future closing prices.
* **Trend Prediction and Analysis**: Forecasts future stock prices and analyzes trend direction for the last 50 days.
* **Portfolio Optimization**: Constructs a portfolio of selected stocks with minimized risk (variance) for a target return.
* **Visualization**: Plots actual vs. predicted stock prices for clear visual analysis of model performance.

---

## Technical Stack 🛠️

* **Python 3.x**
* **Jupyter Notebook**
* **Libraries**:
    * Pandas
    * NumPy
    * Scikit-learn
    * TensorFlow / Keras
    * Matplotlib
    * Seaborn
    * yfinance
    * Joblib
    * Tabulate
    * SciPy
    * ta

---

## Methodology ⚙️

### 1. Data Collection and Preprocessing

* Historical daily closing prices for a list of stocks are downloaded from **Yahoo Finance**.
* The data for each stock is individually normalized using **MinMaxScaler** from Scikit-learn to scale the values between 0 and 1. This helps the LSTM model converge more effectively.
* The dataset is split into a **95% training set** and a **5% testing set**.

### 2. LSTM Model for Price Prediction

* An LSTM network is employed to capture long-term dependencies in the stock price data. The model is trained to predict the next day's closing price based on the **past 100 days** of data.
* The model architecture consists of two LSTM layers with **128 and 64 units**, respectively, followed by a Dense output layer.
* The model's performance is evaluated using the **Root Mean Square Error (RMSE)**.

### 3. Trend Analysis

* The project analyzes the directional trend (up or down) of both the actual and predicted stock prices.
* The trend for the **last 50 days** is compared between the predicted and actual values to count the number of matching trend predictions.

### 4. Portfolio Optimization

* This project extends beyond simple price prediction to create an optimized, risk-averse investment portfolio.
* The optimization is performed on the following stocks:
    * TCS.NS
    * NESTLEIND.NS
    * TITAN.NS
    * ASTRAL.BO
    * TATAPOWER.NS
    * SUZLON.NS
    * HINDPETRO.BO
    * HDFCBANK.NS
    * INFY.NS
    * ULTRACEMCO.NS
    * MAHLIFE.BO
    * ADANIGREEN.BO
* The objective is to **minimize the portfolio's variance (risk)** while ensuring it meets a specified return threshold.
* The optimization ensures that the sum of the weights of all stocks in the portfolio is equal to 1, with all weights being non-negative.

---

## Results 📊

For each stock analyzed, the project provides:
* The **RMSE** of the LSTM model's predictions.
* A **comparative plot** of actual vs. predicted stock prices.
* The number of **matching trend predictions** for the last 50 days.
* The **predicted closing price** for the next trading day.
* An **optimized portfolio** with calculated weights for each stock, designed to balance risk and return.

---
