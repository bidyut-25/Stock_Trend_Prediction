# Trend Insight: Stock Forecasting & Portfolio Optimization

## Project Overview
This project explores the effectiveness of LSTM models in predicting stock prices and understanding market trends. By analyzing historical data, the model attempts to forecast future stock prices and assess the accuracy of trend predictions across various stocks. Following the prediction phase, the project focuses on portfolio optimization to construct a less risky investment strategy by minimizing variance while achieving desired returns.

## Features
- **Data Loading:** Load data from Yahoo Finance.
- **Data Preprocessing:** Normalize and reshape stock data for LSTM input.
- **Model Training:** Train an LSTM model on historical stock data to predict future prices.
- **Prediction:** Forecast future stock prices and analyze trend predictions.
- **Trend Matching:** Compare predicted trends with actual trends and count matching trends.
- **Visualization:** Plot actual vs. predicted stock prices for visual analysis.
- **Portfolio Optimization:** Optimize a portfolio of selected stocks to minimize risk while meeting return expectations.

## Main Libraries Used
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Joblib
- Tabulate

## Data Collection
The data for this project is collected from Yahoo Finance, focusing on daily closing prices of various stocks.

## Model Training and Prediction

### Data Preprocessing:
- Normalize each stock's closing prices using `MinMaxScaler`.
- Split the data into training (95%) and testing (5%) sets.

### Model Architecture:
- The LSTM model consists of two LSTM layers with 128 and 64 units, respectively.
- The model is trained to predict the next day's closing price using the past 100 days of data.

### Prediction:
- After training, the model generates predictions on the test data.
- The performance is evaluated using root mean square error (RMSE).

## Trend Analysis
The project examines the trends (whether prices increase or decrease) of the actual and predicted stock prices:

### Trend Calculation:
- Trends for both actual and predicted prices are calculated based on the difference between consecutive days.

### Trend Matching:
- The trends for the last 50 days are compared, and the number of matching trends between actual and predicted values is counted.

## Portfolio Optimization
Beyond stock price prediction, the project also delves into portfolio optimization. The goal is to create an optimized portfolio that minimizes risk (variance) while meeting a specific return threshold. This optimization is conducted on the following stocks:
- TCS
- NESTLEIND
- TITAN
- ASTRAL
- TATAPOWER
- SUZLON
- HINDPETRO
- HDFCBANK
- INFY
- ULTRACEMCO
- MAHLIFE
- ADANIGREEN

The optimization ensures that the sum of weights equals 1, with all weights being non-negative, to strike a balance between risk and return.

## Results
For each stock, the project provides:
- The RMSE of the LSTM model.
- A plot comparing actual vs. predicted prices.
- The number of matching trends for the last 50 days.
- The predicted price for the next day.
- An optimized portfolio of the listed stocks.
