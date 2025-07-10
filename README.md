# 📈 Stock Price Prediction

This project aims to predict the future stock prices of **Apple Inc. (AAPL)** using historical stock data. The project leverages **Long Short-Term Memory (LSTM)** networks to forecast future prices based on previous data, using technical analysis indicators such as **Moving Averages**, **RSI**, and **Bollinger Bands**.

---

## 📁 Dataset

The dataset used is historical stock data fetched from **Yahoo Finance** using the **yfinance** library:
- `AAPL` (Apple Inc.) stock data from **2010-01-01 to 2021-12-31**.
- Features used: `Open`, `High`, `Low`, `Close`, `Volume`.

---

## 📊 Technologies Used

- **Python** 🐍
- **Jupyter Notebook**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn** (Data Visualization)
- **Scikit-learn** (Modeling)
- **TensorFlow / Keras** (LSTM Model)
- **TA-Lib / ta** (Technical Analysis Indicators)

---

## ✅ Project Steps

### 1. **Data Collection**
- Download historical stock data using the **yfinance** library.
- Preprocess and clean the data (handling missing values, scaling features).

### 2. **Feature Engineering**
- Add **technical indicators** like **RSI**, **Bollinger Bands**, and **SMA** (Simple Moving Averages).
- Normalize the data using **MinMaxScaler**.

### 3. **Model Building & Training**
- Built an **LSTM model** for time-series forecasting.
- Trained the model using previous stock prices and technical indicators.

### 4. **Prediction**
- Used the trained model to predict future stock prices.
- Plotted the predicted vs actual stock prices.

### 5. **Visualization**
- Visualized the actual vs predicted stock prices.
- Plotted **Bollinger Bands**, **RSI**, and **Moving Averages** to enhance the understanding of stock price behavior.

---

## 📈 Model Performance

### LSTM Model:
- **Performance Metric**: Evaluated based on accuracy and prediction error.
- **Predicted vs Actual Stock Prices**: The model's predictions are compared with actual stock prices to measure accuracy.

---

## 📂 Files in this Repo

| File                     | Description                                |
|--------------------------|--------------------------------------------|
| `stock_price_prediction.ipynb`    | Main Jupyter notebook with code & plots    |
| `predicted_stock_prices.csv` | Final predicted stock prices for the test set|
| `train.csv`              | Historical stock data from Yahoo Finance   |
| `test.csv`               | Test dataset for stock price prediction   |

---

## ✍️ Author

**Muhammad Bin Mehmood**  
Data Science Intern – July 2025  
Arch Technologies

---

## 📧 Contact

For technical issues or feedback:  
📩 **fawada8110@gmail.com**

---
