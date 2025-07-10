

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


stock_data = yf.download('AAPL', start='2010-01-01', end='2021-12-31')

stock_data = stock_data[['Close']] 

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data)

# 3. Create Dataset for Training the Model
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Features (previous 60 days)
        y.append(data[i, 0]) 
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)


X = X.reshape(X.shape[0], X.shape[1], 1)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # First LSTM Layer
model.add(LSTM(units=50, return_sequences=False))  # Second LSTM Layer
model.add(Dense(units=1))  # Output Layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 8. Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values back to original scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 9. Plot the results
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# 10. Save the predictions to a CSV file (Optional)
predictions_df = pd.DataFrame(predictions, columns=['Predicted Price'])
predictions_df.to_csv('predicted_stock_prices.csv', index=False)
print("Predictions for the test set saved to 'predicted_stock_prices.csv'.")
