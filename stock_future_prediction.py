# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 2025
@author: Saurabh
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Time-series & ML
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ================== USER LOGIN SECTION ==================
USER_CREDENTIALS = {
    "saurabh": "12345",
    "admin": "admin123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Welcome {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

if not st.session_state.logged_in:
    login()
    st.stop()

# ================== LOAD LSTM MODEL ==================
try:
    lstm_model = load_model("C:/Users/Saurabh/Desktop/stock_website/lstm_stock_model.keras")
except:
    lstm_model = None
    st.warning("⚠️ LSTM model file not found. Please ensure 'lstm_stock_model.keras' is available.")

# ================== COMPANY LIST (MNCs) ==================
mnc_stocks = {
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Wipro": "WIPRO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tech Mahindra": "TECHM.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
}

# ================== MAIN APP ==================
st.title("🏢 MNC Stock Forecast with ARIMA & LSTM")
st.write(f"👋 Hello, **{st.session_state.username}** | ")
if st.button("Logout"):
    logout()

# Select stock
company_name = st.selectbox("Choose a company:", list(mnc_stocks.keys()))
ticker = mnc_stocks[company_name]

# Download stock data
data = yf.download(ticker, period="5y")
data.reset_index(inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Future prediction input
user_date = st.date_input("Enter future date to predict:", datetime(2025, 12, 25))
user_date = datetime.combine(user_date, datetime.min.time())
steps = (user_date - data.index.max()).days

# ================== MODEL SELECTION ==================
st.sidebar.header("⚙️ Model Selection Mode")
mode = st.sidebar.radio("Select Mode:", ["Single Model", "Comparison"])

# ================== SINGLE MODEL ==================
if mode == "Single Model":
    model_choice = st.sidebar.radio("Choose Model:", ["ARIMA", "LSTM"])
    pred_close = None

    if model_choice == "ARIMA":
        arima_model = ARIMA(data["Close"], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        forecast = arima_fit.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        pred_close = forecast_values.iloc[-1]

        # Compute trend first
        last_close = float(data["Close"].iloc[-1])
        trend = "📈 Increase" if pred_close > last_close else "📉 Decrease"
        change_pct = ((pred_close - last_close) / last_close) * 100

        # Show prediction info ABOVE the graph
        st.subheader(f"{company_name} Prediction ({model_choice})")
        st.write(f"📅 Predicted Closing Price on **{user_date.date()}**: ₹{pred_close:.2f}")
        st.write(f"Last Close: ₹{last_close:.2f}")
        st.write(f"Trend: {trend} ({change_pct:.2f}%)")

        # Plot
        st.header("📊 Historical vs Predicted Price (ARIMA)")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)
        future_dates = pd.date_range(start=data.index.max() + timedelta(days=1), periods=steps)
        ax.plot(future_dates, forecast_values, color="red", label="ARIMA Forecast", linewidth=2)
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "LSTM" and lstm_model:
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
        seq_length = 60
        preds = []
        last_sequence = close_scaled[-seq_length:]

        for _ in range(steps):
            X_input = last_sequence.reshape(1, seq_length, 1)
            pred_scaled = lstm_model.predict(X_input, verbose=0)
            preds.append(pred_scaled[0][0])
            last_sequence = np.append(last_sequence[1:], pred_scaled)

        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        pred_close = preds[-1]

        # Compute trend first
        last_close = float(data["Close"].iloc[-1])
        trend = "📈 Increase" if pred_close > last_close else "📉 Decrease"
        change_pct = ((pred_close - last_close) / last_close) * 100

        # Show prediction info ABOVE the graph
        st.subheader(f"{company_name} Prediction ({model_choice})")
        st.write(f"📅 Predicted Closing Price on **{user_date.date()}**: ₹{pred_close:.2f}")
        st.write(f"Last Close: ₹{last_close:.2f}")
        st.write(f"Trend: {trend} ({change_pct:.2f}%)")

        # Plot
        st.header("📊 Historical vs Predicted Price (LSTM)")
        future_dates = pd.date_range(start=data.index.max() + timedelta(days=1), periods=steps)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)
        ax.plot(future_dates, preds, color="green", label="LSTM Forecast", linewidth=2)
        ax.legend()
        st.pyplot(fig)

# ================== COMPARISON MODE ==================
else:
    preds = {}
    future_dates = pd.date_range(start=data.index.max() + timedelta(days=1), periods=steps)

    # ARIMA
    try:
        arima_model = ARIMA(data["Close"], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        forecast = arima_fit.get_forecast(steps=steps)
        preds["ARIMA"] = forecast.predicted_mean.iloc[-1]
        arima_forecast = forecast.predicted_mean
    except:
        preds["ARIMA"] = np.nan
        arima_forecast = None

    # LSTM
    if lstm_model:
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
        seq_length = 60
        pred_list = []
        last_sequence = close_scaled[-seq_length:]

        for _ in range(steps):
            X_input = last_sequence.reshape(1, seq_length, 1)
            pred_scaled = lstm_model.predict(X_input, verbose=0)
            pred_list.append(pred_scaled[0][0])
            last_sequence = np.append(last_sequence[1:], pred_scaled)

        pred_list = scaler.inverse_transform(np.array(pred_list).reshape(-1, 1)).flatten()
        preds["LSTM"] = pred_list[-1]
        lstm_forecast = pred_list
    else:
        preds["LSTM"] = np.nan
        lstm_forecast = None

    last_close = float(data["Close"].iloc[-1])
    result_df = pd.DataFrame({
        "Model": preds.keys(),
        "Predicted Price (₹)": preds.values(),
        "Change %": [(v - last_close) / last_close * 100 if not np.isnan(v) else np.nan for v in preds.values()]
    })

    # Show comparison table ABOVE graph
    st.subheader(f"{company_name} - Model Comparison on {user_date.date()}")
    st.table(result_df.round(2))

    best_model = result_df.iloc[result_df["Change %"].abs().idxmin()]["Model"]
    st.success(f"✅ Best Model: **{best_model}** (closest to last close)")

    # Plot comparison
    st.header("📊 Historical vs Predicted Prices (Comparison)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)

    if arima_forecast is not None:
        ax.plot(future_dates, arima_forecast, label="ARIMA Forecast", linewidth=2, color="red")

    if lstm_forecast is not None:
        ax.plot(future_dates, lstm_forecast, label="LSTM Forecast", linewidth=2, color="green")

    ax.legend()
    st.pyplot(fig)

# ================== DATA VISUALIZATION ==================
st.sidebar.header("📊 Visualization Options")
viz_option = st.sidebar.radio("Select Visualization:", ["None", "Moving Averages", "Correlation Heatmap"])

if viz_option == "Moving Averages":
    st.header("📈 Moving Averages (SMA 10 & 50)")
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Close", linewidth=2)
    ax.plot(data.index, data["SMA_10"], label="SMA 10")
    ax.plot(data.index, data["SMA_50"], label="SMA 50")
    ax.legend()
    st.pyplot(fig)

elif viz_option == "Correlation Heatmap":
    st.header("🔗 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data[["Open", "High", "Low", "Close", "Volume"]].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
