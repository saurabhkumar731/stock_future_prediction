import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

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

# ================== LOAD MODELS ==================
lr_model = pickle.load(open("C:/Users/Saurabh/Desktop/desktop/miniinfosys/mini_stock_linear.pkl", "rb"))
tree_model = pickle.load(open("C:/Users/Saurabh/Desktop/desktop/miniinfosys/mini_stock_tree.pkl", "rb"))
rf_model = pickle.load(open("C:/Users/Saurabh/Desktop/desktop/miniinfosys/mini_stock.pkl", "rb"))

scaler_X = pickle.load(open("C:/Users/Saurabh/Desktop/desktop/miniinfosys/scaler_X.pkl", "rb"))
scaler_Y = pickle.load(open("C:/Users/Saurabh/Desktop/desktop/miniinfosys/scaler_Y.pkl", "rb"))

# Penny stocks (India)
penny_stocks = {
    "Suzlon Energy": "SUZLON.NS",
    "Vodafone Idea": "IDEA.NS",
    "Yes Bank": "YESBANK.NS",
    "South Indian Bank": "SOUTHBANK.NS",
    "Alok Industries": "ALOKINDS.NS",
    "RattanIndia Power": "RTNPOWER.NS",
    "Jaiprakash Power Ventures": "JPPOWER.NS",
    "Reliance Power": "RPOWER.NS",
    "UCO Bank": "UCOBANK.NS",
    "Punjab National Bank": "PNB.NS",
}

# ================== MAIN APP ==================
st.title("📊 Penny Stock Price Forecast & Analysis")
st.write(f"👋 Hello, **{st.session_state.username}** | ")
if st.button("Logout"):
    logout()

# Select stock
company_name = st.selectbox("Choose a penny stock company:", list(penny_stocks.keys()))
ticker = penny_stocks[company_name]

# Download data
data = yf.download(ticker, period="2y")
data.reset_index(inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Future prediction input
user_date = st.date_input("Enter future date to predict:", datetime(2025, 12, 25))
user_date = datetime.combine(user_date, datetime.min.time())
last_row = data.iloc[-1]

future_df = pd.DataFrame({
    "Date": [user_date],
    "Open": [last_row["Open"]],
    "High": [last_row["High"]],
    "Low": [last_row["Low"]],
    "Volume": [last_row["Volume"]],
    "Date_ordinal": [user_date.toordinal()]
})

future_dates = pd.date_range(start=data.index.max() + timedelta(days=1), end=user_date)
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Open": [last_row["Open"]] * len(future_dates),
    "High": [last_row["High"]] * len(future_dates),
    "Low": [last_row["Low"]] * len(future_dates),
    "Volume": [last_row["Volume"]] * len(future_dates)
})
forecast_df["Date_ordinal"] = forecast_df["Date"].map(datetime.toordinal)

# ================== MODEL SELECTION ==================
st.sidebar.header("⚙️ Model Selection Mode")
mode = st.sidebar.radio("Select Mode:", ["Single Model", "Comparison"])

# ================== SINGLE MODEL ==================
if mode == "Single Model":
    model_choice = st.sidebar.radio("Choose Model:", ["Linear Regression", "Decision Tree", "Random Forest"])
    model = {"Linear Regression": lr_model, "Decision Tree": tree_model, "Random Forest": rf_model}[model_choice]

    X_future = future_df[["Date_ordinal", "Open", "High", "Low", "Volume"]]
    pred_close = scaler_Y.inverse_transform(model.predict(scaler_X.transform(X_future)).reshape(-1, 1))[0][0]
    last_close = float(last_row["Close"])

    trend = "📈 Increase" if pred_close > last_close else "📉 Decrease"
    change_pct = ((pred_close - last_close) / last_close) * 100

    st.subheader(f"{company_name} Prediction ({model_choice})")
    st.write(f"📅 Predicted Closing Price on **{user_date.date()}**: ₹{pred_close:.2f}")
    st.write(f"Last Close: ₹{last_close:.2f}")
    st.write(f"Trend: {trend} ({change_pct:.2f}%)")

    # Forecast
    X_fcast = forecast_df[["Date_ordinal", "Open", "High", "Low", "Volume"]]
    forecast_df["Predicted_Close"] = scaler_Y.inverse_transform(model.predict(scaler_X.transform(X_fcast)).reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)
    ax.plot(forecast_df["Date"], forecast_df["Predicted_Close"], label=f"Forecast ({model_choice})", linestyle="--")
    ax.set_title(f"{company_name} Stock Forecast - {model_choice}")
    ax.legend()
    st.pyplot(fig)

# ================== COMPARISON ==================
else:
    X_future = future_df[["Date_ordinal", "Open", "High", "Low", "Volume"]]
    last_close = float(last_row["Close"])

    preds = {
        "Linear Regression": scaler_Y.inverse_transform(lr_model.predict(scaler_X.transform(X_future)).reshape(-1, 1))[0][0],
        "Decision Tree": scaler_Y.inverse_transform(tree_model.predict(scaler_X.transform(X_future)).reshape(-1, 1))[0][0],
        "Random Forest": scaler_Y.inverse_transform(rf_model.predict(scaler_X.transform(X_future)).reshape(-1, 1))[0][0]
    }

    result_df = pd.DataFrame({
        "Model": preds.keys(),
        "Predicted Price (₹)": preds.values(),
        "Change %": [(v - last_close) / last_close * 100 for v in preds.values()],
        "Absolute Error (₹)": [abs(v - last_close) for v in preds.values()]
    })

    st.subheader(f"{company_name} - Model Comparison on {user_date.date()}")
    st.table(result_df.round(2))

    best_model = result_df.loc[result_df["Absolute Error (₹)"].idxmin(), "Model"]
    st.success(f"✅ Best Model: **{best_model}** (closest to last close)")

    # Multi-model forecast
    X_fcast = forecast_df[["Date_ordinal", "Open", "High", "Low", "Volume"]]
    forecast_df["LR_Pred"] = scaler_Y.inverse_transform(lr_model.predict(scaler_X.transform(X_fcast)).reshape(-1, 1))
    forecast_df["Tree_Pred"] = scaler_Y.inverse_transform(tree_model.predict(scaler_X.transform(X_fcast)).reshape(-1, 1))
    forecast_df["RF_Pred"] = scaler_Y.inverse_transform(rf_model.predict(scaler_X.transform(X_fcast)).reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)
    ax.plot(forecast_df["Date"], forecast_df["LR_Pred"], "--", label="LR Forecast")
    ax.plot(forecast_df["Date"], forecast_df["Tree_Pred"], "--", label="Tree Forecast")
    ax.plot(forecast_df["Date"], forecast_df["RF_Pred"], "--", label="RF Forecast")
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
