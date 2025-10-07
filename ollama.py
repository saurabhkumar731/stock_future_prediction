# -*- coding: utf-8 -*-
"""
Stock Forecasting App with ARIMA, LSTM, AI Insights & Chatbot (Ollama Integration + Comparison Mode)
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime, timedelta

# Time-series & ML
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



# ================== USER LOGIN ==================
USER_CREDENTIALS = {"saurabh": "12345", "admin": "admin123"}

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

# ================== COMPANY LIST ==================
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
    "Larsen & Toubro": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "NTPC": "NTPC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Oil & Natural Gas Corporation (ONGC)": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Nestle India": "NESTLEIND.NS",
    "ITC Limited": "ITC.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Grasim Industries": "GRASIM.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
}

# ================== OLLAMA HELPER FUNCTION ==================
def query_ollama(model, prompt):
    try:
        url = "http://localhost:11434/api/chat"   # Ollama must be running locally
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            full_reply = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        full_reply += data["message"]["content"]
            return full_reply.strip()
        else:
            return f"⚠️ Ollama API error: {response.status_code}"
    except Exception as e:
        return f"⚠️ Ollama not reachable: {str(e)}"

# ================== MAIN APP ==================
st.title("🏢 MNC Stock Forecast with ARIMA, LSTM & AI Insights")
st.write(f"👋 Hello, **{st.session_state.username}** | ")
if st.button("Logout"):
    logout()

# Sidebar AI model selection
st.sidebar.header("🤖 AI Model Selection")
ai_model = st.sidebar.selectbox("Choose AI Model:", ["gemma3:4b", "mistral", "llama2", "None"])

# Select stock
company_name = st.selectbox("Choose a company:", list(mnc_stocks.keys()))
ticker = mnc_stocks[company_name]

# Download stock data
data = yf.download(ticker, period="5y")
data.reset_index(inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# ================== AI INSIGHT SECTION ==================
last_30 = data["Close"].tail(30)
price_change = float((last_30.iloc[-1] - last_30.iloc[0]) / last_30.iloc[0] * 100)

if ai_model != "None":
    user_prompt = f"Give a short 1-line financial insight about {company_name} stock price trend. Recent 30-day change: {price_change:.2f}%."
    ai_insight = query_ollama(ai_model, user_prompt)
else:
    ai_insight = "💡 Select an AI model to view stock insights."

st.subheader("📌 AI Stock Insight")
st.info(ai_insight)

# ================== FUTURE PREDICTION ==================
user_date = st.date_input("Enter future date to predict:", datetime(2025, 12, 25))
user_date = datetime.combine(user_date, datetime.min.time())
steps = (user_date - data.index.max()).days

st.sidebar.header("⚙️ Model Selection Mode")
mode = st.sidebar.radio("Select Mode:", ["Single Model", "Comparison"])

if steps > 0:
    if mode == "Single Model":
        model_choice = st.sidebar.radio("Choose Model:", ["ARIMA", "LSTM"])
        pred_close = None

        if model_choice == "ARIMA":
            arima_model = ARIMA(data["Close"], order=(5, 1, 0))
            arima_fit = arima_model.fit()
            forecast = arima_fit.get_forecast(steps=steps)
            forecast_values = forecast.predicted_mean
            pred_close = forecast_values.iloc[-1]

            last_close = float(data["Close"].iloc[-1])
            trend = "📈 Increase" if pred_close > last_close else "📉 Decrease"
            change_pct = ((pred_close - last_close) / last_close) * 100

            st.subheader(f"{company_name} Prediction ({model_choice})")
            st.write(f"📅 Predicted Closing Price on **{user_date.date()}**: ₹{pred_close:.2f}")
            st.write(f"Last Close: ₹{last_close:.2f}")
            st.write(f"Trend: {trend} ({change_pct:.2f}%)")

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

            last_close = float(data["Close"].iloc[-1])
            trend = "📈 Increase" if pred_close > last_close else "📉 Decrease"
            change_pct = ((pred_close - last_close) / last_close) * 100

            st.subheader(f"{company_name} Prediction ({model_choice})")
            st.write(f"📅 Predicted Closing Price on **{user_date.date()}**: ₹{pred_close:.2f}")
            st.write(f"Last Close: ₹{last_close:.2f}")
            st.write(f"Trend: {trend} ({change_pct:.2f}%)")

            future_dates = pd.date_range(start=data.index.max() + timedelta(days=1), periods=steps)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)
            ax.plot(future_dates, preds, color="green", label="LSTM Forecast", linewidth=2)
            ax.legend()
            st.pyplot(fig)

    # ============ NEW COMPARISON MODE ============
    elif mode == "Comparison" and lstm_model:
        # --- ARIMA ---
        arima_model = ARIMA(data["Close"], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        forecast = arima_fit.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        arima_pred = forecast_values.iloc[-1]

        # --- LSTM ---
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

        lstm_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        lstm_pred = lstm_preds[-1]

        last_close = float(data["Close"].iloc[-1])

        # Comparison Table
        st.subheader(f"📊 {company_name} Comparison: ARIMA vs LSTM")
        comp_df = pd.DataFrame({
            "Model": ["ARIMA", "LSTM"],
            "Predicted Close": [arima_pred, lstm_pred],
            "Change %": [((arima_pred - last_close) / last_close) * 100,
                         ((lstm_pred - last_close) / last_close) * 100],
            "Trend": ["📈 Increase" if arima_pred > last_close else "📉 Decrease",
                      "📈 Increase" if lstm_pred > last_close else "📉 Decrease"]
        })
        st.table(comp_df)

        # Combined Chart
        future_dates = pd.date_range(start=data.index.max() + timedelta(days=1), periods=steps)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data["Close"], label="Historical Close", linewidth=2)
        ax.plot(future_dates, forecast_values, color="red", label="ARIMA Forecast", linewidth=2)
        ax.plot(future_dates, lstm_preds, color="green", label="LSTM Forecast", linewidth=2)
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

# ================== CHATBOT SECTION (ARIMA + LSTM) ==================
import dateparser
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

st.header("💬 Stock Chatbot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_company" not in st.session_state:
    st.session_state.last_company = company_name  # initialize with sidebar-selected

user_query = st.text_area("Ask about stocks, finance, or comparisons:", height=120)

# ----------- Helper functions -----------

def extract_company_names(query, last_company=None):
    """Detect companies mentioned in the query and handle pronouns correctly."""
    found = []
    query_lower = query.lower()
    
    # Match explicit company names
    for name in mnc_stocks.keys():
        short_name = name.split("(")[0].strip().lower()
        if short_name in query_lower or name.lower() in query_lower:
            found.append(name)

    # Match pronouns like "it" or "this company"
    pronouns = ["it", "this company", "the selected company"]
    if last_company and any(p in query_lower for p in pronouns):
        found.append(last_company)
    
    # Always include sidebar-selected company if not already included
    if company_name not in found:
        found.insert(0, company_name)

    return list(set(found))

def extract_date(query):
    """Detect date in user query"""
    parsed = dateparser.parse(query)
    if parsed:
        return parsed.date()
    import re
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", query)
    if match:
        day, month, year = map(int, match.groups())
        return datetime.date(year, month, day)
    return None

def get_predicted_prices(ticker, target_date):
    """Return historical or predicted price using ARIMA + LSTM"""
    df = yf.download(ticker, period="5y")
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    last_close = float(df["Close"].iloc[-1])
    steps = (target_date - df.index.max().date()).days
    predictions = {}

    if steps <= 0:
        df_hist = df.loc[df.index <= pd.Timestamp(target_date), "Close"]
        predictions["Historical"] = float(df_hist.iloc[-1]) if not df_hist.empty else None
        predictions["ARIMA"] = None
        predictions["LSTM"] = None
        return predictions, last_close

    # ARIMA prediction
    try:
        arima_model = ARIMA(df["Close"], order=(5,1,0))
        arima_fit = arima_model.fit()
        forecast = arima_fit.get_forecast(steps=steps)
        predictions["ARIMA"] = float(forecast.predicted_mean.iloc[-1])
    except:
        predictions["ARIMA"] = None

    # LSTM prediction
    if lstm_model:
        scaler = MinMaxScaler(feature_range=(0,1))
        close_scaled = scaler.fit_transform(df["Close"].values.reshape(-1,1))
        seq_length = 60
        if len(close_scaled) < seq_length:
            predictions["LSTM"] = None
        else:
            last_seq = close_scaled[-seq_length:]
            preds = []
            for _ in range(steps):
                X_input = last_seq.reshape(1, seq_length, 1)
                pred_scaled = lstm_model.predict(X_input, verbose=0)
                preds.append(pred_scaled[0][0])
                last_seq = np.append(last_seq[1:], pred_scaled)
            predictions["LSTM"] = float(scaler.inverse_transform(np.array(preds).reshape(-1,1))[-1])
    else:
        predictions["LSTM"] = None

    return predictions, last_close

# ----------- Handle user query -----------

if st.button("Send") and user_query.strip():
    st.session_state.chat_history.append(("🧑 You", user_query.strip()))
    query = user_query.lower()

    # Extract companies from query (handle pronouns)
    companies_in_query = extract_company_names(query, last_company=st.session_state.get("last_company"))

    # Remove duplicates and ensure valid
    companies = [c for c in companies_in_query if c in mnc_stocks]

    # Extract date
    date_asked = extract_date(query)
    response = ""

    if companies and date_asked:
        # Fetch predictions for all companies
        prices_dict = {}
        for c in companies:
            ticker = mnc_stocks[c]
            preds, last_close = get_predicted_prices(ticker, date_asked)
            prices_dict[c] = {"preds": preds, "last_close": last_close}

        # Prepare comparison table
        table_data = []
        best_company = None
        best_price = -float("inf")

        for model in ["Historical", "ARIMA", "LSTM"]:
            row = {"Model": model}
            for c in companies:
                price = prices_dict[c]["preds"].get(model)
                last_close = prices_dict[c]["last_close"]
                if price is not None:
                    trend = "📈 Increase" if price > last_close else "📉 Decrease"
                    row[c] = f"₹{price:.2f} ({trend})"
                    # Track best
                    if price > best_price:
                        best_price = price
                        best_company = c
                else:
                    row[c] = "N/A"
            table_data.append(row)

        comp_df = pd.DataFrame(table_data)
        st.subheader(f"📊 Stock Price Comparison on {date_asked.strftime('%d-%b-%Y')}")
        st.table(comp_df)

        st.markdown(f"🏆 **Best predicted closing:** {best_company} → ₹{best_price:.2f}")
        st.session_state.last_company = companies[0]

    else:
        if not date_asked:
            response = "⚠️ Could not detect a valid date. Please use format dd-mm-yyyy."
        elif ai_model != "None":
            response = query_ollama(ai_model, f"Finance question: {user_query}")
        else:
            response = "⚠️ Please select an AI model for general finance queries."

        if response:
            st.session_state.chat_history.append(("🤖 AI", response))

# ----------- Display chat -----------

for role, msg in st.session_state.chat_history:
    if role == "🧑 You":
        st.markdown(f"**{role}:** {msg}")
    else:
        st.markdown(
            f"<div style='background-color:#f0f2f6;padding:8px;border-radius:8px;margin-bottom:4px'>{role}: {msg}</div>",
            unsafe_allow_html=True
        )
