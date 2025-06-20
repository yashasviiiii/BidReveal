import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from PIL import Image
import os
import time

# ✅ Load the model
model = joblib.load("xgboost_model.pkl")

# ✅ Label map
label_map = {0: "down", 1: "flat", 2: "up"}

st.set_page_config(page_title="Live BTC Orderbook Predictor", layout="wide")
st.title("📈 Live BTCUSDT Order Book Predictor")
st.caption("Predicts market move (up/down/flat) using live Binance order book data.")

# ✅ Fetch latest orderbook from Binance
@st.cache_data(ttl=2)
def fetch_orderbook(symbol="BTCUSDT", depth=5):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth}"
    try:
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

# ✅ Feature extractor (copied from features.py)
def extract_features(orderbook):
    best_bid = float(orderbook['bids'][0][0])
    best_ask = float(orderbook['asks'][0][0])
    bid_volume = sum(float(bid[1]) for bid in orderbook['bids'])
    ask_volume = sum(float(ask[1]) for ask in orderbook['asks'])
    spread = best_ask - best_bid
    return pd.DataFrame([{
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume
    }])


# ✅ Live spoofing detection logic
def detect_live_spoofing(curr_orderbook, prev_orderbook, cancel_threshold=3.0, price_move_threshold=0.005):
    try:
        curr_bid_price = float(curr_orderbook["bids"][0][0])
        curr_bid_qty = float(curr_orderbook["bids"][0][1])
        prev_bid_price = float(prev_orderbook["bids"][0][0])
        prev_bid_qty = float(prev_orderbook["bids"][0][1])

        curr_ask_price = float(curr_orderbook["asks"][0][0])
        prev_ask_price = float(prev_orderbook["asks"][0][0])

        sudden_disappear = (
            prev_bid_price == curr_bid_price and
            (prev_bid_qty - curr_bid_qty) > cancel_threshold
        )

        price_reaction = abs(curr_ask_price - prev_ask_price) / prev_ask_price > price_move_threshold

        spoof_flag = sudden_disappear and price_reaction
        return spoof_flag
    except Exception as e:
        return False

# ✅ Main Streamlit app logic
if st.button("🔄 Refresh Prediction"):
    prev_orderbook = fetch_orderbook()
    time.sleep(2)
    curr_orderbook = fetch_orderbook()

    if curr_orderbook and prev_orderbook:
        features = extract_features(curr_orderbook)
        pred = model.predict(features)[0]
        label = label_map[int(pred)]

        st.metric("🕒 Timestamp", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        st.metric("💰 Best Bid", features['best_bid'][0])
        st.metric("💰 Best Ask", features['best_ask'][0])
        st.metric("🔮 Prediction", label.upper())
        st.dataframe(features)

        # 🚨 Spoofing alert section
        if detect_live_spoofing(curr_orderbook, prev_orderbook):
            st.error("🚨 Spoofing behavior detected in the order book!")
        else:
            st.success("✅ No spoofing detected.")

        # 📊 Feature importance
        st.subheader("📊 Feature Importance")
        image_path = os.path.join("data", "feature_importance.png")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, caption="XGBoost Feature Importance", use_column_width=True)
            except Exception as e:
                st.error(f"🚫 Failed to load image: {e}")
        else:
            st.warning("⚠️ Feature importance image not found. Please run the training script first.")
