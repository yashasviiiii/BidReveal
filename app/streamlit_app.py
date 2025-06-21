import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from PIL import Image
import os
import time
from collections import deque
import json

# ✅ Load model
model = joblib.load("xgboost_model.pkl")

# ✅ Label mapping
label_map = {0: "down", 1: "flat", 2: "up"}

# ✅ Rolling buffer
snapshot_buffer = deque(maxlen=5)

st.set_page_config(page_title="Live BTC Orderbook Predictor", layout="wide")
st.title("📈 Live BTCUSDT Order Book Predictor")
st.caption("Predicts market move (up/down/flat) using live Binance order book data.")

# ✅ Main toggle instead of sidebar
use_demo = st.checkbox("🧪 Use Spoofing Demo Dataset", value=False)

# ✅ Internally fixed detection parameters
cancel_threshold = 10.0
price_move_threshold = 0.001
volume_drop_threshold = 0.5

@st.cache_data(ttl=2)
def fetch_orderbook(symbol="BTCUSDT", depth=5):
    try:
        res = requests.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth}")
        return res.json()
    except Exception as e:
        st.error(f"Failed to fetch orderbook: {e}")
        return None

def extract_features(orderbook):
    best_bid = float(orderbook["bids"][0][0])
    best_ask = float(orderbook["asks"][0][0])
    bid_volume = sum(float(b[1]) for b in orderbook["bids"])
    ask_volume = sum(float(a[1]) for a in orderbook["asks"])
    spread = best_ask - best_bid
    return pd.DataFrame([{
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume
    }])

def compute_tension(features):
    spread = features["spread"][0]
    imbalance = (features["bid_volume"][0] - features["ask_volume"][0]) / (features["bid_volume"][0] + features["ask_volume"][0] + 1e-9)
    wall_bid = features["bid_volume"][0] > 10000
    wall_ask = features["ask_volume"][0] > 10000

    score = (1 / (spread + 1e-5)) * 0.4 + abs(imbalance) * 0.4 + (wall_bid or wall_ask) * 0.2
    norm = min(score / 10, 1.0)

    if norm > 0.7:
        return norm, "🔥 HIGH"
    elif norm > 0.4:
        return norm, "🟠 MEDIUM"
    else:
        return norm, "🟢 LOW"

def detect_live_spoofing(curr, prev, cancel_threshold, price_move_threshold):
    try:
        cbp, cbq = float(curr["bids"][0][0]), float(curr["bids"][0][1])
        pbp, pbq = float(prev["bids"][0][0]), float(prev["bids"][0][1])
        cap, pap = float(curr["asks"][0][0]), float(prev["asks"][0][0])

        sudden_disappear = pbp == cbp and (pbq - cbq) > cancel_threshold
        price_reaction = abs(cap - pap) / pap > price_move_threshold

        print("🚨 DEBUG SPOOF CHECK 🚨")
        print(f"prev_bid_price={pbp}, prev_bid_qty={pbq}")
        print(f"curr_bid_price={cbp}, curr_bid_qty={cbq}")
        print(f"cancel={sudden_disappear}")
        print(f"price_jump={abs(cap - pap) / pap}")

        return sudden_disappear and price_reaction
    except:
        return False

def detect_fake_liquidity(buffer, side="bids", volume_drop_threshold=0.5):
    if len(buffer) < 5:
        return []

    prev = buffer[0][side]
    curr = buffer[-1][side]
    prev_dict = {float(p): float(q) for p, q in prev}
    curr_dict = {float(p): float(q) for p, q in curr}

    fake_orders = []
    for price in prev_dict:
        if price in curr_dict:
            drop = prev_dict[price] - curr_dict[price]
            if drop > 0 and (drop / prev_dict[price]) > volume_drop_threshold:
                print(f"💡 FAKE ORDER DETECTED @ {price} → Drop: {drop}")
                fake_orders.append({
                    "Price": price,
                    "Initial Volume": round(prev_dict[price], 2),
                    "Final Volume": round(curr_dict[price], 2),
                    "Drop %": round((drop / prev_dict[price]) * 100, 1)
                })

    return fake_orders

# ✅ App logic
if st.button("🔄 Refresh Prediction"):
    if use_demo:
        try:
            with open("data/spoof_demo/spoof_snapshots.json", "r") as f:
                snaps = json.load(f)
                if len(snaps) >= 5:
                    snapshot_buffer.clear()
                    snapshot_buffer.extend(snaps[:5])
                    prev_orderbook = snapshot_buffer[-2]
                    curr_orderbook = snapshot_buffer[-1]
                else:
                    st.error("❌ Not enough demo snapshots.")
                    st.stop()
        except Exception as e:
            st.error(f"Failed to load spoof demo: {e}")
            st.stop()
    else:
        prev_orderbook = fetch_orderbook()
        time.sleep(2)
        curr_orderbook = fetch_orderbook()
        if curr_orderbook:
            snapshot_buffer.append(curr_orderbook)

    if curr_orderbook and prev_orderbook:
        features = extract_features(curr_orderbook)
        pred = model.predict(features)[0]
        label = label_map[int(pred)]

        tension_score, tension_level = compute_tension(features)
        st.subheader("📊 Market Tension Meter")
        st.metric("🔥 Market Tension", tension_level)
        st.progress(tension_score)

        st.metric("🕒 Timestamp", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        st.metric("💰 Best Bid", features["best_bid"][0])
        st.metric("💰 Best Ask", features["best_ask"][0])
        st.metric("🔮 Prediction", label.upper())
        st.dataframe(features)

        if detect_live_spoofing(curr_orderbook, prev_orderbook, cancel_threshold, price_move_threshold):
            st.error("🚨 Spoofing behavior detected!")
        else:
            st.success("✅ No spoofing detected.")

        st.subheader("🎯 Fake Liquidity Radar")
        fake_bids = detect_fake_liquidity(snapshot_buffer, "bids", volume_drop_threshold)
        fake_asks = detect_fake_liquidity(snapshot_buffer, "asks", volume_drop_threshold)

        if fake_bids:
            st.write("🚨 **Fake Bids Detected:**")
            st.dataframe(pd.DataFrame(fake_bids))
        if fake_asks:
            st.write("🚨 **Fake Asks Detected:**")
            st.dataframe(pd.DataFrame(fake_asks))
        if not fake_bids and not fake_asks:
            st.success("✅ No fake liquidity behavior detected.")

        st.subheader("📊 Feature Importance")
        image_path = os.path.join("data", "feature_importance.png")
        if os.path.exists(image_path):
            st.image(Image.open(image_path), caption="XGBoost Feature Importance", use_container_width=True)
        else:
            st.warning("⚠️ Feature importance image not found.")
