import json
import numpy as np
import pandas as pd
import time

def load_snapshots(filepath="data/orderbook_snapshots.json"):
    with open(filepath, "r") as f:
        snapshots = json.load(f)
    return snapshots

def extract_features(snapshots, wall_threshold=10.0):
    features = []

    for snap in snapshots:
        bids = [[float(p), float(q)] for p, q in snap["bids"]]
        asks = [[float(p), float(q)] for p, q in snap["asks"]]

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        spread = best_ask - best_bid
        bid_volume = sum([q for _, q in bids[:5]])
        ask_volume = sum([q for _, q in asks[:5]])

        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)

        wall_bid = any(q > wall_threshold for _, q in bids[:5])
        wall_ask = any(q > wall_threshold for _, q in asks[:5])

        features.append({
            "timestamp": snap["timestamp"],
            "bid_ask_spread": spread,
            "order_book_imbalance": imbalance,
            "total_bid_volume": bid_volume,
            "total_ask_volume": ask_volume,
            "wall_bid": int(wall_bid),
            "wall_ask": int(wall_ask)
        })

    return pd.DataFrame(features)

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

def label_price_movement(snapshots, horizon=1, threshold=0.00001):
    labels = []
    for i in range(len(snapshots) - horizon):
        try:
            curr_price = float(snapshots[i]["asks"][0][0])
            future_price = float(snapshots[i + horizon]["asks"][0][0])
            delta = (future_price - curr_price) / curr_price

            if i < 10:  # Show debug for first few
                print(f"Index {i}: curr={curr_price}, future={future_price}, delta={delta}")

            if delta > threshold:
                labels.append("up")
            elif delta < -threshold:
                labels.append("down")
            else:
                labels.append("flat")
        except Exception as e:
            print(f"Error at {i}: {e}")
            labels.append("flat")

    labels += ["flat"] * horizon
    return labels

def detect_spoofing(snapshots, cancel_threshold=3.0, price_move_threshold=0.005):
    spoof_flags = []

    for i in range(1, len(snapshots) - 1):
        try:
            prev_bids = snapshots[i - 1]["bids"]
            curr_bids = snapshots[i]["bids"]
            future_asks = snapshots[i + 1]["asks"]

            prev_price, prev_qty = float(prev_bids[0][0]), float(prev_bids[0][1])
            curr_price, curr_qty = float(curr_bids[0][0]), float(curr_bids[0][1])

            sudden_disappearance = (prev_price == curr_price) and (prev_qty - curr_qty > cancel_threshold)

            current_ask = float(snapshots[i]["asks"][0][0])
            future_ask = float(future_asks[0][0])
            price_jump = abs(future_ask - current_ask) / current_ask > price_move_threshold

            spoof = int(sudden_disappearance and price_jump)
            spoof_flags.append(spoof)
        except Exception as e:
            print(f"Spoofing error at {i}: {e}")
            spoof_flags.append(0)

    spoof_flags = [0] + spoof_flags + [0]
    return spoof_flags

if __name__ == "__main__":
    snaps = load_snapshots()
    df = extract_features(snaps)
    print("✅ Features extracted.")

    labels = label_price_movement(snaps)
    df["label"] = labels

    spoof_flags = detect_spoofing(snaps)
    df["spoof_flag"] = spoof_flags

    df.to_csv("data/labeled_features.csv", index=False)
    print("✅ Labeled features saved to data/labeled_features.csv")

    print("\n🔍 Label Distribution:")
    print(pd.Series(labels).value_counts())
