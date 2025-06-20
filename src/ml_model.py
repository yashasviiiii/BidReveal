import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import joblib

def extract_features_and_labels(json_path):
    with open(json_path, 'r') as f:
        snapshots = json.load(f)

    data = []

    for i in range(1, len(snapshots)):
        curr = snapshots[i]
        prev = snapshots[i - 1]

        try:
            best_bid = float(curr['bids'][0][0])
            best_ask = float(curr['asks'][0][0])
            spread = best_ask - best_bid
            bid_volume = sum(float(bid[1]) for bid in curr['bids'][:5])
            ask_volume = sum(float(ask[1]) for ask in curr['asks'][:5])
        except (IndexError, ValueError):
            continue

        try:
            prev_mid = (float(prev['bids'][0][0]) + float(prev['asks'][0][0])) / 2
            curr_mid = (best_bid + best_ask) / 2
        except:
            continue

        if curr_mid > prev_mid:
            label = 'up'
        elif curr_mid < prev_mid:
            label = 'down'
        else:
            label = 'flat'

        data.append({
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'label': label
        })

    return pd.DataFrame(data)

def balance_dataset(df, label_encoder):
    df['label_num'] = label_encoder.transform(df['label'])
    class_counts = df['label'].value_counts()
    min_count = class_counts.min()

    # Downsample all classes to match the smallest one
    balanced_frames = []
    for label in class_counts.index:
        subset = df[df['label'] == label]
        balanced = resample(subset, replace=False, n_samples=min_count, random_state=42)
        balanced_frames.append(balanced)

    df_balanced = pd.concat(balanced_frames)
    return df_balanced.sample(frac=1, random_state=42)  # shuffle

def train_model():
    json_path = os.path.join("data", "orderbook_snapshots.json")
    print("🔍 Reading and processing order book data...")

    df = extract_features_and_labels(json_path)

    print("\n📊 Original Label Distribution:")
    print(df["label"].value_counts())

    # Label encode
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # Save encoder
    joblib.dump(label_encoder, "data/label_encoder.pkl")

    # Decode back to strings for balancing
    df["label_str"] = label_encoder.inverse_transform(df["label"])

    # Balance the dataset
    df["label"] = df["label_str"]
    df_balanced = balance_dataset(df, label_encoder)

    print("\n📊 Balanced Label Distribution:")
    print(df_balanced["label"].value_counts())

    X = df_balanced[["best_bid", "best_ask", "spread", "bid_volume", "ask_volume"]]
    y = label_encoder.transform(df_balanced["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n🚀 Training XGBoost on balanced data...")
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    # Save model and encoder
    joblib.dump(model, "xgboost_model.pkl")
    print("✅ Model saved as 'xgboost_model.pkl'")

    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type='gain')
    plt.title("Feature Importances (Gain)")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    print("📊 Feature importance chart saved as 'data/feature_importance.png'")

    y_pred = model.predict(X_test)

    print("\n📈 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\n🧪 Prediction Class Distribution on Test Set:")
    print(pd.Series(label_encoder.inverse_transform(y_pred)).value_counts())

if __name__ == "__main__":
    train_model()


