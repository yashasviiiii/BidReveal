import requests
import time
import json
import os
from datetime import datetime

def fetch_orderbook(symbol="BTCUSDT", depth=100, interval=2, iterations=1000, save_path="data/orderbook_snapshots.json"):
    """
    Collects live order book data from Binance for a given symbol.
    Saves data as a list of snapshots with timestamps, bids, and asks.
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth}"
    snapshots = []

    print(f"📡 Collecting {iterations} order book snapshots for {symbol} every {interval} sec...")

    for i in range(iterations):
        try:
            response = requests.get(url)
            data = response.json()

            snapshot = {
                "timestamp": datetime.utcnow().isoformat(),
                "bids": data['bids'],
                "asks": data['asks']
            }

            snapshots.append(snapshot)
            print(f"✅ Snapshot {i + 1}/{iterations} collected at {snapshot['timestamp']}")
            time.sleep(interval)

        except Exception as e:
            print(f"⚠️ Error during snapshot {i + 1}: {e}")
            time.sleep(interval)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(snapshots, f, indent=2)

    print(f"\n📁 Order book data saved to: {save_path}")

if __name__ == "__main__":
    fetch_orderbook()
