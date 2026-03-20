import requests
import pandas as pd

BASE = "https://gamma-api.polymarket.com"

def get_polymarket_events(limit=100, offset=0, active=True, closed=False):
    params = {
        "limit": limit,
        "offset": offset,
        "active": str(active).lower(),
        "closed": str(closed).lower(),
    }
    r = requests.get(f"{BASE}/events", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

rows = []
offset = 0
page_size = 100

while True:
    events = get_polymarket_events(limit=page_size, offset=offset)
    if not events:
        break

    for ev in events:
        for m in ev.get("markets", []):
            rows.append({
                "platform": "polymarket",
                "series_ticker": ev.get("id"),
                "series_title": ev.get("title"),
                "market_ticker": m.get("id"),
                "market_title": m.get("question"),
                "status": "active" if m.get("active") and not m.get("closed") else "closed",
                "close_time": m.get("endDate"),
                "rules_text": ev.get("description"),
                "resolution_source": ev.get("resolutionSource"),
                "subtitle": ev.get("subtitle"),
            })

    if len(events) < page_size:
        break

    offset += page_size


if __name__ == "__main__":

    df = pd.DataFrame(rows)
    print(df.info())
    print(df.head())
    print(df["status"].value_counts(dropna=False))
    print(df["series_ticker"].nunique())
    print(df["market_ticker"].nunique())

    df.to_csv("Data/polymarket_markets.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to Data/polymarket_markets.csv")