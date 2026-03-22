import time
import requests
import pandas as pd
import os
import base64
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

print("Loading environment variables...")
load_dotenv()

KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY")
KALSHI_RSA_PRIVATE_KEY_STR = os.getenv("KALSHI_RSA_PRIVATE_KEY")

print(f"KALSHI_ACCESS_KEY found: {bool(KALSHI_ACCESS_KEY)}")
print(f"KALSHI_RSA_PRIVATE_KEY found: {bool(KALSHI_RSA_PRIVATE_KEY_STR)}")

# Initialize RSA Private Key
if KALSHI_RSA_PRIVATE_KEY_STR:
    # Handle escaped newlines from .env
    key_pem = KALSHI_RSA_PRIVATE_KEY_STR.replace('\\n', '\n')
    print("Initializing RSA key...")
    try:
        priv_key = serialization.load_pem_private_key(
            key_pem.encode(),
            password=None,
            backend=default_backend()
        )
        print("RSA key initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RSA key: {e}")
        priv_key = None
else:
    print("RSA key not found in environment.")
    priv_key = None

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SESSION = requests.Session()
KALSHI_DATA_OUT = "Data/kalshi_markets.csv"
# Settings for stability
MAX_RETRIES = 5
MAX_WORKERS = 10  # Enrichment can be parallel as it hits different endpoints
SAFE_DELAY = 1.0  # 1 request per second for the sensitive bulk search

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    message = text.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def get_auth_headers(method, path):
    if not priv_key or not KALSHI_ACCESS_KEY:
        return {}
    
    # Path should not include query params for signing
    path_no_query = path.split('?')[0]
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    msg = timestamp + method + path_no_query
    signature = sign_pss_text(priv_key, msg)
    
    return {
        "KALSHI-ACCESS-KEY": KALSHI_ACCESS_KEY,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature
    }

import threading

# Global rate limiter state
last_request_time = 0
rate_limit_lock = threading.Lock()
# Use a more conservative delay to avoid 429s on the election API
# SAFE_DELAY is now defined globally and used differently.

def safe_get(url, params=None):
    global last_request_time
    method = "GET"
    # Kalshi path for signing MUST start with /trade-api/v2
    # Extract from URL or hardcode if BASE is fixed
    path = "/trade-api/v2" + url.replace(BASE, "")
    
    for attempt in range(MAX_RETRIES):
        # Precise rate limiting: ensure at least SAFE_DELAY between starts
        with rate_limit_lock:
            now = time.time()
            elapsed = now - last_request_time
            if elapsed < SAFE_DELAY:
                time.sleep(SAFE_DELAY - elapsed)
            last_request_time = time.time()

        headers = get_auth_headers(method, path)
        print(f"  Fetching: {method} {path} (Try {attempt+1})")
        r = SESSION.get(url, params=params, headers=headers, timeout=30)
        
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"  429 hit. Sleeping {wait}s...")
            time.sleep(wait)
            continue
        
        r.raise_for_status()
        return r.json()
    return None

# Settings for stability
MAX_RETRIES = 5
MAX_WORKERS = 10 
SAFE_DELAY = 0.1 # 10 req/s safe for specific calls

def get_politics_series_detail():
    """Fetches all series in the Politics category with their tickers and titles."""
    all_series = []
    cursor = None
    url = f"{BASE}/series"
    print("  Fetching all Politics series...")
    while True:
        p = {"category": "Politics", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        
        data = safe_get(url, params=p)
        if not data:
            break
        
        series = data.get("series", [])
        all_series.extend(series)
        print(f"  Retrieved {len(series)} series. Total: {len(all_series)}")
        
        cursor = data.get("cursor")
        if not cursor or len(series) < 1000:
            break
            
    return all_series

def get_open_markets_for_series(series_ticker):
    """Fetches all open markets for a specific series."""
    url = f"{BASE}/markets"
    data = safe_get(url, params={"series_ticker": series_ticker, "status": "open", "limit": 100})
    return data.get("markets", []) if data else []

def get_market_detail(market_ticker):
    return safe_get(f"{BASE}/markets/{market_ticker}")

def main():
    start_time = time.time()
    
    # 1. Fetch all Politics series
    p_series_list = get_politics_series_detail()
    if not p_series_list:
        print("No Politics series found.")
        return

    # 2. For each series, get open markets in parallel
    print(f"Checking {len(p_series_list)} series for open markets...")
    active_markets_with_series = [] # List of (m, s)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_s = {executor.submit(get_open_markets_for_series, s["ticker"]): s for s in p_series_list}
        num_s = len(p_series_list)
        for i, future in enumerate(as_completed(future_to_s), start=1):
            s = future_to_s[future]
            try:
                markets = future.result()
                if markets:
                    print(f"  Found {len(markets)} markets in {s['ticker']}")
                    for m in markets:
                        active_markets_with_series.append((m, s))
                if i % 100 == 0 or i == num_s:
                    print(f"  [{i}/{num_s}] series checked")
            except Exception as exc:
                print(f"  Error checking series {s['ticker']}: {exc}")

    if not active_markets_with_series:
        print("No active markets found in any Politics series.")
        return

    print(f"Total Politics markets to enrich: {len(active_markets_with_series)}")

    # 3. Enrich found markets in parallel
    print("Enriching market details and rules...")
    rows = []
    
    def safe_enrich(m, s_detail):
        ticker = m["ticker"]
        # Market detail for rules
        m_detail = get_market_detail(ticker) or {}
        m_full = m_detail.get("market", {}) if m_detail else {}
        
        return {
            "platform": "kalshi",
            "series_ticker": s_detail.get("ticker"),
            "series_title": s_detail.get("title"),
            "market_ticker": ticker,
            "market_title": m.get("title"),
            "status": m.get("status"),
            "close_time": m.get("close_time"),
            "rules_text": (m_full.get("rules_primary") or "") + "\n" + (m_full.get("rules_secondary") or ""),
            "series_metadata": str(s_detail),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_m = {executor.submit(safe_enrich, m, s): (m, s) for m, s in active_markets_with_series}
        num_m = len(active_markets_with_series)
        for i, future in enumerate(as_completed(future_to_m), start=1):
            try:
                rows.append(future.result())
                if i % 50 == 0 or i == num_m:
                    print(f"  [{i}/{num_m}] enriched")
            except Exception as exc:
                print(f"  Error enriching market: {exc}")

    if rows:
        df = pd.DataFrame(rows)
        print(df.info())
        df.to_csv(KALSHI_DATA_OUT, index=False)
        print(f"Wrote {len(df)} rows to {KALSHI_DATA_OUT}")
    else:
        print("No data collected.")
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()