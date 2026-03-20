import re
import pandas as pd


# =========================
# Config
# =========================
KALSHI_CSV = "Data/kalshi_markets.csv"
POLYMARKET_CSV = "Data/polymarket_markets.csv"
OUTPUT_CSV = "Data/candidate_series_matches.csv"

MAX_DATE_DIFF_DAYS = 45
MIN_JACCARD = 0.25

STOPWORDS = {
    "will", "would", "could", "should",
    "any", "anyone", "the", "a", "an",
    "in", "of", "to", "for", "from", "on", "at", "by", "with",
    "be", "is", "are", "was", "were", "been", "being",
    "who", "what", "when", "where", "why", "how",
    "win", "wins", "winner", "winners", "winning",
    "vote", "votes", "voting",
    "candidate", "candidates",
    "election", "elections",
    "presidential", "president",
    "primary", "general",
    "round", "party",
    "control", "lose", "gain",
    "next", "this", "that", "before", "after",
    "first", "second", "third",
    "total", "between", "less", "more", "than",
    "yes", "no"
}


# =========================
# Helpers
# =========================
def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = normalize_districts(value)
    value = re.sub(r"\s+", " ", value)
    return value


def tokenize(text):
    text = text.lower()

    text = re.sub(r"([a-z]{2})-(\d+)", r"\1\2", text)

    words = re.findall(r"[a-z0-9]+", text)

    return {
        w for w in words
        if len(w) > 2 and w not in STOPWORDS
    }


def parse_status(series):
    """Normalize status text."""
    return series.astype(str).str.lower().str.strip()

def normalize_districts(text):
    text = text.lower()

    # House CA 9 → ca-9
    text = re.sub(r"house\s+([a-z]{2})\s*(\d{1,2})", r"\1-\2", text)

    # CA 9 → ca-9
    text = re.sub(r"\b([a-z]{2})\s+(\d{1,2})\b", r"\1-\2", text)

    # Normalize at-large districts (AK-AL stays ak-al)
    text = re.sub(r"([a-z]{2})-(\d+)", r"\1\2", text)

    return text

def extract_district(text):
    text = text.lower()

    m = re.search(r"\b([a-z]{2})-(\d{1,2}|al)\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    return None


def district_conflict(a, b):
    d1 = extract_district(a)
    d2 = extract_district(b)

    if d1 and d2 and d1 != d2:
        return True

    return False

def weighted_jaccard(text1: str, text2: str) -> float:
    """
    Jaccard similarity with a small confidence boost for stronger evidence.

    Boosts:
    - more shared tokens
    - slightly longer phrases

    The boost is capped so long strings do not get over-rewarded.
    """
    set1 = tokenize(text1)
    set2 = tokenize(text2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    jaccard = intersection / union

    avg_len = (len(set1) + len(set2)) / 2

    # Mild boost for longer phrases, capped
    length_factor = min(1 + 0.05 * max(avg_len - 4, 0), 1.3)

    # Stronger evidence when more actual tokens overlap, capped
    overlap_factor = min(1 + 0.1 * intersection, 1.5)

    score = jaccard * length_factor * overlap_factor

    return min(score, 1.0)


def load_and_clean_data():
    print("\n=== LOADING CSV FILES ===")
    kalshi_df = pd.read_csv(KALSHI_CSV)
    polymarket_df = pd.read_csv(POLYMARKET_CSV)

    print(f"Loaded Kalshi rows: {len(kalshi_df)}")
    print(f"Loaded Polymarket rows: {len(polymarket_df)}")

    print("\n=== CLEANING DATAFRAMES ===")

    # Normalize close_time
    kalshi_df["close_time"] = pd.to_datetime(kalshi_df["close_time"], errors="coerce", utc=True)
    polymarket_df["close_time"] = pd.to_datetime(polymarket_df["close_time"], errors="coerce", utc=True)

    # Normalize status
    kalshi_df["status"] = parse_status(kalshi_df["status"])
    polymarket_df["status"] = parse_status(polymarket_df["status"])

    # Filter out non-tradable markets
    kalshi_before = len(kalshi_df)
    polymarket_before = len(polymarket_df)

    kalshi_df = kalshi_df[kalshi_df["status"] != "finalized"].copy()
    polymarket_df = polymarket_df[polymarket_df["status"] == "active"].copy()

    print(f"Kalshi rows after status filter: {len(kalshi_df)} (removed {kalshi_before - len(kalshi_df)})")
    print(f"Polymarket rows after status filter: {len(polymarket_df)} (removed {polymarket_before - len(polymarket_df)})")

    # Build clean title columns if not already present

    if "series_title_clean" not in kalshi_df.columns:
        kalshi_df["series_title_clean"] = kalshi_df["series_title"].apply(clean_text)
    else:
        kalshi_df["series_title_clean"] = kalshi_df["series_title_clean"].apply(clean_text)

    if "market_title_clean" not in kalshi_df.columns:
        kalshi_df["market_title_clean"] = kalshi_df["market_title"].apply(clean_text)
    else:
        kalshi_df["market_title_clean"] = kalshi_df["market_title_clean"].apply(clean_text)

    if "series_title_clean" not in polymarket_df.columns:
        polymarket_df["series_title_clean"] = polymarket_df["series_title"].apply(clean_text)
    else:
        polymarket_df["series_title_clean"] = polymarket_df["series_title_clean"].apply(clean_text)

    if "market_title_clean" not in polymarket_df.columns:
        polymarket_df["market_title_clean"] = polymarket_df["market_title"].apply(clean_text)
    else:
        polymarket_df["market_title_clean"] = polymarket_df["market_title_clean"].apply(clean_text)

    return kalshi_df, polymarket_df


def build_series_tables(kalshi_df, polymarket_df):
    print("\n=== BUILDING UNIQUE SERIES TABLES ===")

    kalshi_series = (
        kalshi_df[
            ["series_ticker", "series_title", "series_title_clean", "close_time", "status", "rules_text"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    polymarket_series = (
        polymarket_df[
            ["series_ticker", "series_title", "series_title_clean", "close_time", "status", "rules_text"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Unique Kalshi series: {len(kalshi_series)}")
    print(f"Unique Polymarket series: {len(polymarket_series)}")

    return kalshi_series, polymarket_series

def generate_candidate_matches(kalshi_series, polymarket_series):
    print("\n=== GENERATING CANDIDATE MATCHES ===")

    matches = []
    pairs_checked = 0
    pairs_surviving_date = 0
    pairs_surviving_jaccard = 0

    for _, krow in kalshi_series.iterrows():
        for _, prow in polymarket_series.iterrows():
            pairs_checked += 1
            if district_conflict(krow["series_title_clean"], prow["series_title_clean"]):
                continue
            # Date pre-filter
            if pd.notna(krow["close_time"]) and pd.notna(prow["close_time"]):
                date_diff_days = abs((krow["close_time"] - prow["close_time"]).days)
                if date_diff_days > MAX_DATE_DIFF_DAYS:
                    continue
            else:
                date_diff_days = None

            pairs_surviving_date += 1

            jaccard = weighted_jaccard(
                krow["series_title_clean"],
                prow["series_title_clean"]
            )

            if jaccard < MIN_JACCARD:
                continue

            pairs_surviving_jaccard += 1

            shared_words = sorted(
                tokenize(krow["series_title_clean"]) & tokenize(prow["series_title_clean"])
            )

            matches.append({
                "kalshi_series_ticker": krow["series_ticker"],
                "kalshi_series": krow["series_title"],
                "kalshi_rules_text": krow.get("rules_text"),
                "kalshi_status": krow["status"],
                "kalshi_close": krow["close_time"],

                "polymarket_series_ticker": prow["series_ticker"],
                "polymarket_series": prow["series_title"],
                "polymarket_rules_text": prow.get("rules_text"),
                "polymarket_status": prow["status"],
                "polymarket_close": prow["close_time"],

                "jaccard_score": round(jaccard, 4),
                "date_diff_days": date_diff_days,
                "shared_words": ", ".join(shared_words),
            })      

    matches_df = pd.DataFrame(matches)

    print(f"Total pairs checked: {pairs_checked}")
    print(f"Pairs surviving date filter: {pairs_surviving_date}")
    print(f"Pairs surviving Jaccard filter: {pairs_surviving_jaccard}")

    if matches_df.empty:
        print("\nNo candidate matches found.")
        return matches_df

    matches_df = matches_df.sort_values(
        by=["jaccard_score", "date_diff_days"],
        ascending=[False, True]
    ).reset_index(drop=True)

    print(f"\nCandidate matches found: {len(matches_df)}")
    print("\nTop 20 candidate matches:")
    print(matches_df.head(20).to_string())

    return matches_df


def main():
    print("=== STARTING JACCARD MATCHING SCRIPT ===")

    kalshi_df, polymarket_df = load_and_clean_data()
    kalshi_series, polymarket_series = build_series_tables(kalshi_df, polymarket_df)
    matches_df = generate_candidate_matches(kalshi_series, polymarket_series)

    print("\n=== WRITING OUTPUT ===")
    matches_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(matches_df)} rows to {OUTPUT_CSV}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()