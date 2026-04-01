# ================================================================
# FILE: feature_engineering.py
# Converts raw trade data into TCN-ready tensors
# ================================================================

import pickle
import json
import numpy as np
from config import EXPORT_DIR, WINDOW_SIZE
 
# Load artifacts once at module level
with open(EXPORT_DIR + "scaler.pkl",       "rb") as f: scaler       = pickle.load(f)
with open(EXPORT_DIR + "cat_encoders.pkl", "rb") as f: cat_encoders = pickle.load(f)
with open(EXPORT_DIR + "feature_config.json")     as f: feat_cfg    = json.load(f)
 
numerical_features   = feat_cfg["numerical_features"]
categorical_features = feat_cfg["categorical_features"]
 
PLAN_SCORE_MAP = {
    "Followed": 3, "Partial": 2, "Deviated": 1, "No Plan": 0
}
 
 
def rr_ratio(s: str) -> float:
    try:
        a, b = s.split(":")
        return float(b) / float(a)
    except Exception:
        return 2.0
 
 
def trade_dict_to_vectors(trade: dict):
    """
    Convert a trade dictionary to scaled numerical vector
    and encoded categorical vector.
    trade keys must match TradeData field names.
    """
    num_raw = {
        "hour":            int(trade.get("hour", 9)),
        "day_of_week":     int(trade.get("day_of_week", 0)),
        "is_night":        int(trade.get("is_night", 0)),
        "RR_ratio":        rr_ratio(trade.get("risk_to_reward", "1:2")),
        "Risk Percentage": float(trade.get("risk_percentage", 1.0)),
        "Lot Size":        float(trade.get("lot_size", 0.1)),
        "Entry Price":     float(trade.get("entry_price", 1.0)),
        "used_stop_loss":  int(bool(trade.get("stop_loss_used", True))),
        "plan_score":      PLAN_SCORE_MAP.get(
                               trade.get("pre_trade_plan", "No Plan"), 0),
        "had_plan":        int(trade.get("pre_trade_plan", "No Plan") != "No Plan"),
        "followed_plan":   int(trade.get("pre_trade_plan", "") == "Followed"),
        "deviated":        int(trade.get("pre_trade_plan", "") == "Deviated"),
        "partial_follow":  int(trade.get("pre_trade_plan", "") == "Partial"),
    }
 
    num_arr    = np.array(
        [[num_raw[f] for f in numerical_features]], dtype=np.float32)
    num_scaled = scaler.transform(num_arr)[0]
 
    cat_raw = {
        "Session":          str(trade.get("session", "London Open")),
        "Pair":             str(trade.get("pair", "EURUSD")),
        "Direction":        str(trade.get("direction", "BUY")),
        "Market Condition": str(trade.get("market_condition", "Choppy")),
        "emotion_before":   str(trade.get("emotion_before", "Focused")),
    }
 
    cat_arr = []
    for f in categorical_features:
        le = cat_encoders[f]
        val = cat_raw[f]
        # Handle unseen categories gracefully
        if val in le.classes_:
            cat_arr.append(le.transform([val])[0])
        else:
            cat_arr.append(0)
    cat_arr = np.array(cat_arr, dtype=np.int64)
 
    return num_scaled, cat_arr
 
 
def build_window(current_trade: dict, history: list = None):
    """
    Build a 50-trade sequence window.
    current_trade goes in position [-1] (most recent).
    history fills positions [0..48] (oldest to newest).
    """
    n_num = len(numerical_features)
    n_cat = len(categorical_features)
 
    num_w = np.zeros((WINDOW_SIZE, n_num), dtype=np.float32)
    cat_w = np.zeros((WINDOW_SIZE, n_cat), dtype=np.int64)
 
    if history:
        recent = history[-(WINDOW_SIZE - 1):]
        for i, h in enumerate(recent):
            hn, hc = trade_dict_to_vectors(h)
            num_w[i] = hn
            cat_w[i] = hc
 
    cn, cc = trade_dict_to_vectors(current_trade)
    num_w[-1] = cn
    cat_w[-1] = cc
 
    return num_w, cat_w, cn  # cn = current numerical vector (unscaled path)
 
 
def get_feature_signals(num_scaled_vec: np.ndarray) -> list:
    """
    Generate plain-English explanations of which features
    triggered the TCN prediction. Used for explainability.
    """
    vals = dict(zip(numerical_features, num_scaled_vec))
    signals = []
 
    # These thresholds are in SCALED space (z-scores)
    # negative z-score for used_stop_loss means stop loss NOT used
    if vals.get("used_stop_loss", 0) < -0.5:
        signals.append("No stop loss was set on this trade")
    if vals.get("plan_score", 0) < -1.0:
        signals.append("No pre-trade plan was followed")
    if vals.get("deviated", 0) > 0.5:
        signals.append("You deviated significantly from your plan")
    if vals.get("Lot Size", 0) > 1.0:
        signals.append("Position size was above your typical range")
    if vals.get("Risk Percentage", 0) > 1.0:
        signals.append("Risk percentage was elevated above normal")
    if vals.get("RR_ratio", 0) < -0.5:
        signals.append("Risk-to-reward ratio was unfavorable")
    if vals.get("is_night", 0) > 0.5:
        signals.append("Trade taken during night session (higher emotion risk)")
    if vals.get("followed_plan", 0) < -0.5:
        signals.append("Pre-trade plan was not followed on this trade")
 
    if not signals:
        signals.append("Multiple behavioral indicators combined to flag this pattern")
 
    return signals[:4]
