import urllib.request
import json

BASE_URL = "https://psychoai-backend-production.up.railway.app"

# ── Test 1: Health ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 — Health Check")
print("=" * 50)
result = urllib.request.urlopen(f"{BASE_URL}/health").read().decode()
print(result)

# ── Test 2: Analyze Trade ──────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2 — Analyze Trade (high confidence)")
print("=" * 50)
bad_trade = {
    "session": "London Open",
    "pair": "EURUSD",
    "direction": "BUY",
    "lot_size": 5.0,
    "entry_price": 1.0850,
    "risk_percentage": 5.0,
    "risk_to_reward": "1:0.5",
    "market_condition": "Choppy",
    "emotion_before": "Frustrated",
    "stop_loss_used": False,
    "pre_trade_plan": "Deviated",
    "hour": 9,
    "day_of_week": 1,
    "is_night": 0
}
data = json.dumps({
    "trader_id": "T001_test",
    "fcm_token": None,
    "trade": bad_trade,
    "history": [bad_trade] * 49
}).encode()
req = urllib.request.Request(
    f"{BASE_URL}/analyze_trade",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
result = json.loads(urllib.request.urlopen(req).read().decode())
print(json.dumps(result, indent=2))

# ── Test 3: Trader Profile ─────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 3 — Trader Profile")
print("=" * 50)
result = urllib.request.urlopen(
    f"{BASE_URL}/trader/T001_test/profile"
).read().decode()
print(json.dumps(json.loads(result), indent=2))

# ── Test 4: Latest Coaching ────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 4 — Latest Coaching")
print("=" * 50)
result = urllib.request.urlopen(
    f"{BASE_URL}/coaching/T001_test/latest"
).read().decode()
print(json.dumps(json.loads(result), indent=2))

# ── Test 5: Chat With Plutus ───────────────────────────────────
print("\n" + "=" * 50)
print("TEST 5 — Chat With Plutus")
print("=" * 50)
data = json.dumps({
    "trader_id": "T001_test",
    "message": "Why do I keep making impulsive trades?"
}).encode()
req = urllib.request.Request(
    f"{BASE_URL}/chat",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
result = json.loads(urllib.request.urlopen(req).read().decode())
print(json.dumps(result, indent=2))

print("\n" + "=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)