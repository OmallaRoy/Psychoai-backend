# ================================================================
# FILE: notifications.py
# Firebase Cloud Messaging + daily scheduler
# ================================================================

import logging
import schedule
import time
from threading import Thread
from datetime import datetime
from firebase_admin import messaging, firestore
from config import NOTIFICATION_HOUR
 
log = logging.getLogger("notifications")
db  = firestore.client()
 
 
def send_coaching_notification(
    fcm_token: str,
    pattern: str,
    coaching: str,
    confidence: float,
):
    """
    Send push notification to trader when coaching is ready.
    Called from background task after Groq generates coaching.
    """
    if not fcm_token:
        return
 
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title ="Plutus: {} detected".format(pattern),
                body  =coaching[:120] + "...",
            ),
            data={
                "type":           "coaching",
                "pattern":        pattern,
                "confidence":     str(round(confidence, 2)),
                "full_coaching":  coaching,
                "timestamp":      datetime.utcnow().isoformat(),
            },
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    icon ="ic_plutus",
                    color="#6C63FF",
                )
            ),
            token=fcm_token,
        )
        messaging.send(message)
        log.info("Coaching notification sent for pattern: %s", pattern)
 
    except Exception as e:
        log.error("send_coaching_notification error: %s", e)
 
 
def send_daily_notification(
    fcm_token: str,
    trader_id: str,
    insight: str,
    pattern: str,
):
    """Send daily morning behavioral insight notification."""
    if not fcm_token:
        return
 
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title="Good morning from Plutus",
                body =insight[:120],
            ),
            data={
                "type":         "daily",
                "pattern":      pattern,
                "full_message": insight,
                "trader_id":    trader_id,
            },
            android=messaging.AndroidConfig(
                priority="normal",
                notification=messaging.AndroidNotification(
                    icon ="ic_plutus",
                    color="#6C63FF",
                )
            ),
            token=fcm_token,
        )
        messaging.send(message)
        log.info("Daily notification sent to trader: %s", trader_id)
 
    except Exception as e:
        log.error("send_daily_notification error for %s: %s", trader_id, e)
 
 
def run_daily_job():
    """
    Fetch all active traders, generate daily insight, send notifications.
    Runs every day at NOTIFICATION_HOUR (default 08:00 UTC).
    """
    from memory import get_weekly_summary, get_fcm_token
    from coaching import generate_daily_insight
 
    log.info("Running daily notifications at %s UTC",
             datetime.utcnow().isoformat()[:16])
 
    try:
        traders = db.collection("trader_profiles").stream()
 
        for trader_doc in traders:
            trader_id = trader_doc.id
            summary   = get_weekly_summary(trader_id)
 
            if not summary:
                continue
 
            fcm_token = get_fcm_token(trader_id)
            if not fcm_token:
                continue
 
            insight = generate_daily_insight(
                trader_id,
                summary["patterns"],
                summary["most_common"],
                summary["count"],
                summary["total"],
            )
 
            send_daily_notification(
                fcm_token, trader_id,
                insight, summary["most_common"]
            )
 
    except Exception as e:
        log.error("run_daily_job error: %s", e)
 
 
def start_scheduler():
    """Start background scheduler thread for daily notifications."""
    schedule.every().day.at(NOTIFICATION_HOUR).do(run_daily_job)
    log.info("Daily notification scheduler started — runs at %s UTC",
             NOTIFICATION_HOUR)
 
    def loop():
        while True:
            schedule.run_pending()
            time.sleep(30)
 
    Thread(target=loop, daemon=True).start()
