# PsychoAI Backend

PsychoAI is an AI-powered trading psychology coaching system that analyzes 
a trader's journal to detect behavioral patterns and deliver personalized 
coaching through a conversational AI named Plutus.

## What It Does

Traders upload a CSV or Excel trade journal. The backend runs the journal 
through a causal Temporal Convolutional Network (TCN) trained to classify 
8 common psychological trading mistakes — such as Revenge Trading, 
Overtrading, and Holding Losers Too Long. After classification, a RAG-based 
coaching layer powered by Groq and Qdrant generates contextual coaching 
messages tailored to the detected patterns.

Cross-session memory is stored in Firestore and injected into each Groq 
prompt so that Plutus remembers a trader's history across sessions. Daily 
push notifications are sent at 08:00 UTC via Firebase Cloud Messaging (FCM).

## Tech Stack

- FastAPI — REST API framework
- PyTorch — TCN inference
- Groq (llama-3.1-8b-instant) — LLM coaching responses
- Qdrant — vector store for RAG
- Firebase Admin SDK — Firestore storage and FCM notifications
- Railway — cloud deployment
- Python 3.11.9 / scikit-learn 1.6.1

## Key Endpoints

- `POST /analyze` — accepts a trade journal file, returns detected patterns 
  and coaching snippets
- `GET /trader/{id}/profile` — returns a trader's session history and 
  behavioral pattern log
- `POST /notify` — triggers FCM push notification for a trader

## ML Model

The TCN model has 927K parameters and was trained on a synthetic 100K-row 
dataset. It uses last-timestep feature extraction, MixUp regularization, 
and per-class pseudo-label thresholds via semi-supervised learning with 
non-overlapping splits.

## Deployment

Live at: https://psychoai-backend-production.up.railway.app

The project uses a railway.json forcing the Nixpacks builder. Python is 
pinned to 3.11.9 and Firebase credentials are passed via the 
GOOGLE_CREDENTIALS environment variable.

## Setup

1. Clone the repo
2. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
3. Add your environment variables:
   - GOOGLE_CREDENTIALS
   - GROQ_API_KEY
   - QDRANT_URL and QDRANT_API_KEY
4. Run locally:
   uvicorn main:app --reload
