# RazorRecover AI — V2 Fixed

## What V2 does

V1 predicted whether a payment would fail.

V2 starts after a payment has failed and asks:

**Which recovery action is most likely to recover the money?**

It compares:
- retry
- payment_link
- reminder

For each action it predicts recovery probability and selects the best action.

## Project structure

RazorRecover_AI_V2_Fixed/
- app.py — Streamlit dashboard
- requirements.txt — Python dependencies
- README.md — instructions
- data/recovery_transactions.csv — 15,000 synthetic recovery transactions
- src/train_recovery.py — trains the Random Forest recovery model
- src/recovery_engine.py — scores the three recovery actions
- models/ — trained model is created here after training

## Run

Open the terminal in this exact project folder.

```bash
pip install -r requirements.txt
python src/train_recovery.py
python -m streamlit run app.py
```

## Important

The dataset and recovery labels are synthetic. The metrics do NOT represent Razorpay's real performance.

V3 can add an LLM agent, Razorpay Test Mode APIs, webhooks, a database, and controlled recovery experiments.
