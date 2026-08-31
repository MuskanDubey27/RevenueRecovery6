import joblib
import pandas as pd

ACTIONS=["retry","payment_link","reminder"]

def load_model():
    return joblib.load("models/recovery_model.joblib")

def score_actions(model,row):
    ranked=[]
    for action in ACTIONS:
        x=pd.DataFrame([row.copy()])
        x["action"]=action
        probability=float(model.predict_proba(x)[:,1][0])
        ranked.append({
            "action":action,
            "recovery_probability":probability
        })
    return sorted(
        ranked,
        key=lambda item:item["recovery_probability"],
        reverse=True
    )

def choose_best_action(model,row):
    ranked=score_actions(model,row)
    best=ranked[0]

    # Don't repeatedly retry a payment.
    if row.get("retry_count",0)>=2 and best["action"]=="retry":
        best=next(x for x in ranked if x["action"]!="retry")

    # High-value payments require human approval.
    if row.get("amount_inr",0)>50000:
        return {
            "status":"human_approval_required",
            "ranked_actions":ranked
        }

    return {
        "status":"auto_recover",
        "best_action":best,
        "ranked_actions":ranked
    }
