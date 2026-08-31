import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

DATA="data/recovery_transactions.csv"
MODEL="models/recovery_model.joblib"

NUM=["amount_inr","previous_success_rate","previous_failures","days_since_last_payment",
     "customer_tenure_days","is_returning_customer","checkout_time_sec","retry_count"]
CAT=["payment_method","failure_reason","action"]

df=pd.read_csv(DATA)
X=df[NUM+CAT]
y=df["recovered"]

pre=ColumnTransformer([
    ("num","passthrough",NUM),
    ("cat",OneHotEncoder(handle_unknown="ignore"),CAT)
])

model=Pipeline([
    ("preprocess",pre),
    ("classifier",RandomForestClassifier(
        n_estimators=250,max_depth=10,min_samples_leaf=3,
        class_weight="balanced",random_state=42,n_jobs=-1
    ))
])

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1]

print("Recovery model ROC-AUC:",round(roc_auc_score(yte,p),4))
print(classification_report(yte,(p>=.5).astype(int)))

os.makedirs("models",exist_ok=True)
joblib.dump(model,MODEL)
print("Saved:",MODEL)
