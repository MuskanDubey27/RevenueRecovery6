import streamlit as st
import pandas as pd
import joblib
import sys
import os
import traceback

st.set_page_config(
    page_title="RazorRecover AI V2",
    page_icon="💳",
    layout="wide"
)

st.title("💳 RazorRecover AI — V2")
st.write(
    "AI-powered revenue recovery system that predicts "
    "which recovery action is most likely to recover a failed payment."
)

DATA_PATH="data/recovery_transactions.csv"
MODEL_PATH="models/recovery_model.joblib"

with st.sidebar:
    st.header("System Status")
    st.write("Project folder:")
    st.code(os.getcwd())
    st.write("Dataset:", "✅ Found" if os.path.exists(DATA_PATH) else "❌ Missing")
    st.write("Model:", "✅ Found" if os.path.exists(MODEL_PATH) else "❌ Missing")

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run Streamlit from the RazorRecover_AI_V2_Fixed folder.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.warning("ML model has not been trained yet.")
    st.info("Run: python src/train_recovery.py")
    st.stop()

try:
    sys.path.insert(0, os.path.abspath("src"))
    from recovery_engine import choose_best_action

    df=pd.read_csv(DATA_PATH)
    model=joblib.load(MODEL_PATH)

    st.success(f"✅ System loaded successfully — {len(df):,} transactions available.")

    # Evaluate 1,000 transactions for a fast first dashboard.
    sample=df.head(1000).copy()
    results=[]
    progress=st.progress(0)

    for i,(_,row) in enumerate(sample.iterrows()):
        result=choose_best_action(model,row.to_dict())

        if result["status"]=="auto_recover":
            best=result["best_action"]
            results.append({
                "transaction_id":row["transaction_id"],
                "amount_inr":float(row["amount_inr"]),
                "failure_reason":row["failure_reason"],
                "retry_count":int(row["retry_count"]),
                "best_action":best["action"],
                "recovery_probability":float(best["recovery_probability"]),
                "expected_recovered_inr":
                    float(row["amount_inr"])*float(best["recovery_probability"])
            })

        if i%25==0:
            progress.progress((i+1)/len(sample))

    progress.empty()
    results_df=pd.DataFrame(results)

    if results_df.empty:
        st.error("No predictions were generated.")
        st.stop()

    st.divider()
    st.subheader("📈 Revenue Recovery Overview")

    total=len(results_df)
    expected_revenue=results_df["expected_recovered_inr"].sum()
    avg_probability=results_df["recovery_probability"].mean()

    c1,c2,c3=st.columns(3)
    c1.metric("Transactions Evaluated",f"{total:,}")
    c2.metric("Expected Recoverable Revenue",f"₹{expected_revenue:,.0f}")
    c3.metric("Average Recovery Probability",f"{avg_probability*100:.1f}%")

    st.divider()
    st.subheader("📊 Recommended Recovery Actions")
    st.bar_chart(results_df["best_action"].value_counts())

    st.subheader("🔎 Top Revenue Recovery Opportunities")
    st.dataframe(
        results_df.sort_values(
            "expected_recovered_inr",ascending=False
        ).head(50),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("🤖 Analyze One Failed Transaction")

    transaction_id=st.selectbox(
        "Choose a transaction",
        sample["transaction_id"].tolist()
    )
    selected=sample[
        sample["transaction_id"]==transaction_id
    ].iloc[0]

    result=choose_best_action(model,selected.to_dict())

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Amount",f"₹{selected['amount_inr']:,.0f}")
    c2.metric("Failure",str(selected["failure_reason"]))
    c3.metric("Previous Failures",str(selected["previous_failures"]))
    c4.metric("Retry Count",str(selected["retry_count"]))

    if result["status"]=="auto_recover":
        best=result["best_action"]
        st.success(f"🎯 Recommended Action: {best['action'].upper()}")
        st.metric(
            "Best Recovery Probability",
            f"{best['recovery_probability']*100:.2f}%"
        )
        st.metric(
            "Expected Recovery",
            f"₹{selected['amount_inr']*best['recovery_probability']:,.0f}"
        )

        alternatives=pd.DataFrame(result["ranked_actions"])
        alternatives["recovery_probability"]*=100
        alternatives=alternatives.rename(columns={
            "action":"Action",
            "recovery_probability":"Recovery Probability (%)"
        })

        st.write("### Compare All Recovery Actions")
        st.dataframe(
            alternatives,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Human approval required for this transaction.")

except Exception:
    st.error("❌ The dashboard encountered an error.")
    st.code(traceback.format_exc())
