import streamlit as st
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📞",
    layout="centered"
)

# ── Load pipeline ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('src/churn_pipeline.pkl')

pipeline = load_model()

# ── Header ────────────────────────────────────────────────
st.title("📞 Telecom Customer Churn Predictor")
st.markdown("Enter customer details below to predict churn risk.")
st.divider()

# ── Input Section ─────────────────────────────────────────
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    AccountWeeks    = st.number_input("Account Weeks",          min_value=0,   max_value=300,  value=100)
    DataUsage       = st.number_input("Data Usage (GB)",        min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    DayMins         = st.number_input("Day Minutes",            min_value=0.0, max_value=400.0,value=180.0,step=1.0)
    MonthlyCharge   = st.number_input("Monthly Charge ($)",     min_value=0.0, max_value=150.0,value=65.0, step=0.5)
    OverageFee      = st.number_input("Overage Fee ($)",        min_value=0.0, max_value=50.0, value=10.0, step=0.5)

with col2:
    ContractRenewal = st.selectbox("Contract Renewal",          [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    DataPlan        = st.selectbox("Data Plan",                 [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    CustServCalls   = st.number_input("Customer Service Calls", min_value=0,   max_value=15,   value=1)
    DayCalls        = st.number_input("Day Calls",              min_value=0,   max_value=200,  value=100)
    RoamMins        = st.number_input("Roaming Minutes",        min_value=0.0, max_value=50.0, value=10.0, step=0.5)

st.divider()

# ── Prediction ────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True):

    input_data = np.array([[
        AccountWeeks, ContractRenewal, DataPlan,
        DataUsage, CustServCalls, DayMins,
        DayCalls, MonthlyCharge, OverageFee, RoamMins
    ]])

    # Pipeline handles scaling internally
    prob = pipeline.predict_proba(input_data)[0][1]

    # Business threshold = 0.4 (recall-optimized)
    threshold = 0.4
    prediction = prob >= threshold

    st.subheader("Prediction Result")

    # ── Risk Display ──────────────────────────────────────
    if prob >= 0.7:
        st.error(f"🔴 HIGH RISK — This customer is very likely to churn")
    elif prob >= 0.4:
        st.warning(f"🟡 MEDIUM RISK — This customer shows churn signals")
    else:
        st.success(f"🟢 LOW RISK — This customer is likely to stay")

    # ── Probability Bar ───────────────────────────────────
    st.metric(label="Churn Probability", value=f"{prob:.1%}")
    st.progress(float(prob))

    # ── Key Risk Factors ──────────────────────────────────
    st.subheader("Risk Factor Summary")

    risk_flags = []
    if ContractRenewal == 0:
        risk_flags.append("⚠️ No contract renewal — strongest churn signal")
    if CustServCalls >= 3:
        risk_flags.append("⚠️ High service calls — indicates unresolved issues")
    if MonthlyCharge > 65:
        risk_flags.append("⚠️ Above-average monthly charge")
    if DataPlan == 0:
        risk_flags.append("⚠️ No data plan — lower product engagement")

    if risk_flags:
        for flag in risk_flags:
            st.markdown(flag)
    else:
        st.markdown("✅ No major risk factors detected")

    # ── Retention Suggestion ──────────────────────────────
    if prediction:
        st.divider()
        st.subheader("💡 Recommended Retention Action")
        if ContractRenewal == 0 and CustServCalls >= 3:
            st.info("Priority: Assign a retention agent. Offer contract incentive + resolve open service complaints immediately.")
        elif ContractRenewal == 0:
            st.info("Offer a contract lock-in discount (5-10% charge reduction) before next billing cycle.")
        elif CustServCalls >= 3:
            st.info("Escalate service complaint to senior support. Follow up within 48 hours.")