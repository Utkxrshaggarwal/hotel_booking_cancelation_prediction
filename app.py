"""
Hotel Booking Cancellation Predictor - Streamlit app
====================================================
1. Run  python train_model.py  once (creates models/hotel_cancellation_model.joblib)
2. Start the app:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hotel_cancellation_model.joblib"

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")

# Friendly help text for each input
HINTS = {
    "lead_time": "Days between booking date and arrival date",
    "adr": "Average daily rate (price per night)",
    "arrival_date_week_number": "Week of the year (1-53)",
    "arrival_date_day_of_month": "Day of the month (1-31)",
    "arrival_date_year": "Year of arrival",
    "arrival_date_month": "Month of arrival",
    "stays_in_weekend_nights": "Number of weekend nights (Sat/Sun)",
    "stays_in_week_nights": "Number of week nights (Mon-Fri)",
    "adults": "Number of adults",
    "children": "Number of children",
    "babies": "Number of babies",
    "country": "Guest's country (3-letter code, e.g. PRT, GBR, IND)",
    "agent": "Travel agent ID (0 = no agent)",
    "company": "Company ID (0 = none)",
    "previous_cancellations": "Earlier bookings this guest canceled",
    "previous_bookings_not_canceled": "Earlier bookings this guest kept",
    "booking_changes": "Number of changes made to the booking",
    "total_of_special_requests": "Number of special requests",
    "required_car_parking_spaces": "Parking spaces requested",
    "days_in_waiting_list": "Days the booking was on a waiting list",
    "is_repeated_guest": "1 = returning guest, 0 = new guest",
    "market_segment": "How the booking was sourced",
    "distribution_channel": "Booking distribution channel",
    "deposit_type": "Type of deposit paid",
    "customer_type": "Type of customer",
    "meal": "Meal plan booked",
    "reserved_room_type": "Room type reserved",
    "hotel": "Hotel type",
}

# (min, max) for numeric inputs; anything not listed gets a generic range
RANGES = {
    "arrival_date_week_number": (1, 53),
    "arrival_date_day_of_month": (1, 31),
    "arrival_date_year": (2015, 2035),
    "is_repeated_guest": (0, 1),
    "adr": (0.0, 1000.0),
}
FLOAT_FEATURES = {"adr"}


def nice(name: str) -> str:
    return name.replace("_", " ").capitalize()


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


def render_input(feature, art, container):
    """Create the right widget for a feature and return its value."""
    label = nice(feature)
    help_text = HINTS.get(feature)
    default = art["defaults"][feature]

    if feature in art["categorical_features"]:
        classes = list(art["label_encoders"][feature].classes_)
        index = classes.index(default) if default in classes else 0
        return container.selectbox(label, classes, index=index, help=help_text, key=feature)

    if feature in FLOAT_FEATURES:
        lo, hi = RANGES.get(feature, (0.0, 100000.0))
        return container.number_input(
            label, min_value=float(lo), max_value=float(hi),
            value=float(min(max(default, lo), hi)), step=1.0, help=help_text, key=feature,
        )

    lo, hi = RANGES.get(feature, (0, 100000))
    return container.number_input(
        label, min_value=int(lo), max_value=int(hi),
        value=int(min(max(round(default), lo), hi)), step=1, help=help_text, key=feature,
    )


def main():
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.caption("Estimate the chance that a hotel booking will be canceled.")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python train_model.py` first, then restart this app.")
        st.stop()

    art = load_artifacts()
    features = art["selected_features"]
    metrics = art["test_metrics"]

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("About the model")
        st.write(f"**Algorithm:** {art['model_name']}")
        st.write("**Test performance**")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
        c2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
        c1.metric("Precision", f"{metrics['Precision']:.1%}")
        c2.metric("Recall", f"{metrics['Recall']:.1%}")
        st.divider()
        threshold = st.slider(
            "Decision threshold",
            0.10, 0.90, 0.50, 0.05,
            help="Flag a booking as 'likely canceled' when its cancel probability is at or "
                 "above this value. Lower = catch more cancellations but more false alarms.",
        )
        st.caption(
            "Pipeline: filter feature selection → label encoding → scaling → PCA → "
            "tuned classifier."
        )

    # ---------------- input form ----------------
    st.subheader("Booking details")
    with st.form("booking_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(features):
            values[feat] = render_input(feat, art, cols[i % 3])
        submitted = st.form_submit_button("Predict cancellation", type="primary", use_container_width=True)

    # ---------------- prediction ----------------
    if submitted:
        X_new = pd.DataFrame([values], columns=features)
        for col in art["categorical_features"]:
            X_new[col] = art["label_encoders"][col].transform(X_new[col].astype(str))

        X_pca = art["pca"].transform(art["scaler"].transform(X_new))
        prob = float(art["model"].predict_proba(X_pca)[0, 1])
        canceled = prob >= threshold

        st.subheader("Result")
        left, right = st.columns([1, 2])
        left.metric("Cancellation probability", f"{prob:.1%}")
        left.metric("Decision threshold", f"{threshold:.0%}")
        with right:
            if canceled:
                st.error("⚠️ **Likely to be CANCELED**")
            else:
                st.success("✅ **Likely to be KEPT**")
            st.progress(min(max(prob, 0.0), 1.0))
            if prob < 0.33:
                st.write("Risk level: **Low**")
            elif prob < 0.66:
                st.write("Risk level: **Medium**")
            else:
                st.write("Risk level: **High**")

        with st.expander("Show the values used"):
            st.dataframe(pd.DataFrame([values]).T.rename(columns={0: "Value"}).astype(str))


main()
