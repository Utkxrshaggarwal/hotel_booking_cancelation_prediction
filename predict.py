"""
Hotel Booking Cancellation - Predict from user input
====================================================
Run train_model.py first, then:  python predict.py

Press Enter on any question to use the default (typical) value.
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hotel_cancellation_model.joblib"

# Short hints so the questions are easy to answer
HINTS = {
    "lead_time": "days between booking and arrival",
    "adr": "average daily rate / price per night",
    "arrival_date_week_number": "week of the year, 1-53",
    "arrival_date_day_of_month": "day of the month, 1-31",
    "arrival_date_month": "e.g. July",
    "stays_in_weekend_nights": "number of weekend nights",
    "stays_in_week_nights": "number of week nights",
    "adults": "number of adults",
    "children": "number of children",
    "babies": "number of babies",
    "country": "3-letter code, e.g. PRT, GBR, FRA, IND",
    "agent": "travel agent ID, 0 = no agent",
    "company": "company ID, 0 = none",
    "previous_cancellations": "earlier bookings the guest canceled",
    "booking_changes": "number of changes made to the booking",
    "total_of_special_requests": "number of special requests",
    "required_car_parking_spaces": "parking spaces requested",
    "days_in_waiting_list": "days on the waiting list",
}


def ask_numeric(feature: str, default: float) -> float:
    hint = f" ({HINTS[feature]})" if feature in HINTS else ""
    while True:
        raw = input(f"  {feature}{hint} [default {default:g}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("    Please enter a number.")


def ask_categorical(feature: str, classes: list, default: str) -> str:
    hint = f" ({HINTS[feature]})" if feature in HINTS else ""
    if len(classes) <= 15:
        print(f"  {feature}{hint}\n    options: {', '.join(classes)}")
    lookup = {c.lower(): c for c in classes}
    while True:
        raw = input(f"  {feature}{hint} [default {default}]: ").strip()
        if raw == "":
            return default
        if raw.lower() in lookup:
            return lookup[raw.lower()]
        print(f"    Not recognised. Examples: {', '.join(classes[:8])} ...")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run  python train_model.py  first.")

    art = joblib.load(MODEL_PATH)
    model, scaler, pca = art["model"], art["scaler"], art["pca"]
    encoders = art["label_encoders"]
    features = art["selected_features"]
    cat_features = art["categorical_features"]
    defaults = art["defaults"]

    print("=" * 60)
    print(" HOTEL BOOKING CANCELLATION PREDICTOR")
    print(f" Model: {art['model_name']}   (test F1: {art['test_metrics']['F1-score']:.3f})")
    print("=" * 60)

    while True:
        print("\nEnter the booking details (Enter = default):\n")
        row = {}
        for feat in features:
            if feat in cat_features:
                classes = list(encoders[feat].classes_)
                row[feat] = ask_categorical(feat, classes, defaults[feat])
            else:
                row[feat] = ask_numeric(feat, defaults[feat])

        X_new = pd.DataFrame([row], columns=features)
        for col in cat_features:
            X_new[col] = encoders[col].transform(X_new[col].astype(str))

        X_pca = pca.transform(scaler.transform(X_new))
        pred = int(model.predict(X_pca)[0])
        prob = float(model.predict_proba(X_pca)[0, 1])

        print("\n" + "-" * 60)
        if pred == 1:
            print(f" RESULT: Likely to be CANCELED   (probability {prob:.1%})")
        else:
            print(f" RESULT: Likely to be KEPT        (cancel probability {prob:.1%})")
        print("-" * 60)

        again = input("\nPredict another booking? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
