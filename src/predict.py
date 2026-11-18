import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# XGBoost models: loaded from JSON (portable across machines)
xgb_s1 = XGBClassifier()
xgb_s1.load_model(str(MODEL_DIR / "xgb_s1.json"))
s2_model = XGBClassifier()
s2_model.load_model(str(MODEL_DIR / "xgb_s2.json"))

# Plain scikit-learn objects: joblib is fine for these
encoder = joblib.load(MODEL_DIR / "onehot_encoder.joblib")
label_enc = joblib.load(MODEL_DIR / "label_encoder.joblib")
with open(MODEL_DIR / "model_meta.json") as f:
    meta = json.load(f)
with open(MODEL_DIR / "road_to_area.json") as f:
    road_to_area = json.load(f)

CLASSES = [str(c) for c in label_enc.classes_]   # order of probability columns
CAT_COLS = ["Road/Intersection Name", "Weather Conditions", "Weekday"]


def predict_congestion(road, incidents, weather, roadwork, date,
                       traffic_volume=None, avg_speed=None):
    if (traffic_volume is None) != (avg_speed is None):
        raise ValueError("Give both Traffic Volume and Average Speed, or neither.")
    nowcast = traffic_volume is not None

    d = pd.to_datetime(date)
    raw = pd.DataFrame([{
        "Road/Intersection Name": road,
        "Weather Conditions": weather,
        "Weekday": d.day_name(),
    }])
    cat = pd.DataFrame(encoder.transform(raw),
                       columns=encoder.get_feature_names_out(CAT_COLS))

    row = cat.copy()
    row["Incident Reports"] = incidents
    row["Roadwork_Flag"] = 1 if roadwork == "Yes" else 0
    row["Is_Weekend"] = int(d.dayofweek >= 5)
    row["Month"] = d.month

    if nowcast:
        row["Traffic Volume"] = traffic_volume
        row["Average Speed"] = avg_speed
        columns, model, mode = meta["s2_columns"], s2_model, "Nowcast (S2)"
    else:
        columns, model, mode = meta["s1_columns"], xgb_s1, "Forecast (S1)"

    row = row[columns]
    proba = model.predict_proba(row)[0]
    probs = {c: float(p) for c, p in zip(CLASSES, proba)}
    label = max(probs, key=probs.get)
    return {"mode": mode, "area": road_to_area.get(road),
            "prediction": label, "probabilities": probs}