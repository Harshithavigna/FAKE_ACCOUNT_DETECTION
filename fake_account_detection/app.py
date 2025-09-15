from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from pathlib import Path
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key_for_demo")  # use env var if set

MODEL_PATH = Path(__file__).parent / "model.pkl"
model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print("⚠️ Failed to load model:", e)
        model = None
else:
    print("⚠️ No model.pkl found — running with fallback heuristic.")

FEATURES = ["followers","following","statuses","account_age_days",
            "has_profile_pic","default_profile","verified","listed_count"]

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        try:
            # parse inputs (with defaults)
            defaults = {
                "followers":"50","following":"100","statuses":"200","account_age_days":"365",
                "has_profile_pic":"1","default_profile":"0","verified":"0","listed_count":"0"
            }
            data = [float(request.form.get(f, defaults[f]).strip() or defaults[f]) for f in FEATURES]
            X = np.array([data])

            if model:
                pred = int(model.predict(X)[0])
                try:
                    prob = float(model.predict_proba(X)[0][1])
                except Exception:
                    prob = None
                label = "Fake" if pred == 1 else "Real"
            else:
                # fallback heuristic if no model available
                score = 0
                if data[0] < 20: score += 1
                if data[2] < 10: score += 1
                if data[3] < 30: score += 1
                if data[4] == 0: score += 1
                label = "Fake" if score >= 2 else "Real"
                prob = None

            return render_template(
                "index.html",
                result=True,
                label=label,
                prob=prob,
                values=dict(zip(FEATURES, data))
            )
        except Exception as e:
            flash(f"Error parsing input: {e}")
            return redirect(url_for("index"))

    return render_template("index.html", result=False, values={})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
