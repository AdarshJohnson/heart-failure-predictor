from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the exact order of features expected by the model
feature_names = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            # Extract values in correct order
            features = [float(request.form[feature]) for feature in feature_names]

            # Scale and predict
            scaled_features = scaler.transform([features])
            pred = model.predict(scaled_features)[0]

            # Interpret result
            prediction = "High Risk 💔" if pred == 1 else "Low Risk ❤️"
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
