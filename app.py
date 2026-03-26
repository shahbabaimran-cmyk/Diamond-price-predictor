from flask import Flask, request, render_template
import pickle
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and features
model = pickle.load(open('diamond_price_model.pkl', 'rb'))
features = json.load(open('model_features.json', 'r'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ 1. Get data from form
    data = request.form

    # ✅ 2. Prepare empty row with all features
    row = {col: 0 for col in features}

    # ✅ 3. Fill numerical feature
    row['carat'] = float(data['carat'])

    # ✅ 4. One-hot encode color and clarity
    color = data['color']
    clarity = data['clarity']
    if color in row:
        row[color] = 1
    if clarity in row:
        row[clarity] = 1

    # ✅ 5. Convert to DataFrame
    df = pd.DataFrame([row])

    # ✅ 6. Predict
    prediction = model.predict(df)[0]

    # ✅ 7. Return result to HTML
    return render_template('index.html', prediction_text=f"💰 Predicted Price: ${prediction:,.2f}")

