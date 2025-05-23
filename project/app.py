from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.sav")

# Define model input columns (after transformation)
model_columns = ['SeniorCitizen', 'TotalCharges', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PaperlessBilling_No',
       'PaperlessBilling_Yes', 'MultipleLines_No',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'DeviceProtection_No', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_Yes', 'Contract_Month-to-month',
       'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# Dummy scaler fitted on training data (replace with your original if saved)
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': float(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    df = pd.DataFrame([input_data])

    # Encode binary columns manually if needed (here we use one-hot)
    df.drop(['gender', 'PhoneService', 'OnlineSecurity', 'OnlineBackup',
             'StreamingTV', 'StreamingMovies'], axis=1, inplace=True)  # Drop unused for your model

    # One-hot encode
    df_encoded = pd.get_dummies(df)

    # Ensure all model columns are present
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[model_columns]  # Ensure column order

    # Scale TotalCharges
    df_encoded['TotalCharges'] = scaler.fit_transform(df_encoded[['TotalCharges']])

    # Predict
    prediction = model.predict(df_encoded)[0]

    result = "Customer is likely to Churn." if prediction == 1 else "Customer is likely to Stay."

    return render_template('home.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
