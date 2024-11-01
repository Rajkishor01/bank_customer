# 
import numpy as np
from flask import Flask,request,jsonify,render_template
import joblib

# Starting of the app
app=Flask(__name__)

# Loading the models
model =joblib.load('rfc_model.pkl')
encoders =joblib.load('encoders.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    input_data = {
            'person_age': int(request.form['person_age']),
            'person_income': int(request.form['person_income']),
            'person_home_ownership': request.form['person_home_ownership'],
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_intent': request.form['loan_intent'],
            'loan_grade': request.form['loan_grade'],
            'loan_amnt': int(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_default_on_file': request.form['cb_person_default_on_file'],
            'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length'])
            }

    cat_cols=['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

    encoded_cols = []
    try:
        for column, value in input_data.items():
            if column in cat_cols and column in encoders:
                if value in encoders[column].classes_:
                    encoded_value = encoders[column].transform([value])[0]
                else:
                    print(f"Unseen label '{value}' for '{column}', assigning placeholder.")
                    encoded_value = -1
                encoded_cols.append(encoded_value)
                print(f"Encoded '{value}' for '{column}' as {encoded_value}")
            else:
                encoded_cols.append(float(value))
                
    except Exception as e:
        print("Error during encoding or processing input data:", e)
        return "Error in encoding or processing input data", 400

    #converting encoded columns into a numpy array for model predictions
    transformed_input = np.array(encoded_cols, dtype=float).reshape(1, -1)



    prediction = model.predict(transformed_input)[0]
    pred_probability = model.predict_proba(transformed_input)[0]

    confidence_threshold = 50
    confidence_level = pred_probability[1] * 100

    if confidence_level >= confidence_threshold:
        pred_class = 'Default'
        binary_prediction = 1
    else:
        pred_class = 'Non-Default'
        binary_prediction = 0

    class_percentage = confidence_level if binary_prediction == 1 else (100 - confidence_level)

    return render_template(
    'prediction.html',
    predicted_class=pred_class,
    class_percentage=class_percentage,
    binary_prediction=prediction
)





if __name__=='__main__':
    app.run(debug=True)