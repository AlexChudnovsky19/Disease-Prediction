import os
import pandas as pd
from flask import Flask, render_template, request
from src.utils import load_object

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('home.html')

def predict_heart_failure(disease_name):
    if request.method == 'POST':
        age = float(request.form.get('Age'))
        sex = int(request.form.get('Sex'))
        chest_pain_type = request.form.get('ChestPainType')
        resting_bp = float(request.form.get('RestingBP'))
        cholesterol = float(request.form.get('Cholesterol'))
        fasting_bs = request.form.get('FastingBS')
        resting_ecg = request.form.get('RestingECG')
        max_hr = float(request.form.get('MaxHR'))
        exercise_angina = request.form.get('ExerciseAngina')
        oldpeak = float(request.form.get('Oldpeak'))
        st_slope = request.form.get('ST_Slope')

        # Load the preprocessor for the specific disease using pickle
        preprocessor_filename = os.path.join('artifacts', f'preprocessor_{disease_name}.pkl')
        loaded_preprocessor = load_object(preprocessor_filename)
    

        # Preprocess the input data
        data = [[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]]
        input_data = pd.DataFrame(
    data=[{
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }]
)
        preprocessed_data = loaded_preprocessor.transform(input_data)

        # Load the machine learning model for the specific disease using pickle
        model_filename = os.path.join('artifacts', f'model_{disease_name}.pkl')
        loaded_model = load_object(model_filename)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(preprocessed_data)

        # Process the prediction and format the result
        result_for_heart_failure = "Positive" if prediction == 1 else "Negative"

    return result_for_heart_failure

def predict_stroke(disease_name):
    if request.method == 'POST':
        gender = request.form.get('gender')
        age = request.form.get('age')
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        ever_married = request.form.get('ever_married')
        work_type = request.form.get('work_type')
        Residence_type = request.form.get('Residence_type')
        avg_glucose_level = request.form.get('avg_glucose_level')
        bmi = request.form.get('bmi')
        smoking_status = request.form.get('smoking_status')
        

        # Load the preprocessor for the specific disease using pickle
        preprocessor_filename = os.path.join('artifacts', f'preprocessor_{disease_name}.pkl')
        loaded_preprocessor = load_object(preprocessor_filename)
    

        # Preprocess the input data
        data = [[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]]
        input_data = pd.DataFrame(
    data=[{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }]
)
        
        preprocessed_data = loaded_preprocessor.transform(input_data)

        # Load the machine learning model for the specific disease using pickle
        model_filename = os.path.join('artifacts', f'model_{disease_name}.pkl')
        loaded_model = load_object(model_filename)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(preprocessed_data)

        # Process the prediction and format the result
        result_for_stroke = "Positive" if prediction == 1 else "Negative"

    return result_for_stroke

def predict_diabetes(disease_name):
    if request.method == 'POST':
        gender = request.form.get('gender')
        age = request.form.get('age')
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        smoking_history = request.form.get('smoking_history')
        bmi = request.form.get('bmi')
        HbA1c_level  = request.form.get('HbA1c_level')
        blood_glucose_level = request.form.get('blood_glucose_level')
        
        # Load the preprocessor for the specific disease using pickle
        preprocessor_filename = os.path.join('artifacts', f'preprocessor_{disease_name}.pkl')
        loaded_preprocessor = load_object(preprocessor_filename)
    

        # Preprocess the input data
        data = [[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level]]
        input_data = pd.DataFrame(
    data=[{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
    }]
)
        preprocessed_data = loaded_preprocessor.transform(input_data)

        # Load the machine learning model for the specific disease using pickle
        model_filename = os.path.join('artifacts', f'model_{disease_name}.pkl')
        loaded_model = load_object(model_filename)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(preprocessed_data)

        # Process the prediction and format the result
        result_for_diabetes = "Positive" if prediction == 1 else "Negative"

    return result_for_diabetes

@app.route('/<disease_name>', methods=['GET', 'POST'])
def predict_disease(disease_name):
    if request.method == 'POST':
        if disease_name == 'Heart_Failure':
            result = predict_heart_failure(disease_name)
            return render_template('Heart_Failure.html', result=result)
        elif disease_name == 'Stroke':
            result = predict_stroke(disease_name)
            return render_template('Stroke.html', result=result)
        elif disease_name == 'Diabetes':
            result = predict_diabetes(disease_name)
            return render_template('Diabetes.html', result=result)
        # Add other diseases here using elif statements

    return render_template(f'{disease_name}.html')

if __name__ == '__main__':
    app.run(debug=True)