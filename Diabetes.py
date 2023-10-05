import os
import pandas as pd
from flask import Flask, render_template, request
from src.utils import load_object

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/<disease_name>', methods=['GET', 'POST'])
def predict(disease_name):
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
        result = "Positive" if prediction == 1 else "Negative"

        return render_template(f'{disease_name}.html', result=result)

    return render_template(f'{disease_name}.html')

if __name__ == '__main__':
    app.run(debug=True)