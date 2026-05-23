import pandas as pd
import joblib

encoder = joblib.load("../../models/heart/new_models/new_one_hot_encoder.pkl")
scaler = joblib.load("../../models/heart/new_models/new_standard_scaler.pkl")

def preprocess_data(input_dict):
    input_data = pd.DataFrame([input_dict])

    original_input_data = input_data.copy()

    binary_map = {"Yes":1 , "No":0}
    gender_map = {"Male":1,"Female":0}
    chest_pain_map = {"Typical Angina":1 , "Atypical Angina": 2 , "Non-anginal Pain": 3 , "Asymptomatic": 4}
    restecg_map = {"Normal": 0 , "ST-T Wave Abnormality": 1 , "Left Ventricular Hypertrophy": 2}
    slope_map = {"Upsloping": 1 , "Flat": 2 , "Downsloping": 3}
    thal_map = {"Normal": 3 , "Fixed Defect": 6 , "Reversible Defect": 7}

    num_cols = ['age','trestbps','chol','thalach','oldpeak','ca']
    binary_cols = ['fbs','exang']
    categorical_cols = ['cp','restecg','slope','thal']

    input_data["sex"] = input_data["sex"].map(gender_map)
    for col in binary_cols:
        input_data[col] = input_data[col].map(binary_map)

    input_data["cp"] = input_data["cp"].map(chest_pain_map)
    input_data["restecg"] = input_data["restecg"].map(restecg_map)
    input_data["slope"] = input_data["slope"].map(slope_map)
    input_data["thal"] = input_data["thal"].map(thal_map)
    encoded_array = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array,columns=encoder.get_feature_names_out(categorical_cols))
    input_data = input_data.drop(columns=categorical_cols)
    input_data = pd.concat([input_data,encoded_df], axis=1)
    
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    return input_data,original_input_data