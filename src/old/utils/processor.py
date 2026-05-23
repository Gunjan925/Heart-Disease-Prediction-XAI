import pandas as pd
import joblib

encoder = joblib.load("../../models/heart/old_models/old_one_hot_encoder.pkl")
scaler = joblib.load("../../models/heart/old_models/old_standard_scaler.pkl")

def preprocess_data(input_dict):
    input_data = pd.DataFrame([input_dict])
    original_input_data = input_data.copy()
    gender_map = {'Female':0,'Male':1} 
    binary_map = {'Yes':1,'No':0}
    bin_cols = ['Family History','Diabetes','Obesity','Exercise Induced Angina']
    one_hot_cols = ['Smoking','Alcohol Intake','Chest Pain Type']
    num_cols = ['Age','Cholesterol','Blood Pressure','Heart Rate','Exercise Hours','Stress Level','Blood Sugar']
    input_data['Gender'] = input_data['Gender'].map(gender_map)
    for col in bin_cols:
        input_data[col] = input_data[col].map(binary_map)
    encoded_array = encoder.transform(input_data[one_hot_cols])
    encoded_df = pd.DataFrame(encoded_array,columns=encoder.get_feature_names_out(one_hot_cols))
    input_data = input_data.drop(columns = one_hot_cols)
    input_data = pd.concat([input_data,encoded_df],axis=1)
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    return input_data,original_input_data