import joblib

lr_explainer = joblib.load("../../models/heart/new_models/new_lr_shap_explainer.pkl")

def generate_explanation(input_data,original_input_data):
    lr_shap_values = lr_explainer(input_data)
    explanations = []
    for i, col in enumerate(input_data.columns):
        impact = lr_shap_values.values[0][i]

        if col in original_input_data.columns:
            value = original_input_data.iloc[0][col]
            if impact > 0:
                # explanations.append([col,str(value),"Increased risk of heart disease"])
                explanations.append([col,str(value),"Increased contribution towards heart disease prediction"])
            elif impact < 0:
                # explanations.append([col,str(value),"Decreased risk of heart disease"])
                explanations.append([col,str(value),"Decreased contribution towards heart disease prediction"])

        else:
            split_col = col.split("_")
            original_feature = split_col[0]
            encoded_value = "_".join(split_col[1:])
            if input_data.iloc[0][col] == 1:
                if original_feature == "cp":
                    cp_reverse_map = {"1.0": "Typical Angina","2.0": "Atypical Angina","3.0": "Non-anginal Pain","4.0": "Asymptomatic"}
                    encoded_value = cp_reverse_map.get(str(encoded_value),encoded_value)
                elif original_feature == "restecg":
                    restecg_reverse_map = {"0.0": "Normal","1.0": "ST-T Wave Abnormality","2.0": "Left Ventricular Hypertrophy"}
                    encoded_value = restecg_reverse_map.get(str(encoded_value),encoded_value)
                elif original_feature == "slope":
                    slope_reverse_map = {"1.0": "Upsloping","2.0": "Flat","3.0": "Downsloping"}
                    encoded_value = slope_reverse_map.get(str(encoded_value),encoded_value)
                elif original_feature == "thal":
                    thal_reverse_map = {"3.0": "Normal","6.0": "Fixed Defect","7.0": "Reversible Defect"}
                    encoded_value = thal_reverse_map.get(str(encoded_value),encoded_value)

                if impact > 0:
                    # explanations.append([original_feature,encoded_value,"Increased risk of heart disease"])
                    explanations.append([original_feature,encoded_value,"Increased contribution towards heart disease prediction"])
                elif impact < 0:
                    # explanations.append([original_feature,encoded_value,"Increased risk of heart disease"])
                    explanations.append([original_feature,encoded_value,"Decreased contribution towards heart disease prediction"])

    return explanations