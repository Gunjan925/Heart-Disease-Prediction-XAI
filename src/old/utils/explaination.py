import joblib

lr_explainer = joblib.load("../../models/heart/old_models/old_lr_shap_explainer.pkl")

def generate_explanation(input_data,original_input_data):
    lr_shap_values = lr_explainer(input_data)
    explanations = []
    for i,col in enumerate(input_data.columns):
        impact = lr_shap_values.values[0][i]
        if col in original_input_data.columns:
            value = original_input_data.iloc[0][col]
            if impact>0:
                # explanations.append([col,str(value),"Increased risk of heart disease"])
                explanations.append([col,str(value),"Increased contribution towards heart disease prediction"])
            elif impact<0:
                # explanations.append([col,str(value),"Decreased risk of heart disease"])
                explanations.append([col,str(value),"Decreased contribution towards heart disease prediction"])
        else:
            split_col = col.split("_")
            original_feature = split_col[0]
            encoded_value = "_".join(split_col[1:])
            if input_data.iloc[0][col] == 1:
                if impact > 0:
                    # explanations.append([original_feature,encoded_value,"Increased risk of heart disease"])
                    explanations.append([original_feature,encoded_value,"Increased contribution towards heart disease prediction"])
                elif impact < 0:
                    # explanations.append([original_feature,encoded_value,"Decreased risk of heart disease"])
                    explanations.append([original_feature,encoded_value,"Decreased contribution towards heart disease prediction"])
    return explanations