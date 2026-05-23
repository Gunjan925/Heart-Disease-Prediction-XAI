import joblib

logistic_regression_model = joblib.load("../../models/heart/new_models/new_logistic_regression_model.pkl")

def make_prediction(input_data):
    result = logistic_regression_model.predict(input_data)[0]
    probability = logistic_regression_model.predict_proba(input_data)[0][1]
    has_disease = ("YES" if result==1 else "NO")
    if probability>=0.70:
        recommendation = ("Immediate doctor consultation is advised.")
    elif probability>=0.40:
        recommendation = ("Regular doctor consultation is advised.")
    else:
        recommendation = ("No immediate consultation needed.")
    return {"has_disease":has_disease,"probability":f"{probability:.2f}","recommendation":recommendation}