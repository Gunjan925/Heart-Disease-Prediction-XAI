from flask import Flask,request,render_template
from utils.processor import preprocess_data
from utils.prediction import make_prediction
from utils.explaination import generate_explanation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_dict = {
        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Cholesterol": float(request.form["Cholesterol"]),
        "Blood Pressure": float(request.form["Blood Pressure"]),
        "Heart Rate": float(request.form["Heart Rate"]),
        "Smoking": request.form["Smoking"],
        "Alcohol Intake": request.form["Alcohol Intake"],
        "Exercise Hours": float(request.form["Exercise Hours"]),
        "Family History": request.form["Family History"],
        "Diabetes": request.form["Diabetes"],
        "Obesity": request.form["Obesity"],
        "Stress Level": float(request.form["Stress Level"]),
        "Blood Sugar": float(request.form["Blood Sugar"]),
        "Exercise Induced Angina": request.form["Exercise Induced Angina"],
        "Chest Pain Type": request.form["Chest Pain Type"]
    }

    input_data , original_input_data = preprocess_data(input_dict=input_dict)

    prediction_data = make_prediction(input_data=input_data)

    explanations = generate_explanation(input_data=input_data,original_input_data=original_input_data)

    return render_template("result.html",has_disease=prediction_data["has_disease"],risk_probability=prediction_data["probability"],recommendation=prediction_data["recommendation"],explanations=explanations)


if __name__ == "__main__":
    app.run(debug=True)