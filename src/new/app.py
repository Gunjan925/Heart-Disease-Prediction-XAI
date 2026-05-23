from flask import Flask,request,render_template
from utils.prepocess import preprocess_data
from utils.prediction import make_prediction
from utils.explaination import generate_explanation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_dict = {
    "age": int(request.form["age"]),
    "sex": request.form["sex"],
    "cp": request.form["cp"],
    "trestbps": float(request.form["trestbps"]),
    "chol": float(request.form["chol"]),
    "fbs": request.form["fbs"],
    "restecg": request.form["restecg"],
    "thalach": float(request.form["thalach"]),
    "exang": request.form["exang"],
    "oldpeak": float(request.form["oldpeak"]),
    "slope": request.form["slope"],
    "ca": float(request.form["ca"]),
    "thal": request.form["thal"]
    }

    input_data , original_input_data = preprocess_data(input_dict=input_dict)

    prediction_data = make_prediction(input_data=input_data)

    explanations = generate_explanation(input_data=input_data,original_input_data=original_input_data)

    return render_template("result.html",has_disease=prediction_data["has_disease"],risk_probability=prediction_data["probability"],recommendation=prediction_data["recommendation"],explanations=explanations)


if __name__ == "__main__":
    app.run(debug=True)