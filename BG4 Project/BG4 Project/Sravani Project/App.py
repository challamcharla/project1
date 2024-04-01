from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('Kidney.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('image.html')

@app.route("/predict", methods=['POST'])
def predict():
    print(request.form)
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    if float(output)==1.0:
        return render_template('result2.html', prediction=prediction)
    else:
        return render_template('result.html', prediction=prediction)  

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
