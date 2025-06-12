from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Predicted House Price: ${output * 1000:.2f}')

if __name__ == '__main__':
    app.run(debug=True)