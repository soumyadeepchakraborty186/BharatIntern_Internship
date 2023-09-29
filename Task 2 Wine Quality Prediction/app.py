from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = joblib.load('wine_quality_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        input_data = []
        for feature in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol']:
            input_data.append(float(request.form[feature]))
        
        prediction = model.predict([input_data])[0]
        
        if prediction > 5.355:
            quality = "Good Quality Wine"
        elif prediction == 5.3:
            quality = "Not as much good"
        else:
            quality = "Bad Quality Wine"
        
        return render_template('index.html', prediction=quality)

if __name__ == '__main__':
    app.run(debug=True)
