from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('linear_regression_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':
        sqft_living = int(request.form['sqft_living'])
        bedrooms = int(request.form['bedrooms'])
        yr_built = int(request.form['yr_built'])
        new_data = pd.DataFrame({'sqft_living': [sqft_living], 'bedrooms': [bedrooms], 'yr_built': [yr_built]})

        predicted_price = model.predict(new_data)
        rounded_price = round(predicted_price[0], 2)

        return render_template('index.html', price=rounded_price)

    return render_template('index.html', price=None)

if __name__ == '__main__':
    app.run(debug=True)
