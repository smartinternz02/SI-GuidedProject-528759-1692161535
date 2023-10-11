from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
model=pickle.load(open(r"C:\Users\lenovo\Dataset\Flask\HDI.pkl",'rb'))
le=pickle.load(open(r"C:\Users\lenovo\Dataset\Flask\label_encoder.pkl",'rb'))
app=Flask(__name__)
@app.route('/')
def about():
    return render_template('home.html')

@app.route('/home', methods=['POST','GET'])
def home():
    return render_template('home.html')

@app.route('/indexnew', methods=['POST', 'GET'])
def indexnew():
    return render_template('indexnew.html')
@app.route('/resultnew', methods=['POST', 'GET'])
def resultnew():
    input_features = []
    country = request.form['country']
    input_features.append(le.transform([country])[0])
    numerical_features = ['life_expectancy', 'mean_years_of_schooling', 'gni_per_capita', 'internet_users']
    for feature in numerical_features:
        input_features.append(float(request.form[feature]))

    features_name = ['Country', 'Life expectancy', 'Mean years of schooling', 'Gross national income (GNI) per capita', 'Internet users']
    df2 = pd.DataFrame([input_features], columns=features_name)
    output = model.predict(df2)
    y_pred = round(output[0][0], 2)

    if 0.3 <= y_pred <= 0.4:
        prediction_text = 'Low HDI ' + str(y_pred)
    elif 0.4 <= y_pred <= 0.7:
        prediction_text = 'Medium HDI ' + str(y_pred)
    elif 0.7 <= y_pred <= 0.8:
        prediction_text = 'High HDI ' + str(y_pred)
    elif 0.8 <= y_pred <= 0.94:
        prediction_text = 'Very High HDI ' + str(y_pred)
    else:
        prediction_text = 'The given values do not match the range of values of the model. Try giving the values in the mentioned range: ' + str(y_pred)

    return render_template('resultnew.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)