from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("heart_cleveland_upload.csv")
df = df.rename(columns={'condition':'target'})
X = df.drop(['target','fbs','exang'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
selected_model=GaussianNB()
selected_model.fit(X_train, y_train)
y_pred = selected_model.predict(X_test)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = float(request.form['age'])
        sex = float(request.form.get('sex'))
        cp = float(request.form.get('cp'))
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        # fbs = float(request.form.get('fbs'))
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        # exang = float(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form.get('slope'))
        ca = float(request.form['ca'])
        thal = float(request.form.get('thal'))
        
        data = np.array([[age,sex,cp,trestbps,chol,restecg,thalach,oldpeak,slope,ca,thal]])
        print(data)
        my_prediction = selected_model.predict(data)
        return render_template('result.html', prediction=my_prediction)

@app.route('/bot', methods=['GET','POST'])
def chatbot():
    genai.configure(api_key='AIzaSyBF6bKuEG9LhwPM7X_LaDdUipFvjw56GYc')
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content()
    print(response.text)
        

if __name__ == '__main__':
	app.run(debug=True)