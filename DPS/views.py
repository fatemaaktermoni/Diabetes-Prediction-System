from random import random

from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')

def result(request):
    dataset = pd.read_csv('diabetes_2.csv')
    X = dataset.drop(columns='Outcome', axis=1)
    Y = dataset['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    value1 = float(request.GET['n1'])
    value2 = float(request.GET['n2'])
    value3 = float(request.GET['n3'])
    value4 = float(request.GET['n4'])
    value5 = float(request.GET['n5'])
    value6 = float(request.GET['n6'])
    value7 = float(request.GET['n7'])
    value8 = float(request.GET['n8'])

    DPS_predict = model.predict([[value1, value2, value3, value4, value5, value6,value7, value8]])

    result1 = ""
    if DPS_predict == [1]:
        result1 = "High Chance of Diabetics"
    else:
        result1 = "No Diabetics"

    return render(request, 'predict.html', {'result2': result1})
