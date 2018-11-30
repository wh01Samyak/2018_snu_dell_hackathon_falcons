from flask import request, redirect
from flask import Flask
from flask import jsonify
from flask import send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.layers import Dense, Embedding, BatchNormalization, Input
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  
import json

data = pd.read_csv('Data_v3.csv')
data = data.drop(['Model Age'],axis = 1)
data = data.drop(['S. No.'],axis = 1)
data = data.drop(['Offer Number'],axis = 1)
model_list = pd.read_csv('model.csv')

app = Flask(__name__)

def prediction_product(arr) :
    data = pd.read_csv('Data_v3.csv')
    data = data.drop(['Model Age'],axis = 1)
    data = data.drop(['S. No.'],axis = 1)
    data = data.drop(['Offer Number'],axis = 1)
    model_list = pd.read_csv('model.csv')
    model_list['Price'] = model_list['Price']/1000
    model_T = model_list.iloc[:,1:].T
    model_T.insert(loc=0,column='Test',value=arr)
    model_coer = model_T.corr()
    list_coer = list(model_coer['Test'])
    temp = 0
    index = 0
    for i in range(len(list_coer)-1) :
        if temp < list_coer[i+1] :
            temp = list_coer[i+1]
            index = i
    existing_model = model_list.loc[index][0]

    prediction = []
    a=[0]*8
   	
    for index,row in data.iterrows():
        if row['Product Model'] == existing_model and row['Year']>2016 and row['Year']<2019:
            if row['Area']==1:
                a[0]=a[0]+1
            if row['Area']==2:
                a[1]=a[1]+1
            if row['Area']==3:
                a[2]=a[2]+1
            if row['Area']==4:
                a[3]=a[3]+1
            if row['Area']==5:
                a[4]=a[4]+1
            if row['Area']==6:
                a[5]=a[5]+1
            if row['Area']==7:
                a[6]=a[6]+1
            if row['Area']==8:
                a[7]=a[7]+1
    
    names = 'This model is similar to existing model'+existing_model
    market = np.asarray(a)
    main_list = []
    for i in range(data['Area'].nunique()): 
        data_dict = {'x':i+1,'y':a[i]}
        main_list.append(data_dict)
	
    main_dict = {'title':names,'xaxis':'Areas','yaxis':'Volume of sales','data':main_list}
    return main_dict

@app.route('/adoption.html', methods = ['POST','GET'])
def signup():
    trigger = 0
    username = request.form['username']
    password = request.form['password']
    f = open("user_pass.txt", "r")
    for x in f:
        data = x.split(" ")
        if data[0]==username and data[1]==password:
            trigger = 1
            break
    if trigger == 1:
    	return send_file('adoption.html')
    else:
        return send_file('error.html')

@app.route('/new_data.html', methods = ['POST','GET'])
def add_data():
    return send_file('new_data.html')


@app.route('/pred.html', methods = ['POST','GET'])
def model():
    ram = int(request.form['ram'])
    proc = int(request.form['proc'])
    graphic= int(request.form['graphic'])
    price = int(request.form['price'])
    battery = int(request.form['battery'])
    arr = [ram,proc,graphic,battery,(price/1000)]
    print(arr)
    main_dict = prediction_product(arr)
    with open('pred.json', 'w') as fp:
        json.dump(main_dict, fp)
    #send_file('pred.json')
    #return send_file('pred.json')
    #return "fuck abhishek"
    return "fg"
'''
@app.route('/pred.json', methods = ['POST','GET'])
def model():
    ram = int(request.form['ram'])
    proc = int(request.form['proc'])
    graphic= int(request.form['graphic'])
    price = int(request.form['price'])
    battery = int(request.form['battery'])
    arr = [ram,proc,graphic,battery,(price/1000)]
    main_dict = prediction_product(arr)
    # with open('pred.json', 'w') as fp:
    #     json.dump(main_dict, fp)
    return json.dump(main_dict)
'''

@app.route('/index.html')
def index():
	return send_file('index.html')

if __name__ == '__main__':
    app.run(port=9999, debug=True)
