from dataclasses import dataclass
from distutils.log import debug
from tkinter.tix import Tree
from itsdangerous import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pymongo import MongoClient
from config import password

#create flask app

app = Flask(__name__)

client = MongoClient(f"mongodb+srv://humbertorodriguez:{password}@cluster0.yyllp.mongodb.net/data_analytics?retryWrites=true&w=majority")
database = client['data_analytics']
collection = database['train_data']

#load the pickle model
model = pickle.load(open("model.pkl", "rb"))

#define home page
@app.route("/")
def home():
    #return render_template("index.html", data=data)
    return render_template("welcome.html")

#this end point will return the database from MongoDB.
@app.route("/database",methods=["POST", "GET"])
def database():
    data = collection.find().limit(10)
    return render_template("database.html", data=data)

#this endpoint is to show the model
@app.route("/show_model")
def show_model():
    #return render_template("index.html", data=data)
    return render_template("show_model.html")

#this endpoint will perform the calculation
@app.route("/predict",methods=["POST", "GET"])
def predict():

    #make sure all the entries are casted into integer
    integer_features = [x for x in request.form.values()]

    #convert the entries into an array
    features = [np.array(integer_features)]

    #use features in model
    prediction = model.predict(features)

    # use it to round and use , separator
    output = f"${prediction[0]:,.2f}"
    
    return render_template("show_model.html", prediction_text =  f"The price of the house is {output}")#"The price of the house is ${}".format(output))

if __name__ == "__main__":
    app.run(debug=True)