from tokenize import group
import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from pymongo import MongoClient
from config import password

#create flask app

app = Flask(__name__)

client = MongoClient(f"mongodb+srv://humbertorodriguez:{password}@cluster0.yyllp.mongodb.net/data_analytics?retryWrites=true&w=majority")
database = client['data_analytics']
collection = database['selected_features']

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

    #query MongoDB and extract the first 10 results of the features
    data = collection.find().limit(10)

    return render_template("database.html", data=data)

#this endpoint is to show the model
@app.route("/show_model")
def show_model():
 
    return render_template("show_model.html")

#this endpoint will perform the calculation and then will show it back in show_model
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
#endpoint for dashboard
@app.route("/dashboard",methods=["POST", "GET"])
def dashboard():
    
    # BAR CHART
    # extract data from MongoDB and insert them in two lists
    year_list = []
    overall_qual_list = []
    for dict in collection.find({}):
        year_list.append(dict["YearBuilt"])
        overall_qual_list.append(dict["OverallQual"])

    #zip to create an zip object and then insert it in a list (list of tuples)
    year_qual_list = list(zip(year_list, overall_qual_list))

    # create a DF out of the year_qual_list
    df = pd.DataFrame(year_qual_list, columns =['Year Built', 'Overall Quality'])
    
    #groupby 'Overall Quality' and make a count out of the 'Year Built' 
    df = df.groupby('Overall Quality', as_index=False)['Year Built'].count()

    #rename the df column with the count() from 'Year Built' to 'Count of Houses by Overall Quality'
    df = df.rename(columns={'Year Built':'Count of Houses by Overall Quality'})

    labels = df['Overall Quality'].tolist()
    values = df['Count of Houses by Overall Quality'].tolist()

    zip_list = list(zip(labels, values))

    labels = [row[0] for row in zip_list]
    values = [row[1] for row in zip_list]

    #LINE CHART
    sale_price_list = []
    year_line_list =[]
    for dict in collection.find({}):
        year_line_list.append(dict["YearBuilt"])
        sale_price_list.append(dict["SalePrice"])

    #zip to create an zip object and then insert it in a list (list of tuples)
    year_sale_price_list = list(zip(year_line_list, sale_price_list))

    # create a DF out of the year_qual_list
    df_line = pd.DataFrame(year_sale_price_list, columns =['Year Built', 'Sale Price'])

    #groupby 'Year Built' and make a count out of the 'Year Built' 
    df_line = df_line.groupby('Year Built', as_index=False)['Sale Price'].mean()

    df_line = df_line.rename(columns={'Sale Price':'Avg Sale Price'})

    labels_line = df_line['Year Built'].tolist()
    values_line = df_line['Avg Sale Price'].tolist()

    zip_list_line = list(zip(labels_line, values_line))

    labels_line = [row[0] for row in zip_list_line]
    values_line = [row[1] for row in zip_list_line]

    # PIE CHART
    gc =[]
    oq = []
    for dict in collection.find({}):
        gc.append(dict["GarageCars"])
        oq.append(dict["OverallQual"])

    gc_oq = list(zip(gc, oq))

    gc_oq = pd.DataFrame(gc_oq, columns =['GarageCars', 'OverallQuality'])

    dough = gc_oq.groupby('GarageCars', as_index=False)['OverallQuality'].sum()

    dough = dough.rename(columns={'OverallQuality':'Count of Houses w/ certain number of Garages'})

    labelsp = dough['GarageCars'].tolist()
    valuesp = dough['Count of Houses w/ certain number of Garages'].tolist()

    zip_listp = list(zip(labelsp, valuesp))

    labelsp = [row[0] for row in zip_listp]
    valuesp = [row[1] for row in zip_listp]

    #assign the CSS class to the classes variable
    tables = [df.to_html(classes='roundedCorners table', header="true")]

    return render_template("dashboard.html", df=df, tables=tables, labels=labels, values=values, max=400, labels_line=labels_line, values_line=values_line, max_line=500000, df_line=df_line, labelsp=labelsp, valuesp=valuesp)

if __name__ == "__main__":
    app.run(debug=True)