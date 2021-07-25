from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


app = Flask(__name__)
model = pickle.load(open("forest_model2", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Area
        Area = int(request.form["Area"])

        #bhk

        bhk = int(request.form["BHK"])
        
        #total no of bathrooms

        bathroom = int(request.form['Bathroom'])

        #total number of parkings

        parking = int(request.form['Parking'])

        per_sqft = int(request.form['Per_Sqft'])
       
        furnishing=request.form['Furnishing']
  
        status = request.form["Status"]
   
        typee = request.form["Type"]
   
        transaction = request.form["Transaction"]

        area_per_bhk = Area/bhk
        bathroom_per_bhk = bathroom/bhk
        parking_per_bhk = parking/bhk
        
        num_pipeline = Pipeline([
                         ('imputer',SimpleImputer(strategy="median")),
        ])
        num_attrib = ['Area','bhk','bathroom','parking','per_sqft','area_per_bhk','bathroom_per_bhk','parking_per_bhk']
        cat_attribs = ["furnishing","status","transaction","typee"]
        full_pipeline = ColumnTransformer([
                                   ("num",num_pipeline,num_attrib),
                                   ("cat",OneHotEncoder(),cat_attribs),
        ])
        data = {'Area':[Area],'bhk':[bhk],'bathroom':[bathroom],'furnishing':[furnishing],'parking':[parking],'status':[status],'transaction':[transaction],'typee':[typee],'per_sqft':[per_sqft],'area_per_bhk':[area_per_bhk],'bathroom_per_bhk':[bathroom_per_bhk],'parking_per_bhk':[parking_per_bhk]}
        dd = pd.DataFrame(data)
        data_prepared = full_pipeline.fit_transform(dd)

        prediction = model.predict(data_prepared)

        output=round(prediction[0],2)

        return render_template('home.html',prediction_text="Your house price is Rs. {}".format(output))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
