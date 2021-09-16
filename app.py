import numpy as np  #numpy
import pandas as pd  #pandas
import tensorflow as tf
import glob
import math
from math import pi
import datetime as dt
from keras import backend as K
import keras
import datetime
from tensorflow import keras
import time
from datetime import datetime
import collections
import os
import pandas as pd
import json
from PIL import Image
import io
import sys
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import UJSONResponse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import json
from PIL import Image
from typing import List
from typing import Optional
import uvicorn

def change_Prediction(Predict):
    if Predict<=1:
        return 'Partly Cloudy'
    elif Predict==2:
        return 'Mostly Cloudy'
    elif Predict==3:
        return 'Foggy'
    elif Predict==4:
        return 'Clear'
    elif Predict==5:
        return 'Overcast'
    elif Predict==6:
        return 'Breezy and Overcast'
    elif Predict==7:
        return 'Breezy and Partly Cloudy'
    elif Predict==8:
        return 'Breezy and Mostly Cloudy'
    elif Predict==9:
        return 'Dry and Partly Cloudy'
    elif Predict==10:
        return 'Windy and Partly Cloudy'
    elif Predict==11:
        return 'Light Rain'
    elif Predict==12:
        return 'Breezy'
    elif Predict==13:
        return 'Windy and Overcast'
    elif Predict=='Humid and Mostly Cloudy':
        return 14
    elif Predict==15:
        return 'Drizzle'
    elif Predict==16:
        return 'Windy and Mostly Cloudy'
    elif Predict==17:
        return 'Breezy and Foggy'
    elif Predict==18:
        return 'Dry'
    elif Predict==19:
        return 'Humid and Partly Cloudy'
    elif Predict==20:
        return 'Dry and Mostly Cloudy'
    elif Predict==21:
        return 'Rain'
    elif Predict==22:
        return 'Windy'
    elif Predict==23:
        return 'Humid and Overcast'
    elif Predict==24:
        return 'Windy and Foggy'
    elif Predict== 25:
        return 'Dangerously Windy and Partly Cloudy'
    elif Predict== 26:
        return 'Windy and Dry'
    elif Predict>= 27:
        return 'Breezy and Dry'

def change_category(Summary):
    if Summary=='Partly Cloudy':
        return 1
    elif Summary=='Mostly Cloudy':
        return 2
    elif Summary=='Foggy':
        return 3
    elif Summary=='Clear':
        return 4
    elif Summary=='Overcast':
        return 5
    elif Summary=='Breezy and Overcast':
        return 6
    elif Summary=='Breezy and Partly Cloudy':
        return 7
    elif Summary=='Breezy and Mostly Cloudy':
        return 8
    elif Summary=='Dry and Partly Cloudy':
        return 9
    elif Summary=='Windy and Partly Cloudy':
        return 10
    elif Summary=='Light Rain':
        return 11
    elif Summary=='Breezy':
        return 12
    elif Summary=='Windy and Overcast':
        return 13
    elif Summary=='Humid and Mostly Cloudy':
        return 14
    elif Summary=='Drizzle':
        return 15
    elif Summary=='Windy and Mostly Cloudy':
        return 16
    elif Summary=='Breezy and Foggy':
        return 17
    elif Summary=='Dry':
        return 18
    elif Summary=='Humid and Partly Cloudy':
        return 19
    elif Summary=='Dry and Mostly Cloudy':
        return 20
    elif Summary=='Rain':
        return 21
    elif Summary=='Windy':
        return 22
    elif Summary=='Humid and Overcast':
        return 23
    elif Summary=='Windy and Foggy':
        return 24
    elif Summary=='Dangerously Windy and Partly Cloudy':
        return 25
    elif Summary=='Windy and Dry':
        return 26
    elif Summary=='Breezy and Dry':
        return 27

def change_Pcategory(PrecipType):
    if PrecipType=='rain':
        return 1
    elif PrecipType=='snow':
        return 2
    elif PrecipType=='NaN':
        return 3

app = FastAPI()

#
# Split data into n_timestamp
#
def data_split(sequence,target, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
          # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix][-1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_timestamp = 100
train_days = 95000  # number of days to train from
testing_days = 1000 # number of days to be predicted
n_epochs = 1
filter_on = 1
def FormatingCSVdata(data_neur2):
  
    data_neur2['T'] = medfilt(data_neur2['T'], 3)
    data_neur2['T'] = gaussian_filter1d(data_neur2['T'], 1.2)
    data_neur2['TA'] = medfilt(data_neur2['TA'], 3)
    data_neur2['TA'] = gaussian_filter1d(data_neur2['TA'], 1.2)
    data_neur2['H'] = medfilt(data_neur2['H'], 3)
    data_neur2['H'] = gaussian_filter1d(data_neur2['H'], 1.2)
    data_neur2['WS'] = medfilt(data_neur2['WS'], 3)
    data_neur2['WS'] = gaussian_filter1d(data_neur2['WS'], 1.2)
    data_neur2['WB'] = medfilt(data_neur2['WB'], 3)
    data_neur2['WB'] = gaussian_filter1d(data_neur2['WB'], 1.2)
    data_neur2['V'] = medfilt(data_neur2['V'], 3)
    data_neur2['V'] = gaussian_filter1d(data_neur2['V'], 1.2)
    data_neur2['P'] = medfilt(data_neur2['P'], 3)
    data_neur2['P'] = gaussian_filter1d(data_neur2['P'], 1.2)


    train_set = data_neur2[0:train_days].reset_index(drop=True)
    test_set = data_neur2[train_days: train_days+testing_days].reset_index(drop=True)
    #print(test_set['Date'].max())
    #print(test_set['Date'].min())
    D=pd.to_datetime(test_set['Date'].max())-pd.to_datetime(test_set['Date'].min())
    c=int(D.days)+1
    
    training_set = train_set.iloc[:, [1,2,3,4,5,6,7,12]].values
    testing_set = test_set.iloc[:, [1,2,3,4,5,6,7,12]].values

    label_train=training_set[:,-1]
    label_test=testing_set[:,-1]

    #
    # Normalize data first
    #
    sc_x = MinMaxScaler(feature_range = (0, 1))
    sc_y=MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc_x.fit_transform(training_set)
    y_train_scaled=sc_y.fit_transform(label_train.reshape(-1,1))
    testing_set_scaled = sc_x.fit_transform(testing_set)
    y_test_scaled=sc_y.fit_transform(label_test.reshape(-1,1))


    X_train, y_train = data_split(training_set_scaled,y_train_scaled,n_timestamp)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 8)
    X_test, y_test = data_split(testing_set_scaled,y_test_scaled, n_timestamp)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 8)
    
    y_predicted = model.predict(X_test)

    y_predicted_descaled = sc_y.inverse_transform(y_predicted)
    y_train_descaled = sc_y.inverse_transform(y_train.reshape(-1,1))
    y_test_descaled = sc_y.inverse_transform(y_test.reshape(-1,1))
    y_pred = y_predicted.ravel()

    p_array= np.around(y_predicted_descaled).astype(int)
    test_set['New_Date']=pd.to_datetime(test_set['Date'])+pd.DateOffset(days=c)
    test_set1=pd.DataFrame(test_set['New_Date'].loc[::-1].reset_index(drop = True))
    test_set1.sort_values(by='New_Date',inplace=True, ascending=True)

    return p_array , test_set1.head(900)

class dataset(BaseModel):
  filename: str
  contenttype: str
  result: Optional[str]= None

class Prediction(BaseModel):
  result: str

#from json_response import JsonResponse
@app.on_event("startup")
def load_model():
    global model
    model =  keras.models.load_model("Model")
 
 
@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}

@app.post("/weatherprediction",response_model=Prediction)
async def create_files(dateid: str):
    dataframe = pd.read_csv('Dataset/ProcessedWeatherHistory.csv')
    #dataframe.drop(dataframe.columns[0],axis=1,inplace=True)
    p_array, test_set1=FormatingCSVdata(dataframe)
    #test_set1=test_set['New_Date'].loc[::-1].reset_index(drop = True)
    p_df = pd.DataFrame({'Summary': p_array[:, 0]})
    p_df['Summary']=p_df['Summary'].apply(change_Prediction)
    #vals = p_df['Summary'].tolist()
    result = p_df.to_json(orient="records")
    test_set1['New_Date']=test_set1['New_Date'].astype(str)
    result1 = pd.concat([test_set1, p_df], axis=1)
    result1.sort_values(by='New_Date',inplace=True, ascending=True)
    result1.dropna(subset=['New_Date'],inplace=True)
    result1['Summary'].fillna(method='ffill', inplace=True)  
    result1[['Date','Time']] = result1.New_Date.str.split(expand=True)
    result1['Date'].astype('str')
    result2=result1.loc[result1['Date']==dateid, 'Summary']
    #date1 = test_set1.to_json(orient="records")
    result2 = result2.to_json(orient="records")
    
    print(result1.head())
    
    return {
      'result' : result2,
    }
 
 
@app.post("/datasetfiles/",response_model=dataset)
async def create_files(files: UploadFile = File(...)):
    try:
        dataframe = pd.read_csv(files.file)
        #dataframe.drop(dataframe.columns[0],axis=1,inplace=True)
        dataframe.to_csv('Dataset/ProcessedWeatherHistory.csv',index=False)
        result="Successful"

    except FileNotFoundError:
        result="File Not Found"

    except Exception:
        result="Error"
    return {
      'filename': files.filename,
      'contenttype': files.content_type,
      'result' : result
    }

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__": 
    uvicorn.run(app)

