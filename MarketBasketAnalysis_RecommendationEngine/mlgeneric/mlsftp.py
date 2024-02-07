# importing modules
import pysftp
import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import pysftp
import json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.sklearn import *
from sklearn.impute import SimpleImputer

# function for pysftp
def ml_pysftp(RemoteFilePath):
    with open(r"utils\authentication.txt","r") as file:
        data = file.readlines()
        password = data[0]
    print(password)
    sftp = pysftp.Connection(host="sftp.demo.glantus.com", username= "ml", password=password)
    localFilePath = tempfile.mkdtemp()
    print(localFilePath)
    filename = os.path.basename(RemoteFilePath)
    print(filename)
    dest_temp_path = os.path.join(localFilePath,filename)
    print(dest_temp_path)
    sftp.get(RemoteFilePath, dest_temp_path)
    print("Connection succesfully stablished ... ")
    os.chdir(localFilePath)
    sftp.close()    
    
            
# function for mlflow initialisation
def ml_mlfow_init(SettingTracking_Server,ExperimentName):
    mlflow.set_tracking_uri(SettingTracking_Server)
    mlflow.set_experiment(ExperimentName)


# Reading function for different types of files (csv, xlsx, pkl)
def ml_read_data(file):
    if file.lower().endswith('.csv'):
        data = pd.read_csv(file)
    elif file.lower().endswith('.xlsx'):
        data = pd.read_excel(file)
    elif file.lower().endswith('.pkl'):
        data = pd.read_pickle(file)
    else:
        return 'This file type is not handled in the read funtion, give the relevent File extension'
    return data


## Function for deviding the dataset in the train and test split with percentage of data used in test
def ml_train_test_targetCol_split(df, Target_col, split_ratio):
    df = df
    X_df = df.loc[:, ~df.columns.isin([Target_col])]
    y_df = df[[Target_col]]
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=split_ratio, random_state=27)
    return X_train, X_test, y_train, y_test


# Generic function - Create tags based on input and output column names
def ml_create_tags(X, y):
    maptype = lambda x: "string" if x == "object" else "number"
    tags = lambda df: [{"name": x[0], "type": maptype(x[1])} for x in zip(df.dtypes.index.to_list(), df.dtypes.astype("str").to_list())]
    input_tags = tags(X)
    output_tags = tags(pd.DataFrame(y))
    return {
        "glantus.cep.input": json.dumps(input_tags), 
        "glantus.cep.output": json.dumps(output_tags)
    }

def ml_up_pysftp(df,remote_path,file_name):
    
    with open(r'utils\authentication.txt',"r") as file:
        data = file.readlines()
        password = data[0]
    with pysftp.Connection("sftp.demo.glantus.com", username='ml', password=password) as sftp:
        print ("Connection succesfully stablished ... ")
        
        localFilePath_1 = tempfile.mkdtemp()
        df.to_csv(os.path.join(localFilePath_1, file_name),index =False)

        localFilePath = os.path.join(localFilePath_1, file_name)
        remoteFilePath = os.path.join(remote_path, file_name)
        sftp.put(localFilePath, remoteFilePath)
        print ("File uploaded succesfully ")

## setting up the variable
conda_env = {
    'name': 'mlflow-env',
    'channels': ['defaults', 'conda-forge'],
    'dependencies': [
        'python=3.10.1',
        'statsmodels=0.11.0',
        'gunicorn=20.0.4',
        'mlflow=1.6.0'
    ]
}