import os
from sklearn.linear_model import LinearRegression
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tempfile
import shutil
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category="DtypeWarning")
np.set_printoptions(suppress=True)
from sklearn import metrics
plt.rcParams.update({'figure.max_open_warning': 0})


## function for model evaluation
def ml_regres_model_evaluate(y_true,y_pred):
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    explained_variance = round(explained_variance,4)
    r2  = round(r2,4)
    MAE = round(mean_absolute_error,4)
    MSE = round(mse,4)
    RMSE = round(np.sqrt(mse),4)
    
    return explained_variance, r2,MAE,MSE,RMSE


# function for plotting regression plot
def ml_regression_plot(y_test, pred):
    df1 = pd.DataFrame(data = np.array(y_test),columns = ["Actual values"])
    df2 = pd.DataFrame(data = pred,columns = ["Predicted Value"])
    df3 = pd.concat([df1,df2],axis=1)
    plt.figure(figsize = (12,10))
    sns.set(font_scale=1.4)#for label size
    x =sns.lmplot(x="Actual values",y="Predicted Value",data=df3,fit_reg=True)
    return x


