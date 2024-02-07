import sys
import missingno as msno
import seaborn as sns
import pandas as pd
#import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import *
import numpy as np
from sklearn.metrics import roc_curve, auc
import sys



pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.precision',1)

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# funtion for finding missing value graph
def ml_missing_value_graph(dataFrame):
    df = dataFrame
    missing_values = df.isnull().sum()
    if missing_values.sum()==0:
        print ('There are no missing values in any Column')
        return
    missing_values = missing_values[missing_values > 0]
    missing_values.sort_values(inplace=True)
    missing_values = pd.DataFrame(missing_values)
    missing_values.columns = ['count']
    missing_values.index.names = ['Name']
    missing_values['Name'] = missing_values.index
    sns.set(style="whitegrid", color_codes=True)
    a4_dims = (14, 5)
    fig, ax = pyplot.subplots(figsize=a4_dims)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x = 'Name', y = 'count', data=missing_values, ax=ax)

## function for deleting the duplicate rows and rows with missing column values
def ml_del_missing_duplicate_rows(df):
    df = df
    df = df.dropna()
    df = df.drop_duplicates()
    print (df.info())
    return df



## makingt the function for finding target class imbalance
def ml_target_class_imbalance(df,target_col):
    target = target_col
    df = df
    print(df[target].value_counts())
    df_imbalane= pd.DataFrame(df[target].value_counts())
    df_imbalane.columns = ['count']
    df_imbalane.columns
    df_imbalane[target] =df_imbalane.index
    sns.set(style="whitegrid", color_codes=True)
    a4_dims = (3, 2)
    fig, ax = pyplot.subplots(figsize=a4_dims)
    sns.barplot(x = target, y = 'count', data=df_imbalane, ax=ax)


## function for one hot encoding of categorical variables
def ml_one_hot_encoding(df,List):
    for i in range(len(List)):
        tmp = pd.get_dummies(df[List[i]])
        df = pd.concat([df,tmp],axis=1)
        df = df.drop([List[i]], axis=1)
    print(df.columns)
    return df


## function for finding feature importance of the independent variables
def ml_feature_imp(df,target_col):
    rf = RandomForestClassifier(n_estimators=100)
    X_train = df.loc[:, ~df.columns.isin([target_col])]
    y_train = df[target_col]
    rf.fit(X_train, y_train)
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
    print(feature_importances)


## function for scaling of the given numerical attributes
def ml_scaled_attribute(df,List):
    #print(df[List])
    scaler = preprocessing.MinMaxScaler()
    for i in range(len(List)):
        df_tmp = scaler.fit_transform(df[[List[i]]])
        df = df.loc[:, ~df.columns.isin([List[i]])]
        df[List[i]] = df_tmp
    return df


## function for checking accuracy,F1 score, recall
def ml_eval_matrix(y_test,pred):
    print(f'accuracy_score {str(accuracy_score(y_test, pred))}')
    accuracy = str(round(accuracy_score(y_test, pred),3))
    print(f'f1_score       {str(f1_score(y_test, pred))}')
    f1_value = str(round(f1_score(y_test, pred),3))
    print(f'recall_score   {str(recall_score(y_test, pred))}')
    recall = str(round(recall_score(y_test, pred),3))
    print(f'Precision      {str(precision_score(y_test, pred))}')
    precision = str(round(precision_score(y_test, pred),3))
    
    tn, fp, fn, tp =  confusion_matrix(y_test,pred).ravel()
    print()
    
    print(f'True Negative  {str(tn)}')
    TN = tn
    print(f'False Positive {str(fp)}')
    FP = fp
    print(f'False Negative {str(fn)}')
    FN = fn
    print(f'True Positive  {str(tp)}')
    TP = tp
    return accuracy, f1_value, recall, precision, TN, FP, FN, TP


def ml_confusion_matrix(y_test, pred):
    data = confusion_matrix(y_test, pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual' 
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size
    x=sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    print(x)
    return x


## confu#roc_auc = auc(fpr, tpr)sion for Roc curve and aoc value
def ml_roc_curve(y_test, pred):
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()

    #matplotlib.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    fig = plt.figure()
    return fig













