#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt

path_folder=os.path.normcase(r'C:\Users\dell\Downloads\Titanic-maching-learning-from-disaster')
train_data=pd.read_csv(os.path.join(path_folder,'train.csv'),sep=',')
test_data_origin=pd.read_csv(os.path.join(path_folder,'test.csv'),sep=',')

def deal_ticket_fare(df):
    '''
    计算每张票有几人共享，以及每人花费的船票钱
    
    paramters:
        df--dataframe，待处理的数据表
        
    return:
        df_count--处理后，添加新列的数据表
    '''
    num_of_tickets=df[['Ticket']].groupby(df['Ticket']).count()
    num_of_tickets.columns=['num_of_tickets']
    df_count=df.merge(num_of_tickets,left_on='Ticket',right_index=True,how='left')
    df_count['fare_per_ticket']=df_count['Fare']/df_count['num_of_tickets']
    return df_count

# 计算 预测准确率
def compute_acc(y,y_pred):
    y_pred_class=np.where(y_pred>=0.5,1,0)
    pred_accuracy=(y==y_pred_class).sum()/len(y)    
    return pred_accuracy

train_data_count=deal_ticket_fare(train_data)
cols=['Sex_T','Embarked_T','Age','Fare','fare_per_ticket','num_of_tickets','Pclass','SibSp','Parch']
#  categorical type data
trans_sex={'male':0,'female':1}
trans_embarked={'S':0,'C':1,'Q':2}
train_data_count['Sex_T']=train_data_count['Sex'].map(trans_sex)
#from sklearn.impute import MissingIndicator
#indicator=MissingIndicator(missing_values=np.nan)
#train_missing_indicator=indicator.fit_transform(train_data_count)
from sklearn.impute import SimpleImputer
imp1=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
train_data_count['Embarked']=imp1.fit_transform(train_data_count[['Embarked']])
imp2=SimpleImputer(missing_values=np.nan,strategy='mean')
train_data_count['Age']=imp2.fit_transform(train_data_count[['Age']])
train_data_count['Embarked_T']=train_data_count['Embarked'].map(trans_embarked).astype(np.int)
cols=['Sex_T','Embarked_T','Age','Fare','fare_per_ticket','num_of_tickets','Pclass','SibSp','Parch']
X_train_data=train_data_count.reindex(columns=cols).values
y_train_data=train_data_count['Survived'].ravel()

