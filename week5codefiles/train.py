#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import pickle
from IPython import get_ipython

print("importing packages")


#get_ipython().system('pip install tqdm')





url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv'
urllib.request.urlretrieve(url,'AER_credit_card_data.csv')


df = pd.read_csv('AER_credit_card_data.csv')



df.head()




df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] =df[c].str.lower().str.replace(' ','_')



categorical_columns



df.dtypes


df.isnull().sum()



df = df.copy()



target = (df.card =="yes").astype(int)
target
target.name ='target'



df =df.join(target)

print("splitting dataset")

df_full_train,df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=1,
)
df_train,df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1,
)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)



y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values




del df_train['target']
del df_val['target']
del df_test['target']




df_train



#question1
df.dtypes


print("computing auc")
#Question1
list = ['reports','dependents','active','share']
features = df[list]
features
scores = []
for i in features.columns:
    fpr, tpr, thresholds = roc_curve((df_train['card'].values =="yes").astype(int),df_train[i].values, pos_label = 1)
    aucval = auc(fpr, tpr)
    if aucval < 0.5:
        fpr, tpr, thresholds = roc_curve((df_train['card'].values =="yes").astype(int),-1*(df_train[i].values), pos_label = 1)
        aucval = auc(fpr, tpr)
        
    scores.append(f"score of {i} = {aucval}")
print(scores)
#AUC goes below 0.5 if the variable is negatively correlated with the target variable.


df.dtypes




categorical = ["owner", "selfemp"]
numerical = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active"]


print("training the model")

#Question3
def train(df_train,y_train,C=1.0):
    dicts = df_train[categorical+numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(solver='liblinear',C=C,max_iter = 1000)
    #C is inverse of regularization parameter, the smaller the C the better the regularization
    model.fit(X_train,y_train)
    return dv,model


dv,model = train(df_train,y_train,C=1.0)



def predict(df_val,dv,model):
    dicts = df_val[categorical+numerical].to_dict(orient='records')
    X =dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred



y_pred = predict(df_val,dv,model)



y_pred.shape,y_val.shape,df_val.shape



fpr,tpr,thresholds = roc_curve(y_val,y_pred)



#Question 2
auc(fpr,tpr).round(3)



#question 3


print("Validating the model")
t = np.arange(0,1.01,0.01)
recall = []
precision = []
F1 = []
for i in t:
    predict_positive = (y_pred >= i)
    predict_negative = (y_pred < i)
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    precision_val = tp/(tp+fp)
    recall_val = tp/(tp+fn)
    precision.append(precision_val)
    recall.append(recall_val)
    F1.append(2*(precision_val*recall_val)/(precision_val+recall_val))
    



#question 3
plt.figure(figsize=(5,5))
plt.plot(t,precision,label='precision')
plt.plot(t,recall,label='recall',linestyle='--')
plt.legend()



i = np.arange(0,1,0.01)
for t in range (0,100):
    print(f"for treshold {i[t]} = {F1[t]}")


#Question 4



i = [0.1,0.4,0.6,0.7]
for t in range (0,4):
    print(f"for treshold {i[t]} = {F1[t]}")




#Question5


kfold = KFold(n_splits=5,shuffle=True,random_state=1)



train_idx,val_idx = next(kfold.split(df_full_train))
train_idx.shape,val_idx.shape




len(df_full_train)



df_train = df_full_train.iloc[train_idx]
y_train = df_full_train.iloc[train_idx].target.values
df_val = df_full_train.iloc[val_idx]
y_val = df_val.target.values




len(train_idx),len(val_idx),len(df_val),len(val_idx),len(y_val)







scores=[]
for train_idx,val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.card.values
    y_values = df_val.card.values

    dv,model = train(df_train,y_train)
    y_pred = predict(df_val,dv,model)
    auc = roc_auc_score(y_val,y_pred)
    scores.append(auc)

stand_dev = np.std(scores).round(3)
print(f"standard deviation is : {stand_dev}")



#Question 6 
#Checking progress of each iteration using tqdm
n_splits = 5
for C in tqdm([0.01, 0.1, 1, 10]):
    kfold = KFold(n_splits=5,shuffle=True,random_state=1)
    
    scores = []
    for train_idx,val_idx in \
        tqdm(kfold.split(df_full_train),total=n_splits):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
    
        y_train = df_train.card.values
        y_val = df_val.card.values

        dv,model = train(df_train,y_train,C=C)
        y_pred = predict(df_val,dv,model)
        auc = roc_auc_score(y_val,y_pred)
        scores.append(auc)
    print('C=%s %.3f += %.3f' % (C,np.mean(scores), np.std(scores)))
    
np.mean(scores), np.std(scores)

#cross validation --> take higher k values for smaller models , i.e split into ,ore chunks
# cross validation helps understand std that gives the stability of model across different data chunks


#Training the model now in full train dataset with the best C value.
dv,model = train(df_full_train,df_full_train.card.values,C=1.0)
y_pred = predict(df_val,dv,model)
auc = roc_auc_score(y_val,y_pred)
auc
# C=1 0.996 += 0.003 C = 1 is the best parameter




dv,model = train(df_full_train,df_full_train.target.values,C=1)
y_pred = predict(df_test,dv,model)
auc = roc_auc_score(y_test,y_pred)
auc


print("saving the model")

#SAVING THE MODEL



output_file = 'model_C=10.bin'
output_file



#get_ipython().system('pip install scikit-learn')





f_out = open(output_file,'wb')
pickle.dump((dv,model),f_out)
f_out.close()
#or
with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)
#with this file closes automatically










