
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train_original = train.copy()
test_original = test.copy()

train.head()

train.drop(['Loan_ID'],axis=1,inplace=True)

test.drop(['Loan_ID'],axis=1,inplace=True)

train['Gender']= train['Gender'].map({'Male':1,'Female':0})
test['Gender']= test['Gender'].map({'Male':1,'Female':0})

train['Married']=train['Married'].map({'No':0,'Yes':1})
test['Married']=test['Married'].map({'No':0,'Yes':1})

train['Dependents'] = train['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
test['Dependents'] = test['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})

train['Education'] = train['Education'].map({'Graduate':1,'Not Graduate':0})
test['Education'] = test['Education'].map({'Graduate':1,'Not Graduate':0})

train['Self_Employed']=train['Self_Employed'].map({'No':0,'Yes':1})
test['Self_Employed']=test['Self_Employed'].map({'No':0,'Yes':1})

train['Property_Area'] = train['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})
test['Property_Area'] = test['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})

train['Loan_Status'] = train['Loan_Status'].map({'N':0,'Y':1})


train_copy = train.copy()
test_copy = test.copy()


imputer = Imputer(missing_values='NaN',strategy='most_frequent')

trans_train = imputer.fit_transform(train)
train_data = pd.DataFrame(data = trans_train, index = train.index, columns=train.columns)

trans_test = imputer.fit_transform(test)
test_data = pd.DataFrame(data = trans_test, index=test.index , columns= test.columns)


x_train = train_data.drop("Loan_Status",axis=1)
y_train = train_data['Loan_Status']
x_test = test_data

# for i in range(1,5):
#     random_forest = RandomForestClassifier(n_estimators=i*5)
#     random_forest.fit(x_train,y_train)
#     y_pred = random_forest.predict(x_test)
#     print (random_forest.score(x_train,y_train))

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
logreg.score(x_train,y_train)
y_pred = logreg.predict(x_test)

submission = pd.DataFrame({'Loan_ID': test_original['Loan_ID'],'Loan_Status':y_pred})
submission['Loan_Status']= submission['Loan_Status'].map({0:'N',1:'Y'})
submission.to_csv('Akshay_Submission_impute.csv',index=False)

