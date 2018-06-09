
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

