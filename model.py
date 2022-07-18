# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('D:\Kaggle Competition\house_train.csv')

X=df.drop('SalePrice',axis=1)
y=df['SalePrice']

# Hyperparameter tuning
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
cv=KFold(n_splits=5,random_state=None,shuffle=False)

# Hyperparameter tuning in order get the best parameter

import numpy as np

# XgBoost algorithm

import xgboost
xgb_reg = xgboost.XGBRegressor(model__learning_rate=0.1, model__max_depth=3,
             model__n_estimators=800)
X=np.array(X)
y=np.array(y)
xgb_reg.fit(X,y)
pickle.dump(xgb_reg, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))
#print(model.predict([[2, 9, 6]]))