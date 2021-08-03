#import libraries
import pandas as pd
import numpy as np
#read the dataset
data=pd.read_csv('data.csv')
#stroing the data into dependnt and independent variable
X=data.iloc[:,:1].values
Y=data.iloc[:,-1].values
#handeling missing data
# from sklearn.impute import SimpleImputer
# imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
# X=X.reshape(1,-1)
# Y=Y.reshape(1,-1)
# X=np.array(imputer.fit_transform(X))
# Y=np.array(imputer.fit_transform(Y))
# #handeling categorical variable
# from sklearn.compose  import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct=ColumnTransformer(transformers=(OneHotEncoder(),))
#diving the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=1)
#Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr=lr.fit(X_train,Y_train)
Y_pred=lr.predict(X_test)
#plotting the training data😂😂😂
import matplotlib.pyplot as plt
# plt.scatter(X_train,Y_train,color='red')
# plt.plot(X_train,lr.predict(X_train),color='blue')
# plt.xlabel("Year of ex")
# plt.ylabel("Salary")
# plt.legend(loc='upper left')
# plt.show()


#plotting the testing data😂😂😂

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,lr.predict(X_test),color='red')
plt.xlabel("Year of ex")
plt.ylabel("Salary")
plt.legend(loc='upper left')
plt.show()
