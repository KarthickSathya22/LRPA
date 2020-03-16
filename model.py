# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('loans.csv')

# Here the purpose is Categorical data.
# We can't directly use Categorical data so we have encode them.
data = pd.get_dummies(data,columns=['purpose'])

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Splitting datset into Independent and Dependent Features:
X = data.iloc[:,data.columns != 'not.fully.paid']
Y = data.iloc[:,data.columns == 'not.fully.paid']

from sklearn.model_selection import train_test_split

# We split the dataset into 80% as Train data and 20% as Test data:
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Impute the missing data using features means
imp = SimpleImputer(strategy='median')
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

# Standardize the data
std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

from imblearn.combine import SMOTETomek
# Implement RandomUnderSampler
smt = SMOTETomek(random_state=0)
X_res, Y_res = smt.fit_sample(X_train, Y_train.values.ravel())


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty = 'l2',C = 0.1,class_weight='balanced')

#Fitting model with trainig data
clf.fit(X_res, Y_res)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.49154247,  0.22281882, -0.563389  ,  0.57856293,  0.06715763,
       -0.62716612, -0.60839185, -0.20837115,  0.10305402, -0.71559771,
       -0.303668  ,  3.64096179, -0.56277978, -0.38921437,  1.19071298,
       -0.19583687, -0.26787737, -0.21714159, -0.26450387]]))