
import pandas as pd
import urllib
import csv


from sklearn.metrics import classification_report
import pickle
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load the csv file
sasi = pd.read_csv("main_data.csv")

print(sasi.head())

# Select independent and dependent variable
X = sasi[["Duration", "src_bytes", "dst_bytes", "logged_in","Count"]]
Y = sasi["Class"]

from sklearn.model_selection import train_test_split
# Split the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

from sklearn.preprocessing import StandardScaler
# Feature scaling
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test= SS.transform(X_test)


from sklearn.tree import DecisionTreeClassifier

# Instantiate the model
DTC=DecisionTreeClassifier()

# Fit the model
DTC.fit(X_train, Y_train)

# Make pickle file of our model
pickle.dump(DTC, open("model.pkl", "wb"))