import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# loading csv data to pandas dataframe
heart_data = pd.read_csv('/content/heart_disease_uci.csv')

# print first 5 rows of dataset
heart_data.head()
# print last 5 rows of dataset
heart_data.tail()
# no of rows and columnns in dataset
heart_data.shape
# getting some info about the data
heart_data.info()
# checking for missing values
heart_data.isnull().sum()

for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:
    heart_data[col].fillna(heart_data[col].median(), inplace=True)

heart_data['fbs'].fillna(heart_data['fbs'].mode()[0], inplace=True)
heart_data['exang'].fillna(heart_data['exang'].mode()[0], inplace=True)
heart_data['restecg'].fillna(heart_data['restecg'].mode()[0], inplace=True)

heart_data['slope'].fillna(heart_data['slope'].mode()[0], inplace=True)
heart_data['ca'].fillna(-1, inplace=True)        # use -1 to denote missing
heart_data['thal'].fillna('unknown', inplace=True)  # if encoded as strings

from sklearn.preprocessing import LabelEncoder

label_cols = ['thal', 'slope', 'restecg']  # if they are categorical
for col in label_cols:
    le = LabelEncoder()
    heart_data[col] = le.fit_transform(heart_data[col])

print(heart_data.isnull().sum())  # Should be all zeros

# statistical measures about the data
heart_data.describe()
# checking the distribution of num value[target value]
heart_data['num'].value_counts()
# Encode only string columns
label_encoder = LabelEncoder()
for col in heart_data.select_dtypes(include=['object']).columns:
    heart_data[col] = label_encoder.fit_transform(heart_data[col])

x = heart_data.drop(columns='num', axis=1)
y = heart_data['num']
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, stratify = y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = RandomForestClassifier()

# training the LogisticRegression model with Training data
model.fit(x_train, y_train)

# accuracy on training data
x_train_prediction = model.predict(x_train)
train_accuracy = accuracy_score(y_train, x_train_pred)
print('accuracy on training data :',training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('accuracy on test data :',test_data_accuracy)

X = heart_data.drop('num', axis=1)
print(list(X.columns))
print(X.shape[1])  # should print 15
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2, 1, 0)
import numpy as np

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# if you used scaling during training:
input_data_scaled = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_scaled)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have Heart Disease")
elif prediction[0] == 1:
    print("The Person has 1st stage Heart Disease")
elif prediction[0] == 2:
    print("The Person has 2nd stage Heart Disease")
elif prediction[0] == 3:
    print("The Person has 3rd stage Heart Disease")
elif prediction[0] == 4:
    print("The Person has 4th stage Heart Disease")

