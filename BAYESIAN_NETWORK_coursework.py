import pandas as pd # importing the pandas library
import numpy as np # importing the numpy library
from sklearn.preprocessing import LabelEncoder # importing the labelencoder from sklearn.preprocessind
from sklearn.model_selection import train_test_split # importing the train_test_split from sklearn.model_selection
from sklearn.linear_model import LinearRegression # importing the linear regression from sklearn.Linear_model
from sklearn.metrics import accuracy_score # importing the accuracy_score from sklean.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # importing the mean caluculating library from sklearn.metrics
import bnlearn as bn # importing bnlearn library

data=pd.read_excel("AirQualityUCI.xlsx") # reding the excell file
data.fillna(0,inplace=True) # filling the na values with 0
print(data) # printing the data
data.info() # printing the info of the data

# Preparing the data
data['Date'] = pd.to_datetime(data['Date']) 
data = data.set_index('Date')

# Define the target variable and the features
target = 'CO(GT)' # targetting the required column
features = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)'] # selecting the required columns

# Change the target variable to categorical
data[target] = LabelEncoder().fit_transform(data[target])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred) # evaluating the mean squared error
mae = mean_absolute_error(y_test, y_pred) # evaluating the mean absolute error
r2 = r2_score(y_test, y_pred) # evaluating the r square score

print("Mean Squared Error:", mse) # printing the mean squared error
print("Mean Absolute Error:", mae) # printing the absolute error
print("R-squared:", r2) # printing the R-squared value

threshold = 0.5  # determining the threshold
predictions_binary = (y_pred > threshold).astype(int) # predicting the y_pred on the threshold

# Evaluate the accuracy of the model
print("Accuracy: ", accuracy_score(y_test, predictions_binary))
print(y_pred)

# Prepare the new data
future_data = pd.DataFrame({'PT08.S1(CO)': [1360], 'NMHC(GT)': [150], 'C6H6(GT)': [11.9], 'PT08.S2(NMHC)': [1046], 'NOx(GT)': [166], 'PT08.S3(NOx)': [1056], 'NO2(GT)': [113], 'PT08.S4(NO2)': [1692], 'PT08.S5(O3)': [1268]})

# Make predictions on the new data
future_predictions = model.predict(future_data)
threshold = 1 # defining th threshold
predictions_binary = (y_pred > threshold).astype(int) # writing the formulae for predictions binary
print("Predicted CO value for the future:", predictions_binary[0]) # printing the predicted value for CO

# Define the structure of the model
Ip = data.drop(['CO(GT)'],axis=1)# dependent variables
op = data['CO(GT)']# independent variables
Xtrain, Xval, Ztrain, Zval = train_test_split(Ip,op, test_size=0.2, random_state=0) # defining the parameters
df_tst = pd.concat([Xval, Zval], axis='columns') # concating of xval and zval
df_train = pd.concat([Xtrain, Ztrain], axis='columns') # concating of xtrain and ztrain
df_train1=df_train # trained 1 df
df_train2=df_train # trained 2 df
DAG = bn.structure_learning.fit(df_train1, methodtype='nb', root_node='CO(GT)', bw_list_method='nodes', verbose=3) #hc

# Plot
G = bn.plot(DAG)

# Parameter_learning
model = bn.parameter_learning.fit(DAG, df_train1, verbose=3);