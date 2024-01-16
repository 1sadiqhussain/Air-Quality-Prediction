# Air-Quality-Prediction
 Here is a description of the project:

This machine learning project focuses on predicting air pollution (CO concentration) using various sensor data. The dataset contains air quality data collected from sensors along with the ground truth CO concentration values. 

The project follows several steps:

1. Load and prepare the dataset - The raw data is loaded into a Pandas dataframe. Missing values are filled and datetime columns are parsed.

2. Split data into training and test sets - The data is split 80/20 into separate sets for model training and testing.

3. Train a linear regression model - A linear regression model is trained on the training set to predict CO concentration from the other sensor inputs.

4. Evaluate model performance - The model's accuracy is evaluated on the test set using MAE, MSE, and R-squared metrics. It achieves decent performance.

5. Make predictions on new data - The trained model is used to make CO predictions on new hypothetical sensor data. 

6. Train a Bayesian Network model - A Bayesian Network is trained on the data to model probabilistic relationships between variables.

Overall, this project demonstrates how different machine learning techniques like linear regression and Bayesian Networks can be applied to air quality data. The models provide valuable CO concentration forecasts that could inform pollution control measures. 

Some limitations are that only a small sample dataset was used for training. More varied data would improve model robustness. Additionally, more advanced deep learning models could potentially achieve better accuracy. Overall though, this project provides a solid foundation for using ML with air quality data.


# Overview
# The code does the following:

Loads air quality data from an Excel file into a Pandas DataFrame
Preprocesses the data (filling NaN, encoding target variable, splitting into train/test)
Trains a linear regression model to predict CO concentration
Evaluates model performance using MSE, MAE, R^2
Makes predictions on new data
Learns a Bayesian Network structure and parameters from the data
Plots the learned Bayesian Network
The core libraries used are Pandas, Scikit-Learn, and bnlearn.

# Usage
The main dependencies are:
Pandas

Scikit-Learn

bnlearn

python air_quality_prediction.py

This will load and preprocess the AirQualityUCI.xlsx data, train and evaluate the linear regression model, make predictions, learn and plot the Bayesian Network.

# Repository Structure
# The repository contains:
air_quality_prediction.py - Main code

AirQualityUCI.xlsx - Input data

sample_bn.png - Sample Bayesian Network plot

README.md - This file
