# Air-Quality-Prediction
This repository contains code to predict air quality (CO concentration) using a Bayesian Network trained on air quality sensor data.

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
