# Here is the main file were models shall be executed

# import libaries
import pandas as pd
import numpy as np
import helper_functions as hf


# 1. Import Data
# Choose the data set by typing it's name, data sets available are the banknote data set and kidney disease data set :
# for the banknote data set please enter ==> 'banknote'
# for the kidney diseas data set please enter ==> 'kidney'
print("Please enter the name of the Dataset. \n for the banknote data set please enter ==> banknote \n for the kidney diseas data set please enter ==> kidney ")
name = input()
df = hf.import_dataset(name)
print(" ✓ The Banknote data set succefuly imported !")
# 2. Preprocess Data ( clean and normalize data)
df = hf.normalize_data(df)
print(" ✓ Data succefuly pre-prossesed ")

# 3. Feature Selection  
df = hf.feature_selection(df,'PCA',variance_threshold=0.95)

# 4. Split Data ( 70% train, 30% test )
X = df.drop(columns=['class '])
y = df['class ']
X_train, X_test, y_train, y_test = hf.split_data(X,y,test_size=0.3)

# Explore data shapes 
X_train.shape
X_test.shape 
print(f'Shape of Data \n Training Data = {X_train.shape}   | Test Data = {X_test.shape} \n Training Labels = {y_train.shape}   |  Test_labels = {y_test.shape} \n ')

# 