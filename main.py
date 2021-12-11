# Here is the main file were models shall be executed

# import libaries
import pandas as pd
import numpy as np
import helper_functions as hf
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


Models = {'SVC':{'model':SVC, #Support vector Classifier
                      'parameters':{'kernel':['linear', 'rbf', 'sigmoid', 'poly'], 
                                                'C'     :[1, 10], 
                                                'degree': [2, 3],
                                                'gamma' : ['scale', 'auto']
                                   }
                },
               'LogisticRegression':{'model':LogisticRegression, # Logistic Regression
                                    'parameters':{'C':[1, 10], 
                                                'fit_intercept' : [True,False],
                                                'intercept_scaling' : [1,10],
                                                 }
                                    },
                'SGDClassifier':{'model':SGDClassifier, # Stochastic gradient descent classifier
                                    'parameters':{'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 
                                                'penalty':['l1', 'l2'], 
                                                'fit_intercept' : [True,False],
                                                 }
                                    },
                'DecisionTreeClassifier':{'model':RandomForestClassifier,
                                    'parameters':{'criterion':['gini', 'entropy'], 
                                                'max_depth' : [2,5,10,None],
                                                 }
                                    },
                'AdaBoostClassifier':{'model':AdaBoostClassifier,
                                    'parameters':{'n_estimators':[50, 100, 150], 
                                                'algorithm':['SAMME', 'SAMME.R'], 
                                                'learning_rate' : [0.1,0.5,1]
                                                 }
                                    },
                'RandomForestClassifier':{'model':RandomForestClassifier,
                                    'parameters':{'n_estimators':[50, 100, 150], 
                                                'criterion':['gini', 'entropy'], 
                                                'max_depth' : [2,5,10,None],
                                                'bootstrap' : [True,False],
                                                 }
                                    }
                }





# 1. Import Data
# Choose the data set by typing it's name, data sets available are the banknote data set and kidney disease data set :
# for the banknote data set please enter ==> 'banknote'
# for the kidney diseas data set please enter ==> 'kidney'
print('\n ---------------- Machine Learning Pipeline---------------- \n ')
print("Please enter the name of the Dataset. \n for the banknote data set please enter ==> banknote \n for the kidney diseas data set please enter ==> kidney ")
name = input("enter here :")
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
print(f'Shape of Data \n | Training Data = {X_train.shape}   | Test Data = {X_test.shape}  | \n | Training Labels = {y_train.shape}   |  Test_labels = {y_test.shape} | \n ')
