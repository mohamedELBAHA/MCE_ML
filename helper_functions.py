# This files contains functions that are important to devellop our model and will help us optimise the code
# and the training time.


import pandas as pd
import numpy as np


# this function will 
# coded by :  (@i19elhad)

def import_dataset(name):
    """
    @author : Ismail El Hadrami
    import the data  based on its name ( kidney, banknote ...)

    input : 
        name : the name of the dataset
        type : string

    output : 
        df   : dataset 
        type : dataframe
    """   
       
    if name == "kidney":
        df = pd.read_csv("Data/kidney_disease.csv")
        return df
    elif name == "banknote":
        df = pd.read_csv("Data/data_banknote_authentication.csv")
        return df


def clean_data(data):
    """
    @author : Mohamed EL BAHA
    Cleaning the data provided by the replacing the uncorrect values 

    input :
        data  : dataset to be cleaned 
        type  : DataFrame

    Output :
        data  : Cleand dataset
        type  : DataFrame    
    """

    
    # 1 . First we drop the index column as it's matches with the indexing 
    # the dataframe and it's not relevant in the processing step.
    if 'id' in data.columns : 
        df = data.drop(columns=['id'])


    # 2. Second change unwanted types of data (?)
    types = ['float32', 'float64', 'float', 'int32', 'int64', 'int']

    for column in data.columns : 
        if data[column].dtype not in types :
            data[column] = data[column].str.replace('\t','')
            data[column] = data[column].replace('?',np.nan)
            data[column] = data[column].str.replace(' ','')


    return data

