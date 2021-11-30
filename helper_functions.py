# This files contains functions that are important to devellop our model and will help us optimise the code
# and the training time.


import pandas as pd


# this function will import the dataset based on its name
# coded by : Ismail El Hadrami (@i19elhad)

def import_dataset(name):   
        if name == "kidney":
            df = pd.read_csv("/Data/kidney_disease.csv")
            return df
        elif name == "banknote":
            df = pd.read_csv("/Data/data_banknote_authentication.csv")
            return df

