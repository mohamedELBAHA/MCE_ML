# This files contains functions that are important to devellop our model and will help us optimise the code
# and the training time.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def import_dataset(name):
    """
    @author : Ismail El Hadrami
    import the data  based on its name ( kidney, banknote ...)

    Parameters :
    ----------- 
        name : the name of the dataset
        type : string

    Returns : 
    ---------
        df   : dataset 
        type : dataframe
    """   
       
    if name == "kidney":
        df = pd.read_csv("Data/kidney_disease.csv")
        return df
    elif name == "banknote":
        df = pd.read_csv("Data/data_banknote_authentication.csv")
        return df
    else:
        print("Invalid dataset name. Choose 'kidney' or 'banknote'")
        return


def clean_data(data):
    """
    @author : Sami RMILI
    Cleaning the data provided by the replacing the unknown values

    Parameters :
        data  : dataset to be cleaned 
        type  : DataFrame

    returns :
    ---------
        data  : Cleand dataset
        type  : DataFrame    
    """

    # Drop the index column since matches the indexing
    # the dataframe, and it's not relevant in the processing step.
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Some columns names might have unnoticeable spaces, remove 'em
    data.columns = data.columns.str.strip()

    # Cleanup spaces and "?" values in "object" data points
    is_str_cols = data.dtypes == object
    str_columns = data.columns[is_str_cols]
    data[str_columns] = data[str_columns].apply(lambda s: s.str.strip())
    data = data.replace("?", np.nan)

    # Convert columns to numerical values if possible
    data = data.apply(pd.to_numeric, errors='ignore')

    # Pick the "true" string columns again
    is_str_cols = data.dtypes == object
    str_columns = data.columns[is_str_cols]

    # Replace NAN data points (object columns) with most used value
    data[str_columns] = data[str_columns].fillna(data.mode().iloc[0])

    # Replace numerical data points with mean of column
    data = data.fillna(data.mean())
    return data


def normalize_data(data):
    """
        @author : Sami RMILI
        One-hot encode the data and normalize it

        Parameters
        ----------

            data : DataFrame
            The pandas dataframe to preprocess

        Returns
        -------

            DataFrame
            The normalized dataset

    """

    # One-hot encoding
    encoded_data = pd.get_dummies(data, drop_first=True)

    # Normalization
    normalized_data = (encoded_data - encoded_data.mean()) / (encoded_data.std())

    return normalized_data


def preprocess_data(data, classif="class"):
    """
    @author : Sami RMILI
    Preprocessing the data (cleaning and normalization)

    Parameters
    ----------

        data : DataFrame
        The pandas dataframe to preprocess

        classif : string, optional
        Specify the name of the ground-truth column (default is "class")

        Returns
        -------

        DataFrame
        The preprocessed data (without the ground-truth)

        DataFrame
        The ground-truth dataframe
    
    """

    # Clean data
    data = clean_data(data)

    # Remove ground-truth
    target = data[classif]
    data = data.drop(columns=classif)

    # Normalized data
    data = normalize_data(data)

    return data, target



def split_data(X,y,test_size):
    """
    @author : Mohamed EL BAHA
    Split the data into a training set and a validation set
    
    Parameters :
    ------------
        X    : features of the data
        type : array 

        test_size : the test size 
        type      : float
    
    Returns :
    ---------
        training set for features and labels, validation set features and labels. 
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)

    return X_train, X_test, y_train, y_test




df = import_dataset("kidney")
df, target = preprocess_data(df, "classification")
print(df.describe())

