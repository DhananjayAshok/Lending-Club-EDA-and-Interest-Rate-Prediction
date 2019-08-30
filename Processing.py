# Imports
#region
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

from Exploration import drop_nan_columns, handle_nans, handle_types, type_list_generator
from Loader import load_split_data, timer
import time
timing = timer(time.time())
#endregion

# Data Wrangling
#region
def manage_skews(train_target, test_target):
    """
    Applying Square Root in order
    """
    return np.sqrt(train_target), np.sqrt(test_target)

def manage_outliers(X_train, y_train):
    pass

def scale_numerical_data(X_train, X_test, numericals):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[numericals] = sc.fit_transform(X_train[numericals])
    X_test[numericals] = sc.transform(X_test[numericals])
    return

def shrink_categoricals(X_train, X_test, categoricals, top=25):
    """
    Mutatues emp_title to only keep the jobs which are the top 15 of the daframe otherwise they become other
    """
    for category in categoricals:
        if category not in X_train.columns:
            continue
        tops = X_train[category].value_counts().index[:top]
        def helper(x):
            if x in tops:
                return x
            else:
                return "Other"
        X_train[category] = X_train[category].apply(helper)
        X_test[category] = X_test[category].apply(helper)

def encode_categorical_data(X_train, X_test, categoricals):
    from sklearn.preprocessing import LabelEncoder
    for category in categoricals:
        if category not in X_train.columns:
            continue
        le = LabelEncoder()
        X_train[category] = le.fit_transform(X_train[category])
        X_test[category] = le.transform(X_test[category])
        #X_test[category] = le.transform(X_test[category])
    return
    
#endregion


# PCA and dimensionality reduction
def dimensionality_reduction(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

def final(size=500000, reduce_dims=False):
    """
    Processing Function, optional dimensionality reduction
    """
    # Load in data
    #region
    timing.timer("Starting Data Processing")
    X_train, X_test, y_train, y_test = load_split_data(size, purpose="time_of_issue")
    numericals, strings, categoricals = type_list_generator(X_train, separated=True)
    timing.timer("Loaded Data")
    #endregion

    # Basic Data Cleaning
    #region
    X_train = drop_nan_columns(X_train, ratio=0.5)
    X_test = drop_nan_columns(X_test, ratio=0.5)
    handle_nans(X_train)
    handle_nans(X_test)
    handle_types(X_train, numericals, strings, categoricals)
    handle_types(X_test, numericals, strings, categoricals)
    X_train = X_train.drop(strings, axis=1)
    X_test = X_test.drop(strings, axis=1)
    timing.timer("Cleaned Data")
    #endregion

    # Data Wrangling
    #region
    y_train, y_test = manage_skews(y_train, y_test)
    scale_numerical_data(X_train, X_test, numericals)
    timing.timer("Scaled Data")
    shrink_categoricals(X_train, X_test, categoricals)
    timing.timer("Shrunk Categories")
    #handle_nans(X_test)
    encode_categorical_data(X_train, X_test, categoricals)
    timing.timer("Encoded Data")
    #endregion

    # Dimensionality Reduction
    if reduce_dims:
        X_train, X_test = dimensionality_reduction(X_train, X_test)
        timing.timer("Reduced Data")
    timing.timer("Data Processing Complete")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = final(reduce_dims=True)