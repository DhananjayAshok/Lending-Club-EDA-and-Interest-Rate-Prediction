#region
import pandas as pd
import os
import time
#endregion



def load_data(number_of_rows:int =None, purpose=None)->pd.DataFrame:
    """
    Returns a pandas DataFrame with the loan data inside
    number_of_rows: Controls the number of rows read in, default and maximum is 22,60,668 rows
    restriction: Restricts the columns read in to correct for information you should not have depending on the task at hand
        "time_of_issue": Returns only the data that the lender has access to during the issuing of the loan
    """
    root = "loan_data"
    use_cols= None
    if purpose not in [None, 'time_of_issue']:
        raise ValueError(f"Invalid Purpose {purpose}")
    if purpose:
        columnframe = pd.read_csv(os.path.join(root, purpose+".csv"))
        illegals = ['sec_app_fico_range_low ', 'sec_app_inq_last_6mths ', 'sec_app_earliest_cr_line ', 'revol_bal_joint ', 'sec_app_mths_since_last_major_derog ', 'sec_app_revol_util ', 'sec_app_collections_12_mths_ex_med ', 'sec_app_open_acc ', 'fico_range_low', 'sec_app_fico_range_high ', 'verified_status_joint', 'last_fico_range_low', 'sec_app_chargeoff_within_12_mths ', 'fico_range_high', 'total_rev_hi_lim \xa0', 'sec_app_mort_acc ', 'sec_app_num_rev_accts ', 'last_fico_range_high']
        use_cols = [x for x in list(columnframe['name']) if x not in illegals]



    path = os.path.join(root, "loan.csv")

    maximum_rows = 2260668
    if not number_of_rows:
        return pd.read_csv(path, low_memory=False, usecols=use_cols)
    else:
        if number_of_rows > maximum_rows or number_of_rows < 1:
            raise ValueError(f"Number of Rows Must be a Number between 1 and {data.shape[0]}")
        else:
            return pd.read_csv(path, low_memory=False, nrows=number_of_rows, usecols=use_cols)
    

def load_split_data(number_of_rows=None, purpose=None, column='int_rate', test_size=0.2):
    from sklearn.model_selection import train_test_split
    data = load_data(number_of_rows=number_of_rows, purpose=purpose)
    target = data[column]
    data.drop(column, axis=1, inplace=True)
    return train_test_split(data, target, test_size=test_size)


class timer:
    """
    Utility class to time the program while running. 
    """
    def __init__(self, start_time):
        self.start_time = start_time
        self.counter = 0

    def timer(self, message=None):
        """
        Timing function that returns the time taken for this step since the starting time. Message is optional otherwise we use a counter. 
        """
        if message:
            print(f"{message} at {time.time()-self.start_time}")
        else:
            print(f"{self.counter} at {time.time()-self.start_time}")
            self.counter += 1
        return

