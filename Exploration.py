#Imports
#region
import os
import time
start_time = time.time()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.matplotlylib as plty

from Loader import load_data, timer
#endregion


# Initial Data Cleaning before Exploration Functions
#region
def drop_nan_columns(data, ratio=1.0)->pd.DataFrame:
    """
    From an initial look at the data it seems like some columns are entirely nan columns (e.g. id, there are 24 such columns)
    The ratio parameter (0.0<=ratio<1.0) lets you drop columns which has 'ratio'% of nans. (i.e if ratio is 0.8 then all columns with 80% or more entries being nan get dropped)
    Returns a new dataframe
    """
    col_list = []
    na_df = data.isna()
    total_size = na_df.shape[0]
    for col in na_df:
        a = na_df[col].value_counts()
        if False not in a.keys():
            col_list.append(col)
        elif True not in a.keys():
            pass
        else:
            if a[True]/total_size >= ratio:
                col_list.append(col)
    print(f"{len(col_list)} columns dropped- {col_list}")
    return data.drop(col_list, axis=1)

def investigate_nan_columns(data)->None:
    """
    Prints an analysis of the nans in the dataframe
    Tells us that employment title and length have very few nans, title has barely any nans
    Upon further looking at the data it seems for employment nans are all unemployed (there is no unemployed category otherwise), length nans is also unemployed

    """
    col_dict = {}
    na_df = data.isna()
    total_size = na_df.shape[0]
    for col in na_df:
        a = na_df[col].value_counts()
        if False not in a.keys():
            col_dict[col] = 1.0
        elif True not in a.keys():
            pass
        else:
            col_dict[col] =  a[True]/total_size
    print(f"{col_dict}")
    return

def handle_nans(data)->None:
    """
    Handle the nans induvidually per column
    emp_title: make Nan -> Unemployed
    emp_length: make Nan - > 10+ years this is both mode filling and value filing
    title: make Nan -> Other
    """
    data['emp_title'] = data['emp_title'].fillna("Unemployed")
    data['title'] = data['title'].fillna('Other')
    mode_cols = ['emp_length', 'annual_inc', 'mort_acc']
    for col in mode_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    return

def investigate(data)->None:
    print(data.shape)
    print(data.info())
    print(data.describe())

def type_list_generator(data, separated=False):
    numericals = ['loan_amnt','funded_amnt','funded_amnt_inv', 'annual_inc','mort_acc','emp_length', 'int_rate']
    if separated:
        numericals.pop()
    strings = ['issue_d', 'zip_code']
    categoricals = [x for x in data.columns if x not in numericals and x not in strings] # ['term', 'grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'purpose', 'title', 'addr_state', 'initial_list_status', 'application_type', 'disbursement_method']
    return numericals, strings, categoricals

def handle_types(data, numericals, strings, categoricals):
    def helper_emp_length(x):
        if x == "10+ years": return 10
        elif x == "2 years": return 2
        elif x == "< 1 year": return 0
        elif x == "3 years": return 3
        elif x == "1 year": return 1
        elif x == "4 years": return 4
        elif x == "5 years": return 5
        elif x == "6 years": return 6
        elif x == "7 years": return 7
        elif x == "8 years": return 8
        elif x == "9 years": return 9
        else:
            return 10
    data['emp_length'] = data['emp_length'].apply(helper_emp_length)

    for category in categoricals:
        try:
            data[category] = data[category].astype('category')
        except:
            pass
    data['issue_d'] = data['issue_d'].astype('datetime64')


#endregion




# Correlation Heatmap
#region
def correlation_heatmap(data):
    corrmat = data.corr()
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
    timing.timer("Heatmap")
#endregion


# Skews with Distribution plot
#region
def skew_with_distplot(data):
    """
    Reveals a positive skew
    """
    from scipy.stats import norm
    sns.distplot(data['int_rate'], fit=norm)
    plt.show()
    timing.timer("Skew with distplot")

#endregion


# Box Plots for outliers
#region
def boxplot(data):
    """
Creates 4 boxplots
            
    """
    fig, axes = plt.subplots(2,2) # create figure and axes
    col_list = ['annual_inc', 'loan_amnt', 'int_rate', 'emp_length']
    by_dict = {0: 'home_ownership', 1:"disbursement_method", 2:"verification_status", 3:"grade"}

    for i,el in enumerate(col_list):
        a = data.boxplot(el, by=by_dict[i], ax=axes.flatten()[i])

    #fig.delaxes(axes[1,1]) # remove empty subplot
    plt.tight_layout() 
    plt.show()
#endregion

#Line Graphs
#region
def lines(data):
    """
    Employment length vs interest rate
    Date Taken vs Loan Amount
    """
    sns.lineplot(x=data['emp_length'], y=data['int_rate'])
    plt.show()
    timing.timer("Lines")
#endregion


# 3D Scatterplot
#region
def three_D_scatter(data):
    """
    Loan Amount vs Employment Length vs Interest Rate
    """
    from mpl_toolkits import mplot3d
    import numpy as np
    info = data[:1000]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xs = info['loan_amnt']
    zs = info['emp_length']
    ys = info['int_rate']
    ax.scatter(xs, ys, zs, s=1, alpha=1)


    ax.set_xlabel('Loan Amount')
    ax.set_ylabel('Interest Rate')
    ax.set_zlabel('Employment Length')
    plt.show()
    timing.timer("3D Scatter")

#endregion


# Scatterplot
#region
def scatter(data):
    """
    Shows that lower grade means higher risk
    """
    info = data.copy()
    a = info.groupby('sub_grade').mean()
    
    sns.scatterplot(x=a.index, y=a['int_rate'])
    plt.show()
    timing.timer("Scatter")


#endregion



# Violin Plot
#region
def violin_plot(data):
    sns.violinplot(x="home_ownership", y="int_rate", data=data, hue="term")
    plt.show()
    timing.timer("Violin")
    return

#endregion


# Bubble Chart
#region
def bubble_chart(data):
    info = data[:1000]
    sns.lmplot(x="loan_amnt", y="int_rate",data=info,  fit_reg=False,scatter_kws={"s": info['annual_inc']*0.005})
    plt.show()
    timing.timer("bubble")
    return
#endregion


# Final Execution
#region
if __name__ == "__main__":
    # Loading in Data
    #region
    timing = timer(start_time)
    data = load_data(500000, purpose="time_of_issue")
    timing.timer()
    #endregion
    # Execution of Data Cleaning
    #region
    data = drop_nan_columns(data, ratio=0.5)

    #investigate_nan_columns(data)
    handle_nans(data)

    #investigate(data)
    numericals, strings, categoricals = type_list_generator(data)
    handle_types(data, numericals, strings, categoricals)
    timing.timer()
    #endregion
    boxplot(data)
    """
    func_list = [correlation_heatmap, skew_with_distplot, boxplot, lines, three_D_scatter, scatter, violin_plot, bubble_chart]
    for i, func in enumerate(func_list):
        func(data)
        timing.timer(f"Function {i+1}")
    """
#endregion
