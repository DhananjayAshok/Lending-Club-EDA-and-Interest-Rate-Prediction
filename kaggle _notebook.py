"""
"""

# Should do earlier itself
def investigate(data)->None:
    print(data.shape)
    print(data.info())
    print(data.describe())


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

data = drop_nan_columns(data, ratio=0.5)

# Now we've taken out the really useless columns, let's check the other ones so that we get a sense of how many NaN entries the rest of our data has

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

investigate_nan_columns(data)

# Tells us that employment title and length have very few nans, title has barely any nans
# Upon further looking at the data it seems for employment nans are likely all unemployed (there is no unemployed category otherwise), length nans we will use mode filling

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

handle_nans(data)

# Now we're done with handling the NaN values we can check that the dataframe truly has no Nan values

any(data.isna().any())

# Now we can look at the data again and actually understand it

investigate(data)

# Looking at the datatypes it's easy to tell there are a lot of categorical columns (e.g. grade) but right now these are only being read as object or strings. We convert them
# We can also safely convert employment length to numbers so that the model can use it as a numerical column
# We also need to convert date to a datetime datatype to best use it

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

handle_types(data, numericals, strings, categoricals)

# And that's it. We have finished an extremely basic cleaning of the dataset. We can now start Exploratory Data Analysis to find deeper patterns in the data.

# Now just copy paste each graph function and run it immediately

"""
Correlation Heatmap
Correlation Heatmaps are a great way to spot linear relations between numerical columns in your dataset.
The basic theory is that you use an inbuilt pandas function corr to calculate each variables correlation to every other variable
Plotting this resulting correlation matrix in a heatmap gives you a sense of which features are correlated to the target variables, hinting at the features you should select or are more important in the model you will make.
"""

"""
From this plot we can clearly see that there is a huge correlation (nearly one to one) with the loan_amnt (Which is the amount requested by the borrower), and the funded amounts (amount funded by investors). This suggests that we probably want to merge these columns as they add dimensionality but do not provide that much extra information
From looking at the variables related to interest rates the first observation is that some variables like mortgage account balance and (surprisingly) annual income seem to have nearly no correlation
In general the most correlated variable seems to be employment length, we could plot these two variables against each other to get a clearer sense of their relationship
Unfortunately overall it seems the numerical variables are not the most correlated, either there is a non-linear relationship in the data or our categorical features are where the bulk of our useful features will be
"""


"""
Distribution Plot
Distribution Plots are very similar to histograms and essentially show how data is distributed across its different values. 
They are extremely useful in finding skews in the data. Most models perform best when the data they deal with is normally distributed (especially linear models). So if we find a skew we may want to apply a skew solution function the variable in order to make it resemble a normal distribution
"""

"""
The extended tail forward gives a clear sign of a positive skew in our interest rate. This means that there are much more lower values than there are high values.
Possible solutions we could apply include the square root and log functions
"""

"""
Boxplots
Boxplots are an extremely useful way to plot outliers, while also seeing how numerical data varies across different categories. 
{Read up and insert what boxplots represent}
"""

"""
The insights we can take from each plot 
(0,0) - This graph tells us nothing about the relation to interest rates, but gives us interesting insights on the economy from which the data was extracted, namely it is likely not a savings based economy. You can tell this by looking at how people who own their houses are not that much wealthier than those who have it on a mortgage. This implies that even when induviudals have enough income to perhaps save an eventually buy a house or a buy a lower grade house they could afford they are opting to take a loan and commit to this expenditure. 
(0,1) - This graph tells us the intutive idea that cash loans on average are of a smaller sum than DirectPay loans, presumably for security reasons. The suprising observation is the lack of any outliers, implying that this lending institution is a relatively safe one which caps the loans it gives, meaning there isn't a single loan so high that it would count as an outlier.
(1,0) - This graph suggests that verification status does seem to have a relationship with interest rate. The average steadily increases the more verfified the borrower is.
(1,1) - This graph is interesting as it seems like the grade of the borrower is low if they either worked in a particular job for very little (less than 1 year or 1 year) or if they have worked the same job for very long (9, 10 years). This suggest some sort of aversion to inexperience, but also stagnation in one job, considering both to be factors that make a loan more risky.
"""

"""
LinePlots
Good for seeing trends in data between induvidual variables
"""

"""
It seems interest rate vs employment length mirrors the relationship between grade and enployment length witnessed in the boxplot above, presumably for the same reasons.
The interest rate for people who have worked less than a year seems low though, possibly this is because these are small buisness or enterprise loans that are valued at a lower interest rate so that the buisness itself has a greater chance of success and thus repaying the loan.
"""

"""
Scatter Plot
The most basic type of plot, but we will scatter averages because otherwise the graph will be too dense for us to actually learn anything
"""
"""
Clearly there is a massive relationship between subgrade and interest rate. This makes it the best feature we have seen so far, and understandably so because the interest rate is in most cases a function of the risk of a loan.
"""


"""
Etc.
The rest of the plots are various different plots which show relationships in the data
3D scatter
Violin Plot
Bubble plot
"""

# Exploratory Data Analysis done. Now we will prepare our data for the model
# First let us define a function to split data and return it to us. This is useful because we want to be very sure of what manipulations we are doing to test data, in order to ensure we aren't cheating
def load_split_data(number_of_rows=None, purpose=None, column='int_rate', test_size=0.2):
    from sklearn.model_selection import train_test_split
    data = load_data(number_of_rows=number_of_rows, purpose=purpose)
    target = data[column]
    data.drop(column, axis=1, inplace=True)
    return train_test_split(data, target, test_size=test_size)

X_train, X_test, y_train, y_test = load_split_data(size, purpose="time_of_issue")

X_train = drop_nan_columns(X_train, ratio=0.5)
X_test = drop_nan_columns(X_test, ratio=0.5)
handle_nans(X_train)
handle_nans(X_test)
handle_types(X_train, numericals, strings, categoricals)
handle_types(X_test, numericals, strings, categoricals)
# For this notebook we will ignore the string variables, however there are ways to use them using other prepreocessing techniques if desired
X_train = X_train.drop(strings, axis=1)
X_test = X_test.drop(strings, axis=1)
timing.timer("Cleaned Data")

# First, let us fix the skew that we saw in the distribution plot using the square root transformation
def manage_skews(train_target, test_target):
    """
    Applying Square Root in order
    """
    return np.sqrt(train_target), np.sqrt(test_target)

y_train, y_test = manage_skews(y_train, y_test)

# Next, we should normalize all of our data, this once again simply makes most models more effective and speeds up convergence
def scale_numerical_data(X_train, X_test, numericals):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[numericals] = sc.fit_transform(X_train[numericals])
    X_test[numericals] = sc.transform(X_test[numericals])
    return
scale_numerical_data(X_train, X_test, numericals)
timing.timer("Scaled Data")

"""
Finally we will encode all of our categorical variables so that models can process them, but before we do that, we need to realize something about the size of our dataset
i.e if you look at the columns such as employment title you can already see a whole bunch of different occupations
"""

data['emp_title'].value_counts()

"""
There are other columns, like purpose, that have this same issue. When there are so many different categories the model is likely to get extremely confused or is just unlikely to generalize well.
Additionally there is also a harm that once we fit an encoder onto our categorical columns, then there will be a completely new profession in the test set that the encoder hasn't seen before, this would throw an exception
To solve this problem we keep only the instances that make up the top 15 categories of that variable, and cast the rest to a standard value like "Other"
"""

def shrink_categoricals(X_train, X_test, categoricals, top=25):
    """
    Mutatues categoricals to only keep the entries which are the top 25 of the daframe otherwise they become other
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

shrink_categoricals(X_train, X_test, categoricals)
timing.timer("Shrunk Categories")

data['emp_title'].value_counts()

# Now we can encode and transform our categorical data safely

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

encode_categorical_data(X_train, X_test, categoricals)
timing.timer("Encoded Data")

"""
We are nearly done with our data processing. The final step is to run a dimensionality reduction algorithm.
Having many dimensions to data can make training models significantly slower and also sometimes less accurate as well. 
We will use PCA, an algorithm which tries to project data into lower dimensions while preserving as much entropy as possible. Instead of stating how many dimensions the PCA algorithm should project down to, we will simply state what percentage of entropy we would like preserved, 95% is a good standard bar.
The reason we are doing the step last is because after the PCA it is virtually impossible to figure out what each column represents in terms of the original data. s
"""

def dimensionality_reduction(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

X_train, X_test = dimensionality_reduction(X_train, X_test)

X_train.shape()
"""
From over 20 columns to only 4! While it may seem like we ought to have lost a lot of data the 95% of entropy being saved ensures we have preserved most core patterns.
You could do the notebook without the above step to see how much slower the training of the models is without this step
"""

# We are finally onto the modelling stage. We will create 4 different models - Random Forest, Support Vector Machine, Linear Regression, KNearestNeighbors and fine-tune them using scikit-learn

Literally copy paste all of the models

# Now we find the best hyperparameters for each model

"""
From these models, while all of them have a decent mse loss the SVM clearly performs best, let us use an averaging ensemble technique to combine the models and see if it improves the loss

Unfortunately the ensemble doesn't perform any better than the SVM in this case.

Finally we will try to use a Boosting technique -Gradient Boosting, 
Gradient Boosting has a simple theory a prelimnary model is trained on a dataset, and its residual errors are recorded. Another model is then trained to model these residual errors, the sum of the induvidual predictions is considered the prediction of the overall system.
Scikit-Learn has an inbuilt GradientBoostingRegressor, but this uses Decision trees as its fundamental unit. We would rather use the SVM that is performing so well induvidually, so we will have to manually define the class

Now we can train and test all of these models to compare them. 

Unfortunately even Gradient Boosting could not reduce the error beyond the SVM.

Let us visualize our models accuracy now to get a sense of how close we are to the true interest rate.

Thank you for viewing this Kernel. The notebook is a work in progress and will be updated as per feedback and future ideas.
"""



Check
    All docstrings
    All plot legends