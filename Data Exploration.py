#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



listings = pd.read_csv('data/listings.csv')
calendar = pd.read_csv('data/calendar.csv')
reviews = pd.read_csv('data/reviews.csv')

data = [listings, calendar, reviews]


# In[ ]:



def obs_data(dfs): # function for quick observation of our data
    for df in dfs:
#         display(df.head())
        print('\n-----------------------Loading Next DF------------------------------\n')
        print('The shape of the dataframe is\n', df.shape, '\n')
        null_df = df[df.isna()]  # creating a df of columns records with null values
        null_cols = df.columns[df.isna().any()]  # creating a df of the column names to that are missing data 
        print('Names of Missing Columns')
        print(null_cols.tolist(), '\n')

        num_cols_mising = null_cols.nunique()
        print('Number of Missing Columns')
        print(num_cols_mising)

        # percent of missing columns
        tot_cols = df.columns.nunique()
        print('\nPercent of columns missing records (%)')
        print(np.round((1-((tot_cols-num_cols_mising)/tot_cols))*100))

        # Missing Rows ---------------------------
        num_missing_values = df.isna().sum()
#         print(num_missing_values.values)
        percent_missing_values = np.round(((1-(df.shape[0]-num_missing_values)/df.shape[0])*100),2)
#         # print(df.shape[0])
#         print(percent_missing_values)
        print('\nColumn','|' 'Percent of Missing Values (%)')
        print('----------------------------------- \n')
        for col, value in percent_missing_values.items():
            if value != 0:
        #         temp_dict = {'Columns': col, 'Percent Missing Records':values}
                print((col),'|',(value))
            else:
                pass
        print('n', df.describe())
    print('\n...')
    print('...')
    print('...\n')
    print('DONE!')



obs_data(data)
            


# We can see some interesting insights from the intial analysis of this data. For example ...

# In[ ]:


# Now we can write an equation to clean the data for us

def clean_data(dfs):
    for df in dfs:
        df.dropna(thresh=(.3*df.shape[0]), axis=1, inplace=True) # dropping entire columns where the % of null values is greater than 30
        df.dropna(axis=0, inplace=True)  # drops the rows of all the other columns
        
        empty_check = len((df.columns[df.isna().any()]).tolist())  # checking for empty columns converting to list and then counting the length
        if empty_check == 0:
            print (df.isna().any())
            print('No null values')
        else:
            print(df.isna().sum())
            print('We missed something')
        
        
clean_data(data)


# We have no missing values!

# ### Feature Engineering 
# From inital analysis we can devide our listing data it 3 segments that might factor into the rate of rentals of the home
#     1. Host info.
#     2. Neighborhood info.
#     3. Actual retal info.
# * First removing unsessary data columns
# * 

# In[ ]:


listings.head(2)


# In[ ]:


# selecting columns we are interested in 
host_raiting = listings[['host_name', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
       'host_is_superhost', 'host_listings_count',
       'host_total_listings_count','host_has_profile_pic', 'host_identity_verified', 'number_of_reviews', 'number_of_reviews_ltm',
       'number_of_reviews_l30d', 'first_review', 'last_review',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'instant_bookable' ]]

# renaming columns using string matching
host_raiting.columns = [col.replace('host_','').replace('_scores_', '_') for col in host_raiting_cols] 


# In[ ]:


# working with categorical features
host_raiting_obj = host_raiting.select_dtypes(include ='object')
print(host_raiting_obj.columns.values)


# Here when we look at our fields that our ojects we see something unexpected - "since" - which is a date, "response_rate", and "acceptance_rate" which are percentages we'll covert to string numbers. There are other categorical variables like "

# In[ ]:


# featuring engineering date variables
import datetime as dt
now = dt.datetime.now().date()  # coverts date times to just date with no time
host_raiting['since'] = pd.to_datetime(host_raiting['since'])
host_raiting['first_review'] = pd.to_datetime(host_raiting['first_review'])
host_raiting['last_review'] = pd.to_datetime(host_raiting['last_review'])

host_raiting['year'] = host_raiting['since'].dt.year

host_raiting['total_days_rental'] = (now - host_raiting['since'].dt.date) # time difference must confirm datetime to date to match
host_raiting['days_since_first_rev'] = (now - host_raiting['first_review'].dt.date)
host_raiting['days_since_last_rev'] = (now - host_raiting['last_review'].dt.date)

# feature engineering rates 
def p2f(x):  # function that strips the % sign and converts to decimal
    return float(x.strip('%'))/100

host_raiting['response_rate'] = host_raiting['response_rate'].apply(lambda x: p2f(x))
host_raiting['acceptance_rate'] = host_raiting['acceptance_rate'].apply(lambda x: p2f(x))

# feat engieering of true and falses
def tof(x):
    if x == 't':
        return 1
    elif x == 'f':
        return 0
    
host_raiting['is_superhost'] = host_raiting['is_superhost'].apply(lambda x: tof(x))
host_raiting['has_profile_pic'] = host_raiting['has_profile_pic'].apply(lambda x: tof(x))
host_raiting['identity_verified'] = host_raiting['identity_verified'].apply(lambda x: tof(x))
host_raiting['instant_bookable'] = host_raiting['instant_bookable'].apply(lambda x: tof(x))

# one hot encoding using sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder()

# label encoding response time
encoded = le.fit_transform(host_raiting['response_time'])
le_encoded = encoded.reshape(len(encoded), 1)
host_raiting['response_time_encoded'] = le_encoded 
# print(le.inverse_transform(le_encoded))

# label encoding names
le2 = LabelEncoder()
names_encoded = le2.fit_transform(host_raiting['name'])
le_names_encoded = names_encoded.reshape(len(names_encoded), 1)
host_raiting['names_encoded'] = le_names_encoded 
# print(le2.inverse_transform(le_names_encoded))

# for uniq in host_raiting['response_time'].unique():
#     if uniq == 'within a few hours':
#         host_raiting[uniq] = 1
#     elif uniq == 'within an hour':
#         host_raiting[uniq] = 1
#     elif uniq == 'within a day':
#         host_raiting[uniq] = 1
#     elif uniq == 'a few days or more':
#         host_raiting[uniq] = 1
#     else:
#         host_raiting[uniq] = 0
display(host_raiting.head(50))
print(host_raiting.select_dtypes(include='object').columns)


# In[ ]:


demog_le = LabelEncoder()
demog_encoded = demog_raiting.apply(demog_le.fit_transform)  # simple way to label encode over multiple cols must be aware of 
# can find classes using the inverse_transfrom as before

demog_encoded.head()


# In[ ]:





# In[ ]:




