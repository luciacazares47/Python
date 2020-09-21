# -*- coding: utf-8 -*-

""" Model Template by Lucia Cazares"""

"""The following report analyzes TripAdvisor data for 21 hotel in Las Vegas""" 

#importing libraries

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm                      
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from prettytable import PrettyTable


"""Function Definitions"""

#Function to create summary table with each variable and its traits
def make_table(table_title, elements, element_name, **traits):
    #Check if all traits are list-like and if they have same length
    
    if is_iterable(elements):
    
        non_iterables = []
        dif_len = []
        for key, value in traits.items():
            if is_iterable(value):
                trait_len = len(value)
                
                if trait_len != len(elements):
                    dif_len.append(key)
            else:
                non_iterables.append(key)
                
        break_flag = 0
        if non_iterables:
            print("Unable to build table: the following variables are not list-like:")
            print(non_iterables)
            break_flag = 1
            
        if dif_len:
            print("Unable to build table: the following lists have different lengths:")
            print(dif_len)
            break_flag = 1
            
        if not break_flag:
            trait_list = list(traits.keys())
            t = PrettyTable([element_name] + trait_list)
            t.title = table_title
            for i in range(len(elements)):
                row_list = []
                
                row_list.append(elements[i])
                
                for trait in trait_list:
                    row_list.append(traits[trait][i])
                
                t.add_row(row_list)
                
        print(t) 
        
    else:
        
        print("Impossible to build table: elements are not in a list")
        
#Helper function to make sure something is list-like       
def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

#load file 
data_file = "las_vegas_hotels.xlsx"

#reading file into Python and checking data
hotels = pd.read_excel(data_file, sheet_name = "Sheet1")

print("""\n\n***Data exploration***\n\n""")

#checking data types
print("""\n**Data type**\n""")
hotels.info(verbose = True) #the data has the correct data type for each column

#descriptive statistics of the data set
print("\n\n**Data descriptive statistics**\n\n")
hotels_desc = hotels.describe()

#explore how the data is distributed to see if there are any outliers
hotels_desc.loc["+3_std"] = hotels_desc.loc["mean"] + (hotels_desc.loc["std"] * 3)
hotels_desc.loc["-3_std"] = hotels_desc.loc["mean"] - (hotels_desc.loc["std"] * 3)

print(hotels_desc) #display descriptive statistics table

"""We noticed that member years has a negative value which needs to be removed.
Also aprox. 3 out of 4 people that are giving reviews on TripAdvisor are giving a 
4 or higher - a review of 4 is an average review, any score less than 3 is 
probably very bad"""

#explore the data columns to understand where the majority of data lies

hotels.groupby(["User country"])["User country"].count()
hotels.groupby(["User continent"])["User continent"].count()

"""These groupings help us see that some countries have data that is not
representative to enter the model"""

print( """\n\n***Data cleanup***\n\n""")

#filtering user countries that have too few inputs to be representative

user_country = ["Australia","Canada","India","Ireland","UK","USA"] #main user countries
hotels = hotels[hotels["User country"].isin(user_country)] #filter user country

#cleaning up the data to eliminate negative values of member years

cols = list(hotels)

for col in cols:
    if is_numeric_dtype(hotels[col]):
        hotels = hotels.loc[hotels.loc[:, col]>=0]
        
#convert columns that have YES/NO variable to 0 and 1

hotel_traits = ["Gym","Pool","Tennis court","Spa","Casino","Free internet"]

hotels.loc[: , hotel_traits]\
= hotels.loc[ : , hotel_traits]\
.replace(to_replace={ "NO": 0, "YES": 1},inplace = False)

#checking results
print(hotels) #check that changes were applied correctly

#checking correlation between variables in original dataset
df_corr_original = hotels.corr().round(2)
df_corr_original

"""We will plot a heatmap to have a better view of the variable correlation"""

 # setting the plot size
fig, ax = (plt.subplots(figsize = (10, 10)))

# visualizing the correlation matrix
sns.heatmap(data       = df_corr_original,
            cmap       = 'Blues',
            square     = True,
            annot      = True,
            linecolor  = 'black',
            linewidths = 0.5)

# saving the figure and displaying the plot
print("\n\n*Correlation matrix of original dataset*\n\n")
plt.show()

"""From the correlation heatmap we can see that Nr.hotel reviews, Nr.reviews 
and Helpful votes are highly correlated, therefore it is important to remove
some of these variables to avoid affecting the model."""

#drop the highly correlated variable - Nr.reviews
hotels_after = hotels.drop("Nr. reviews",axis=1)

"""now we will check the correlation heatmap one more time to see results"""

#checking correlation between variables
df_corr = hotels_after.corr().round(2)
df_corr

 # setting the plot size
fig, ax = (plt.subplots(figsize = (10, 10)))

# visualizing the correlation matrix
sns.heatmap(data       = df_corr,
            cmap       = 'Blues',
            square     = True,
            annot      = True,
            linecolor  = 'black',
            linewidths = 0.5)

# saving the figure and displaying the plot
print("\n\n*Correlation matrix after removing highky correlated values*\n\n")
plt.show()

"""In the correlation heatmap we can see that the variables have a really low
correlation with score, almost zero. This is likely to give us poor results
when building the Linear Regression model"""

print("""\n\n***Regression Analysis***\n\n""")

#now we will create dummy variables for the variables that are in text form

hotel_dummy = hotels_after.copy()

text_cols = ["User country","Period of stay","Traveler type",\
                                 "User continent","Review month",\
                                 "Review weekday"]

hotel_dummy = pd.get_dummies(data=hotel_dummy, columns=text_cols, drop_first=False)
    
list(hotel_dummy) #checking which variables are available for modeling
    
#preparing data variable (input) -  all except "Hotel name"and "Score" because
#"Hotel name" does not affect the "Score" variable and "Score" is the target variable

hotel_data = hotel_dummy.drop(["Score","Hotel name"],axis = 1)

#preparing target variable (output) - Score
hotel_target = hotel_dummy[["Score"]]

#split dataset into training and testing portion
X_train,X_test,y_train,y_test = train_test_split(hotel_data,hotel_target,\
                                test_size = .20, random_state = 1)

"""We will select the most contributing features to use in our model"""

rfe = RFE(estimator = Ridge(), n_features_to_select = 12)
rfe.fit(X_train, y_train)
feature_list = pd.DataFrame({'col':list(X_train.columns.values),'sel':list(rfe.support_ *1)})
x_variables = feature_list[feature_list.sel==1].col.values

print("The most contributing features in 'Score' are:\n")
print(x_variables)
print("\nThese features will be used to calculate R-squared")
print('-'*100)

#store selected contributing features (x-variables) and target variable (y-variable)

hotel_data_sel = hotel_dummy.loc[ : ,list(x_variables)]

hotel_target_sel = hotel_dummy[["Score"]]

#start model based on selected features

X_train,X_test,y_train,y_test = train_test_split(hotel_data_sel,hotel_target_sel,\
                                test_size = .20, random_state = 1)
#create instance of the model
rm = LinearRegression()

#fit the model
rm_fit = rm.fit(X_train, y_train)

#Intercept and coefficients
rm_pred = rm_fit.predict(hotel_data_sel)
intercept = rm.intercept_[0]
coef = rm.coef_[0].round(3)

print("The intercept for our model is {:.4}".format(intercept))
print('-'*100)

#R-square results
print("R-Square Score is ",rm.score(hotel_data_sel,hotel_target_sel).round(6))
print('-'*100)

rm_score = rm.score(hotel_data_sel,hotel_target_sel).round(6)

# P-values
rm_stats        = sm.OLS(hotel_target_sel, hotel_data_sel)
rm_stats_fit    = rm_stats.fit()
rm_stats_output = rm_stats_fit.summary2().tables[1][['P>|t|']].round(3)

print("\n\n***Summary statistics***\n\n")

""" Output: summary statistics - model type, R-square, Mean absolute deviation,
model size, degrees of freedom. 
Outputs in variable-level data: Coefficients, p-values, 95% confidence interval
boundaries (Upper and Lower)"""

#Adjusted R-squared
total_rows = len(hotels_after)
x_var_rows = len(x_variables)
Adj_R2 = round(1-(1-rm_score)*(total_rows-1)/(total_rows-x_var_rows-1),6)

#Mean absolute deviation (MAD)

rm_pred = rm_pred.flatten()
rm_pred_series = pd.Series(rm_pred)

MAD = round(rm_pred_series.mad(),6)

#degress of freedom

DOF = round(total_rows - (x_var_rows + 1))

#95% confidence interval boundaries (upper and lower)

hotels_const = sm.add_constant(hotel_data_sel)
conf_int = rm_stats_fit.conf_int(alpha = 0.05, cols = None)
Lower_95 = round(pd.Series(conf_int.loc[ : ,0]),3)
Upper_95 = round(pd.Series(conf_int.loc[ :,1]),3)

#creating a table to view summary statistics results

t = PrettyTable(["Statistics","Values"])
t.add_row(["Model type","Linear Regression"])
t.add_row(["R-Square",rm_score])
t.add_row(["Adjusted R-Square",Adj_R2])
t.add_row(["Mean Absolute Deviation",MAD])
t.add_row(["Model size",x_var_rows])
t.add_row(["Degrees of Freedom",DOF])

print(t.get_string(title = "Summary statistics"))

print("""\n\n***Variable-level data***\n\n""") 

#Let user choose how to build her variable-level data table

table_build =input("""Would you like to build a table to summarize the model's variable-level data?: [Y/N] """)

if table_build:

    input_flag = 1
    input_list = []
    input_validation_list = ["1", "2", "3", "4"]
    while input_flag == 1:
        user_input = input("""Please choose a trait to include in the table\
                           
        1 = Coefficients
        2 = P-values
        3 = Confidence Interval: Lower
        4 = Confidence Interval: Upper
        Any other input = start building the table!""")
        
        if user_input in input_validation_list:
            if user_input not in input_list:
                input_list.append(user_input)
            else:
                print("\nYou already chose that trait! Please choose another answer")
        else:
            input_flag = 0
    
    if input_list:
        input_dict = {}
        for item in input_list:
            if item == "1":
                input_dict["Coefficients"] = coef
            if item == "2":
                input_dict["P-Values"] = rm_stats_output.iloc[:, 0]
            if item == "3":
                input_dict["Conf. Int: Lower"] = list(Lower_95)
            if item == "4":
                input_dict["Conf. Int: Upper"] = list(Upper_95)
         
        table_title = "Variable-level data"
        make_table(table_title, x_variables, "Variables", **input_dict)        
        
        input_list = []

            













