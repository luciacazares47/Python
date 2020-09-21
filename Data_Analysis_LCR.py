# -*- coding: utf-8 -*-

print(""" Consulting Report for Sunset Oasis Hotel written by Lucia Cazares""")

print("""The objective of the following report is to understand Sunset Oasis Hotel's
competition by analyzing the data and providing recommendations for the hotel's
design which will help them achieve the goal of maximizing the score on 
Tripadvisor. The report will focus on specific actions that can be taken such as
the facilities that should be built (tennis court, spa, casino), what type of
travelers should be targeted and further additional recommendations. """)

#Let's begin by importing the packages 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm     
import seaborn                 
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from prettytable import PrettyTable
from Data_Analysis_Functions import make_table,is_iterable,normalize,ez_avg,\
ez_score_bars,ez_pie,ez_stacked_bar,avg_heatmap

#load file 
data_file = "las_vegas_hotels.xlsx"

#reading file into Python and checking data
hotels = pd.read_excel(data_file, sheet_name = "Sheet1")

print("""\n\n***Now, let's continue by exploring the data we have on hotels in
      Las Vegas***\n\n""")

#checking data types
hotels.info(verbose = True) 
print("""\n We can see that we have data with information of hotel characteristics
      (Hotel name, hotel stars, facilities, etc) and user characteristics 
      (user country, period of stay, traveler type, etc). Also, we can see that 
      the data has the correct data type for each column, now we can move on\n""")

#descriptive statistics of the data set
print("""\nNow we will view the data descriptive statistics to see if 
      we have any outliers and clean the data\n""")

#add +/- 3 standard deviations to view if there are any numbers out of the ordinary
hotels_desc = hotels.describe()
hotels_desc.loc["+3_std"] = hotels_desc.loc["mean"] + (hotels_desc.loc["std"] * 3)
hotels_desc.loc["-3_std"] = hotels_desc.loc["mean"] - (hotels_desc.loc["std"] * 3)
print(hotels_desc) #display descriptive statistics table

print("""We noticed that member years has a negative value which needs to be removed
      from the dataset because it represents an outlier that can affect our 
      analysis""")

print( """\nNow that we have viewed the data types and identified data
      that needs to be removed, we will clean the dataset\n""")

#cleaning up the data to eliminate negative values of member years
cols = list(hotels)

for col in cols:
    if is_numeric_dtype(hotels[col]):
        hotels = hotels.loc[hotels.loc[:, col]>=0]
        
#convert columns that have YES/NO variables to binary variables (0 and 1)

hotel_traits = ["Gym","Pool","Tennis court","Spa","Casino","Free internet"]

hotels.loc[: , hotel_traits] = hotels.loc[ : , hotel_traits]\
.replace(to_replace={ "NO": 0, "YES": 1},inplace = False)

print("""First, We will perform a correlation analysis to identify variables that are 
highly correlated between each other and make sure to keep only the most 
important ones to avoid affecting the analysis""")

df_corr = hotels.corr().round(2)
df_corr

"""We will plot a heatmap to have a better view of the variable correlation"""

fig, ax = (plt.subplots(figsize = (10, 10)))  # setting the plot size

# visualizing the correlation matrix
sns.heatmap(data       = df_corr,
            cmap       = 'Blues',
            square     = True,
            annot      = True,
            linecolor  = 'black',
            linewidths = 0.5)
fig, ax.set_title("Correlation heatmap")

plt.show() # displaying the heatmap

print("""From the correlation heatmap we can see that "Nr.hotel reviews", "Nr.reviews" 
and "Helpful votes" are highly correlated, therefore we need to remove
some of these variables to avoid affecting the model.

We can also see that the variables have a really low
correlation with "Score", almost zero. This is likely to give us poor results
when building the Linear Regression model""")

#drop the highly correlated variable - Nr.reviews
hotels_after = hotels.drop("Nr. reviews",axis=1)

"""we will rename the data from "Period of stay" column to fit season names 
to better understand the preference of travelers in each one"""

hotels_after["Period of stay"] = hotels_after["Period of stay"]\
.map({"Dec-Feb": "winter","Mar-May" : "spring","Jun-Aug" : "summer", "Sep-Nov" : "autumn"})

print("""Ready! Now we can analyse the data further to start looking for insights""")

# explore how many hotels we have in the dataset
num_hotels = hotels_after["Hotel name"].nunique()
print(num_hotels)
print("\nThere are", num_hotels, "hotels in the dataset")

#explore the traveler types

groupers = ["Traveler type"] #create list of variables to group
fig, ax = (plt.subplots(figsize = (10, 10)))
print(ez_pie(hotels_after,groupers))  #create a pie chart to view results

print("""\nThe majority of travelers are couples, business, and families.""")

#explore the period of stay with the highest scores
groupers = ["Period of stay"]
print(ez_avg(hotels_after,groupers))

print("\nSummer and spring are the highest scored seasons for travelers")

#lets take a look at the seasons by traveler type with a heatmap

idx = ["Traveler type"]
cols = ["Period of stay"]
fig, ax = (plt.subplots(figsize = (10, 10)))
print(avg_heatmap(hotels_after,idx,cols))

print("""\nAutumn for solo travelers, winter for friends, summer for families, 
      summer for couples and spring for business are the best scored""")

#explore the countries

grouped = hotels_after.groupby("User country")["User country"].count().sort_values(ascending = False)
print(grouped)

print("""\n85% of travelers come from USA, UK and Canada""")

#exploration the average score by main countries and traveller type

groupers = ["User country","Traveler type"]
fig, ax = (plt.subplots(figsize = (10, 10)))
print(ez_score_bars(hotels_after.loc[hotels_after["User country"].isin(["USA","UK",\
                               "Canada"])],groupers))

print("""\nThe highest scores are given by travelers from Canada that travel
      with Friends, Solo travelers from UK and couple from Canada""")

#exploring the facilities of the hotels (tennis court, spa and casino)

groupers = ["Pool","Gym","Casino","Tennis court","Spa","Free internet"]
print(ez_avg(hotels_after,groupers))

print("""\nAll of the hotels with highest scores offer Free internet.""")

print("""\nNow we will create our model""")

#create dummy variables for the variables that are in text form

hotel_dummy = hotels_after.copy()

text_cols   = ["User country","Period of stay","Traveler type",\
                                 "User continent","Review month",\
                                 "Review weekday","Hotel name"]

hotel_dummy = pd.get_dummies(data=hotel_dummy, columns=text_cols, drop_first=False)
    
list(hotel_dummy) #checking which variables are available for modeling
    
#preparing input and output data 
hotel_data   = hotel_dummy.drop(["Score"],axis = 1)
hotel_target = hotel_dummy[["Score"]]

"""We will select the most contributing features to use in our model"""

rfe = RFE(estimator = Ridge(), n_features_to_select = 12)
rfe.fit(hotel_data, hotel_target)
feature_list  = pd.DataFrame({'col':list(hotel_data.columns.values),'sel':list(rfe.support_ *1)})
sel_variables = feature_list[feature_list.sel==1].col.values

print("\nWe have identified the most contributing features for 'Score' which are:\n")
print(sel_variables)
print('-'*100)

#store selected contributing features

hotel_data_sel   = hotel_dummy.loc[ : ,list(sel_variables)]
hotel_target_sel = hotel_dummy[["Score"]]

"""We will start the model based on selected features"""

#we will test three different models to chose the best one 

"""Linear Regression Model"""

rm = LinearRegression()
rm_fit = rm.fit(hotel_data_sel, hotel_target_sel)
rm_pred = rm_fit.predict(hotel_data_sel)

#Intercept and coefficients

rm_intercept = rm.intercept_[0].round(3)
coef         = rm.coef_[0].round(3)
rm_score     = rm.score(hotel_data_sel,hotel_target_sel).round(6)

print('-'*100)

"""Ridge model"""

ridge_model = Ridge()
ridge_fit   = ridge_model.fit(hotel_data_sel,hotel_target_sel)
ridge_pred  = ridge_model.predict(hotel_data_sel)

ridge_score = ridge_model.score(hotel_data_sel,hotel_target_sel).round(6)

"""Lasso model"""

lasso_model = Lasso()
lasso_fit   = lasso_model.fit(hotel_data_sel,hotel_target_sel)
lasso_pred  = lasso_model.predict(hotel_data_sel)

lasso_score = lasso_model.score(hotel_data_sel,hotel_target_sel).round(6)

#comparing results

print("""These are the results of running three different models based on the
      features selected.\n""")
print(f"""
Model      Score
-----      -----
LinearReg  {rm_score}
Ridge      {ridge_score}
Lasso      {lasso_score}
""")

print('-'*100)

# P-values
rm_stats        = sm.OLS(hotel_target_sel, hotel_data_sel)
rm_stats_fit    = rm_stats.fit()
rm_stats_output = rm_stats_fit.summary2().tables[1][['P>|t|']].round(3)

print("\n\n***Summary statistics***\n\n")

#Adjusted R-squared
total_rows = len(hotels_after)
x_var_rows = len(sel_variables)
Adj_R2     = round(1-(1-rm_score)*(total_rows-1)/(total_rows-x_var_rows-1),6)

#Mean absolute deviation (MAD)

rm_pred        = rm_pred.flatten()
rm_pred_series = pd.Series(rm_pred)

MAD = round(rm_pred_series.mad(),6)

#degress of freedom

DOF = round(total_rows - (x_var_rows + 1))

#creating a table to view summary statistics results

t = PrettyTable(["Statistics","Values"])
t.add_row(["Model type","Linear Regression"])
t.add_row(["R-Square",rm_score])
t.add_row(["Adjusted R-Square",Adj_R2])
t.add_row(["Mean Absolute Deviation",MAD])
t.add_row(["Model size",x_var_rows])
t.add_row(["Degrees of Freedom",DOF])

print(t.get_string(title="Summary Statistics"))


print("""\nEven when selecting the best features, each model returns really low scores, 
meaning that the way the data is organized does not let the model reach any
conclusions. We will refine the model by reorganizing the dataset by grouping columns. 
This will make the model more accurate because it will have one row per hotel 
with each of its features.""")

"""Restructuring the data to have better results for the model"""

#grouping the dataset by "Hotel name" and "Nr. hotel reviews"

reviews = hotels_after.groupby(["Hotel name"])["Nr. hotel reviews"].sum().reset_index()
reviews.rename(columns={"Nr. hotel reviews": "total_reviews"}, inplace=True)

#merging new "Total reviews column" in original dataset

hotels_mod = pd.merge(hotels_after, reviews, how="left", on="Hotel name")

#create a weighted average for the Nr.hotel reviews vs the Total reviews

hotels_mod["review_wt"] = hotels_mod["Nr. hotel reviews"] / hotels_mod["total_reviews"]

#Multiply the new review weighted average with the hotel Score to have a weighted score by hotel
hotels_mod["wtd_score"] = hotels_mod["review_wt"] * hotels_mod["Score"]

#create a list of the variables that need to be grouped 
groupers = ["Hotel name", 'Pool',
 'Gym',
 'Tennis court',
 'Spa',
 'Casino',
 'Free internet',
 'Hotel stars',
 'Nr. rooms']

#create a new dataset with the weighted scores and groupers list
hotels_wtd = hotels_mod.groupby(groupers)["wtd_score"].sum().reset_index()

"""we normalized the data of "Hotel stars" and "Nr. rooms" by putting 
all the variables in the same scale. This way we will avoid unnecesary upward 
or downward trends"""

hotels_wtd["norm_stars"] = normalize(hotels_wtd, "Hotel stars")
hotels_wtd["norm_rooms"] = normalize(hotels_wtd, "Nr. rooms")

print("""\nWe will view the correlation matrix one more time with the new dataset""")

new_df_corr = hotels_wtd.corr().round(2)
new_df_corr

fig, ax = (plt.subplots(figsize = (10, 10)))  # setting the plot size

# visualizing the correlation matrix
sns.heatmap(data       = new_df_corr,
            cmap       = 'Blues',
            square     = True,
            annot      = True,
            linecolor  = 'black',
            linewidths = 0.5)
fig, ax.set_title("Correlation heatmap")

plt.show() # displaying the heatmap

print("""\nIn the heatmap we can see that Pool ans Free internet have the highest
      correlation with Score""")

#divide the variables between numerical and categorical

x_num = ['norm_stars']

x_cat = ['Pool',
 'Gym',
 'Spa',
 'Casino',
 'Free internet']

#create dummy variables of the columns that have text (categorical)

x_dummies = pd.get_dummies(hotels_wtd[x_cat], drop_first=True)

#prepare a list with the explanatory and target variables 

x_data   = pd.concat([x_dummies, hotels_wtd[x_num]], axis=1)
y_target = hotels_wtd["wtd_score"]

#saving dataset with all the changes made for future analysis

hotels_final = pd.concat([x_data,y_target],axis = 1)

print("""\nAfter reorganizing the data, we will run the model again""")

"""Linear Regression"""

new_rm       = LinearRegression()
new_rm_fit   = new_rm.fit(x_data,y_target) #fit the training data 
new_rm_pred  = new_rm_fit.predict(x_data)
new_coef     = new_rm.coef_.round(3)
new_rm_score = new_rm.score(x_data, y_target).round(6) #scoring the model

#Adjusted R-squared
total_rows = len(hotels_final)
x_var_rows = len(x_data.columns)
new_Adj_R2     = round(1-(1-new_rm_score)*(total_rows-1)/(total_rows-x_var_rows-1),6)

#Mean absolute deviation (MAD)

rm_pred        = new_rm_pred.flatten()
rm_pred_series = pd.Series(rm_pred)

new_MAD = round(rm_pred_series.mad(),6)

#degress of freedom

new_DOF = round(total_rows - (x_var_rows + 1))

"""Summary statistics of re-calculated model"""

#creating a table to view summary statistics results

t = PrettyTable(["Statistics","Values"])
t.add_row(["Model type","Linear Regression"])
t.add_row(["R-Square",new_rm_score])
t.add_row(["Adjusted R-Square",new_Adj_R2])
t.add_row(["Mean Absolute Deviation",new_MAD])
t.add_row(["Model size",x_var_rows])
t.add_row(["Degrees of Freedom",new_DOF])

print(t.get_string(title="Summary Statistics"))

#p-values with new data 

new_rm_stats            = sm.OLS(y_target, x_data)
new_rm_stats_fit        = new_rm_stats.fit()
new_rm_stats_output = new_rm_stats_fit.summary2().tables[1][['P>|t|']].round(3)

print(rm_stats_output)

#95% confidence interval boundaries (upper and lower)

hotels_const = sm.add_constant(x_data)
conf_int     = new_rm_stats_fit.conf_int(alpha = 0.05, cols = None)
Lower_95     = round(pd.Series(conf_int.loc[ : ,0]),3)
Upper_95     = round(pd.Series(conf_int.loc[ :,1]),3)

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
                input_dict["Coefficients"] = new_coef
            if item == "2":
                input_dict["P-Values"] = new_rm_stats_output.iloc[:, 0]
            if item == "3":
                input_dict["Conf. Int: Lower"] = list(Lower_95)
            if item == "4":
                input_dict["Conf. Int: Upper"] = list(Upper_95)
         
        table_title = "Variable-level data"
        make_table(table_title, x_data.columns, "Variables", **input_dict)        
        
        input_list = []

print("""\nSummary of insights gathered from the data:
    
    1.Traveler types: The majority of travelers are couples, business and families.
    
    2.The highest scoring seasons per traveler type is autumn for solo travelers, 
    winter for friends, summer for families, summer for couples and spring 
    for business.
    
    3.85% of travelers come from USA, UK and Canada. 
    
    4.The highest scores are given by travelers from Canada that travel
    with friends, solo travelers from UK and couples from Canada
    
    4.All of the  with highest scoring hotels offer Free internet""")
    
print("""\nSummary of insights gathered from the model:
    
    1.The facilities that have higher correlation with Score are Pool and 
    Free internet
    
    2.Pool, Gym, free internet and casino have the highest coefficients,
    meaning that they will have the highest impact on the score.
    
    3. The hotel stars also influence the Score, this means that the higher 
    the hotel stars, its most likely that the Score will increase because it is
    more probable that the hotel will have more facilities.""")
    
print("""\nRecommendations: 
    
    1. Sunset Oasis hotel should focus on building a Casino, Pool and Gym. 
    These three facilities have a higher impact on Score than building a Tennis
    court or a spa. Building a Pool is likely to increase the Score by 1.2, 
    building a Gym will likely increase the score by 0.5 and building a Casino
    will likely increase the score by 0.4.
    
    2. In addition, Sunset Oasis hotel should most definitely offer free internet
    as this is likely to increase the score by 1. Also, because most competitors
    are offering free internet as well and not doing so would put Sunset Oasis
    on a disadvantage.
    
    3.Finally, Sunset Oasis hotel should focus on targeting travelers from US,
    Canada and UK. Specifically, customers that travel in couples, business and 
    families because these are the most common traveler types. 
    
    4.The hotel should also take into account the seasons, and target the marketing
    efforts accordingly. For example,.in autumn it is better to target solo travelers,
    winter targeted for groups that travel with friends, summer for couples and spring
    for the ones traveling for business.""")



