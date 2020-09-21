# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:15:15 2020

@author: lucia
"""

import pandas as pd
import seaborn
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
            t          = PrettyTable([element_name] + trait_list)
            t.title    = table_title
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

#function to normalize the variables that have numbers
        
def normalize(data, col):
    maximum   = data[col].max()
    minimum   = data[col].min()
    max_range = maximum - minimum
    return (data[col] - minimum) / max_range

#Function to group data and view weighted score
def ez_avg(data, groupers, wt_var="Nr. hotel reviews"):
    new_data = data.copy()
    new_data = pd.merge(new_data, new_data.groupby(groupers)[wt_var].sum().reset_index().rename(columns={wt_var: "total_wt_var"}), how="left") 
    new_data["wtd_score"] = new_data["Score"] * new_data["Nr. hotel reviews"] / new_data["total_wt_var"]
    new_data = new_data.groupby(groupers)["wtd_score"].sum().reset_index()
    return new_data.sort_values(by = ["wtd_score"],ascending = False)

#Function to group data and create a bar chart by weighted score
    
def ez_score_bars(data, groupers, wt_var="Nr. hotel reviews"):
    new_data = data.copy()
    new_data = pd.merge(new_data, new_data.groupby(groupers)[wt_var].sum().reset_index().rename(columns={wt_var: "total_wt_var"}), how="left") 
    new_data["wtd_score"] = new_data["Score"] * new_data["Nr. hotel reviews"] / new_data["total_wt_var"]
    new_data = new_data.groupby(groupers)["wtd_score"].sum()
    new_data.sort_values().plot.bar()

#Function to create a pie chart

def ez_pie(data, groupers, sum_var="Nr. hotel reviews"):
    data.groupby(groupers)[sum_var].sum().plot.pie()
    
#Function to create a stacked bar chart
    
def ez_stacked_bar(data, idx, cols, sum_var="Nr. hotel reviews"):
    pd.pivot_table(data, index=idx, columns=cols, values=sum_var).plot.bar(stacked=True)
    
#Function to create a heatmap  by weighted score
    
def avg_heatmap(data, idx, cols, wt_var="Nr. hotel reviews"):
    new_data = ez_avg(data, idx + cols, wt_var)
    new_data = pd.pivot_table(new_data, index=idx, columns=cols, values="wtd_score")
    seaborn.heatmap(new_data)
