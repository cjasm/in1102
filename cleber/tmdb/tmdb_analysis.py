# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import ast
import itertools
import re
import json

"""
//TO DO
Feature Selection
"""

dataset = pd.read_csv('../data/train.csv')
shape = dataset.shape
print("Shape:", shape)
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

fields = dataset[["id", "title", "budget", "genres", "popularity", "release_date", "runtime", "revenue", "production_companies", "original_language"]]
pct_nans = round(fields.isnull().sum()/shape[0]*100,1).to_frame().sort_values(by=[0], ascending=False)
pct_nans.iloc[5,] = round((fields[["budget"]]==0).sum()/shape[0]*100,1).to_frame().iloc[0,0]
pct_nans = pct_nans.sort_values(by=[0], ascending=False)
plt.figure(figsize=(20,8))
sns.barplot(x=pct_nans.index, y=pct_nans[0])
plt.xticks(rotation=90)
plt.title("Percentage of missing values")
plt.ylabel("Missing values [%]")
plt.show()

fields.head()

fields.describe()

columns = ['revenue','budget','popularity','runtime']
plt.subplots(figsize=(10, 8))
correlations = fields[columns].corr()
sns.heatmap(correlations, xticklabels=columns,yticklabels=columns, linewidths=.5, cmap="Greens")

def get_genres(row):
    genres = []
    if pd.isna(row):
            return genres
    
    row = ast.literal_eval(row)
    for genre in row:
        genres.append(genre.get("name"))
    return genres
    
def get_release_date(row):
    yr = re.findall(r"\d+/\d+/(\d+)",row)

    if int(yr[0]) >= 18:
        return(row[:-2] + "19" + yr[0])
    else:
        return(row[:-2] + "20" + yr[0])
    
def get_production_companies(row):
    companies = []
    if pd.isna(row):
        return companies
    row = ast.literal_eval(row)
    for companie in row:
        companies.append(companie.get("name"))
    return companies

fields.loc[:, "production_companies"] = dataset["production_companies"].apply(get_production_companies).values
fields.loc[:, "genres"] = dataset["genres"].apply(get_genres).values
fields.loc[:, "release_date"] = dataset["release_date"].apply(get_release_date).values
fields.loc[:, "release_date"] = pd.to_datetime(fields.loc[:, "release_date"])
fields.loc[:,"release_year"] = fields.loc[:,"release_date"].dt.year
fields.loc[:,"release_month"] = fields.loc[:,"release_date"].dt.month

unique_genres = fields["genres"].apply(pd.Series).stack().unique()

genres_dummies = pd.get_dummies(fields["genres"].apply(pd.Series).stack()).sum(level=0)
genres_dummies.head()
fields_genres = pd.concat([fields, genres_dummies],axis=1, sort=False)
fields_genres.head(5)

genres_overall = fields_genres[unique_genres].sum().sort_values(ascending=False)
plt.figure(figsize=(15,5))
ax = sns.barplot(x=genres_overall.index, y=genres_overall.values)
plt.xticks(rotation=90)
plt.title("Popularity of genres overall")
plt.ylabel("count")
plt.show()

# Companies
from collections import Counter
unique_companies = fields["production_companies"].apply(pd.Series).stack().unique()
list_of_companies = fields['production_companies'].apply(lambda x: [i for i in x] if x else []).apply(pd.Series).stack().unique()

companies_overall = np.array(Counter(list_of_companies).most_common(30))
plt.figure(figsize=(15,5))
ax = sns.barplot(x=companies_overall[:, 0], y=companies_overall[:, 1].astype(int))
plt.xticks(rotation=90)
plt.title("Movies per company overall")
plt.ylabel("count")
plt.show()

# Language
fields.groupby("original_language").size()
sns.countplot(x='original_language', data=fields[fields.original_language !="en"])

# I guess we have to use two classifiers (english and others)
# budget x revenue per language
plt.figure(figsize=(15,5))
ax = sns.relplot(x="budget", y="revenue", data=fields[fields.original_language=="en"])
plt.xticks(rotation=90)
plt.title("Budget x Revenue (English)")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(15,5))
ax = sns.relplot(x="budget", y="revenue", hue="original_language", data=fields[fields.original_language!="en"])
plt.xticks(rotation=90)
plt.title("Budget x Revenue (Other Languages)")
plt.ylabel("Revenue")
plt.show()

# Popularity
plt.figure(figsize=(15,5))
ax = sns.relplot(x="popularity", y="revenue", data=fields)
plt.xticks(rotation=90)
plt.ylabel("Revenue")
plt.show()

# Language
plt.figure(figsize=(15,5))
ax = sns.boxplot(x="original_language", y="revenue", data=fields)
plt.xticks(rotation=90)
plt.ylabel("Revenue")
plt.show()

# Distributions
plt.figure(figsize=(15,5))
ax = sns.distplot(fields["revenue"])
plt.show()

plt.figure(figsize=(15,5))
ax = sns.distplot(fields["budget"])
plt.show()

plt.figure(figsize=(15,5))
ax = sns.distplot(fields["popularity"])
plt.show()

columns = ['revenue','budget','popularity', 'original_language', 'release_year']
sns.pairplot(fields[columns], hue='original_language')

sns.pairplot(fields[columns], hue='release_year')