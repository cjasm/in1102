# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import ast
import re
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder, MultiLabelBinarizer

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

def preprocessing(dataset):
    
    fields = dataset[["revenue", "genres", "release_date", "production_companies", "original_language", "budget", "popularity"]]
    
    fields.loc[:, "production_companies"] = dataset["production_companies"].apply(get_production_companies).values
    fields.loc[:, "genres"] = dataset["genres"].apply(get_genres).values
    fields.loc[:, "release_date"] = dataset["release_date"].apply(get_release_date).values
    fields.loc[:, "release_date"] = pd.to_datetime(fields.loc[:, "release_date"])
    fields.loc[:,"release_year"] = fields.loc[:,"release_date"].dt.year
    fields.loc[:,"release_month"] = fields.loc[:,"release_date"].dt.month
    fields = fields.drop("release_date", axis=1)
    
    X = fields.iloc[:, 1:8].values
    y = fields.iloc[:, 0].values

    # Taking care of missing data
    imputer = SimpleImputer(missing_values = 0, strategy = 'mean')
    imputer = imputer.fit(X[:, -4].reshape(-1,1))
    X[:, -4] = imputer.transform(X[:, -4].reshape(-1,1))[:,0]
    
    # Feature Scaling
    # sc_X = StandardScaler()
    # X[:, 3:7] = sc_X.fit_transform(X[:, 3:7])
    # n_X = Normalizer()
    # X[:, 3:7] = n_X.fit_transform(X[:, 3:7])
    
    le_X = LabelEncoder()
    X[:, 2] = le_X.fit_transform(X[:, 2])
    
    # Encoding Genres
    unique_genres = fields["genres"].apply(pd.Series).stack().unique()
    mlb_X_genres = MultiLabelBinarizer(classes=unique_genres)
    X = np.concatenate((X, mlb_X_genres.fit_transform(X[:, 0])),axis=1)
    X = np.delete(X, 0, axis=1)
    
    # Encoding Companies
    list_of_companies = fields['production_companies'].apply(lambda x: [i for i in x] if x else []).apply(pd.Series).stack()
    companies_overall = np.array(Counter(list_of_companies).most_common(30))
    
    mlb_X_companies = MultiLabelBinarizer(classes=companies_overall[:,0])
    X = np.concatenate((X, mlb_X_companies.fit_transform(X[:, 0])),axis=1)
    X = np.delete(X, 0, axis=1)
    
    return X, y