#!/usr/bin/env python
# coding: utf-8

# In[71]:


from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn import preprocessing
# import tensorflow as tf
import seaborn as sns


# In[72]:


df = pd.read_csv('titanic.csv')
df.head()


# In[73]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df


# In[74]:


def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df


# In[75]:


def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df


# In[76]:


def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)


# In[77]:


def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = drop_features(df)
    return df
df_ready = transform_features(df)


# In[78]:


# ###
def encode_features(df):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_ready[feature])
        df_ready[feature] = le.transform(df_ready[feature])
    return df_ready
df_ready = encode_features(df_ready)


# In[79]:


df_ready


# In[88]:


scaler = preprocessing.StandardScaler()
scaler.fit(df_ready)

preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_data = scaler.transform(df_ready)

# PCA
pca = PCA(n_components=3)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

X = x_pca[:,0]
Y = x_pca[:,1]
Z = x_pca[:,2]


# In[100]:


features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']

for feature in features:
        fig = plt.figure(figsize=(6,4), dpi=95)

        ax = Axes3D(fig)
        p = ax.scatter(X, Y, Z, c=df_ready[f"{feature}"], marker='o') 
        plt.title(f"{feature}")
        plt.colorbar(p) 
plt.show()

