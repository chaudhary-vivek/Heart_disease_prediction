# Part 0: Importing libraries and data

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

# Three algorithms will be used, KNN and Random forest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Importing the dataset from csv to dataframe
df = pd.read_csv('heart.csv')

# Part 1: Feature selection

# Getting correlation matrix for all features
import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
# The target is highly negatively correlated with exang, oldpeak, ca and thal
# The target is highly positively correlated with cp, thalach and slope

# Checking if the dataset is balanced
sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')
# Dataset appears balanced

# Part 2: Data processing

# Getting dummy variables for all the categorical features
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# We standarsize the continuous variables using standars scaler
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Making a separate dataset for the target variable
y = dataset['target']
# Dropping target variable from the set
X = dataset.drop(['target'], axis = 1)

# Part 3: Making and evaluating models

# KNN

#Importing cross validation score to evaluate the model
from sklearn.model_selection import cross_val_score
# Starting a loop to see cross validation score from k=1 to k=21 and storing them in knn_scores
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
# The score is maximum at k=12
# Evaluating KNN model with k=12
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)
print(score.mean())

# Random forest classifier

# Importing random forest calssifier
from sklearn.ensemble import RandomForestClassifier
# Making the model
randomforest_classifier= RandomForestClassifier(n_estimators=10)
# Evaluating the model
score=cross_val_score(randomforest_classifier,X,y,cv=10)
print(score.mean())

# The random forest has an accuracy of 82.49%
# Hence, it is better to use KNN which has an accuracy of 84.45%
