#!/usr/bin/env python
# coding: utf-8

# # Course: [Master Machine Learning with scikit-learn](https://courses.dataschool.io/view/courses/master-machine-learning-with-scikit-learn)
# 
# ## Chapters 6-9
# 
# *© 2022 Data School. All rights reserved.*

# # Chapter 6: Encoding text data

# ## 6.1 Vectorizing text

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import set_config


df = pd.read_csv('http://bit.ly/MLtrain', nrows=10)
y = df['Survived']


df_new = pd.read_csv('http://bit.ly/MLnewdata', nrows=10)


ohe = OneHotEncoder()
logreg = LogisticRegression(solver='liblinear', random_state=1)


set_config(display='diagram')


df


# **Ideas for encoding the Name column:**
# 
# - **OneHotEncoder:** Each full name is treated as a category (not recommended)
# - **CountVectorizer:** Each word in a name is treated independently (recommended)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


dtm = vect.fit_transform(df['Name'])
dtm


# **CountVectorizer vs other transformers:**
# 
# - **CountVectorizer:** 1-dimensional input (Series)
# - **Other transformers:** 2-dimensional input (DataFrame)

print(vect.get_feature_names())


# **Default settings for CountVectorizer:**
# 
# - Convert all words to lowercase
# - Remove all punctuation
# - Exclude one-character words

# **About the document-term matrix:**
# 
# - 10 rows and 40 columns
# - Rows represent rows from training data, columns represent words
# - Rows are "documents", feature names are "terms"
# - Sparse matrix

# **How to examine a document-term matrix:**
# 
# 1. Use toarray method to make it dense
# 2. Convert dense matrix into a DataFrame
# 3. Use feature names as column headings

pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())


df.head(1)


# **"Bag of Words" representation:**
# 
# - Ignores word order
# - Only counts how many times a word appears

# ## 6.2 Including text data in the model

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']
X = df[cols]
X


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    ('passthrough', ['Parch', 'Fare']))


ct.fit_transform(X)


ct.get_feature_names() 


# **Output columns:**
# 
# - **Columns 1-3:** Embarked
# - **Columns 4-5:** Sex
# - **Columns 6-45:** Name
# - **Column 46:** Parch
# - **Column 47:** Fare

pipe = make_pipeline(ct, logreg)


pipe.fit(X, y)


X_new = df_new[cols]


pipe.predict(X_new)


# ## 6.3 Q&A: Why is the document-term matrix stored as a sparse matrix?

text = ['Machine Learning is fun', 'I am learning Machine Learning']


pd.DataFrame(vect.fit_transform(text).toarray(), columns=vect.get_feature_names())


dtm = vect.fit_transform(text)
dtm


print(dtm)


# **Preferred matrix representation:**
# 
# - **Most elements are zero:** Sparse matrix
# - **Most elements are non-zero:** Dense matrix

# ## 6.4 Q&A: What happens if the testing data includes new words?

text


dtm = vect.fit_transform(text)
vect.get_feature_names()


text_new = ['Data Science is FUN!']


vect.transform(text_new).toarray()


# **CountVectorizer methods:**
# 
# - **fit:** Learn the vocabulary
# - **transform:** Create the document-term matrix using that vocabulary

# ## 6.5 Q&A: How do I vectorize multiple columns of text?

df[['Name', 'Ticket']]


vect.fit_transform(df['Name'])


vect.fit_transform(df['Ticket'])


vect.fit_transform(df[['Name', 'Ticket']])


ct = make_column_transformer(
    (vect, 'Name'),
    (vect, 'Ticket'))


ct.fit_transform(df)


ct


ct.named_transformers_.keys()


# ## 6.6 Q&A: Should I one-hot encode or vectorize categorical features?

df[['Embarked', 'Sex']]


vect.fit_transform(df['Sex']).toarray()


vect.fit_transform(df[['Embarked', 'Sex']]).toarray()


vect.fit_transform(df['Embarked']).toarray()


# **Advantages of OneHotEncoder for categorical data:**
# 
# - Encodes multiple columns at once
# - Allows one-character category names
# - Gives more options for handling unknown categories

# # Chapter 7: Handling missing values

# ## 7.1 Introduction to missing values

# **Common sources of missing values:**
# 
# - Value purposefully wasn't collected
# - Error in the data collection process

df


# **Missing values vs unknown categories:**
# 
# - **Missing value:** Value encoded as "NaN"
# - **Unknown category:** Category not seen in the training data

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']
X = df[cols]
X


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    ('passthrough', ['Parch', 'Fare', 'Age']))


pipe = make_pipeline(ct, logreg)


pipe.fit(X, y)


# ## 7.2 Three ways to handle missing values

X.dropna()


# **Approach 1: Drop rows with missing values**
# 
# - May discard too much training data
# - May obscure a pattern in the "missingness"
# - Doesn't help you with new data

X.dropna(axis='columns')


# **Approach 2: Drop columns with missing values**
# 
# - May discard useful features

# **Approach 3: Impute missing values**
# 
# - **Benefit:** Keeps more samples and features
# - **Cost:** Imputed values may not match the true values

# **Factors to consider before imputing:**
# 
# - How important are the samples?
# - How important are the features?
# - What percentage of values would need to be imputed?
# - Are there other samples or features that contain the same information?
# - Is the missingness random?

# ## 7.3 Missing value imputation

from sklearn.impute import SimpleImputer
imp = SimpleImputer()


imp.fit_transform(X[['Age']])


# **Simple imputation strategies:**
# 
# - Mean value
# - Median value
# - Most frequent value
# - User-defined value

imp.statistics_


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    ('passthrough', ['Parch', 'Fare']))


ct.fit_transform(X)


pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)


(pipe.named_steps['columntransformer']
     .named_transformers_['simpleimputer']
     .statistics_)


X_new = df_new[cols]
X_new


pipe.predict(X_new)


# **What would have been imputed for Age in X_new?**
# 
# - Imputed value would be the mean of Age in **X**, not the mean of Age in **X_new**
# - Transformer is only allowed to learn from the training data

# **What do transformers learn from the training data?**
# 
# - **OneHotEncoder:** Learns categories
# - **CountVectorizer:** Learns vocabulary
# - **SimpleImputer:** Learns imputation value

# ## 7.4 Using "missingness" as a feature

imp_indicator = SimpleImputer(add_indicator=True)


imp_indicator.fit_transform(X[['Age']])


# **Why add a missing indicator?**
# 
# - Useful when the data is not missing at random
# - Can encode the relationship between "missingness" and the target value

# ## 7.5 Q&A: How do I perform multivariate imputation?

# **Types of imputation:**
# 
# - **Univariate imputation:** Only examines the feature being imputed
#   - SimpleImputer
# - **Multivariate imputation:** Takes multiple features into account
#   - IterativeImputer
#   - KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


imp_iterative = IterativeImputer()
imp_iterative.fit_transform(X[['Parch', 'Fare', 'Age']])


# **How IterativeImputer works:**
# 
# 1. **Age not missing:** Train a regression model to predict Age using Parch and Fare
# 2. **Age missing:** Predict Age using trained model

# **Notes about IterativeImputer:**
# 
# - Only works with numerical features
# - You have to decide which features to include
# - You can include multiple features with missing values
# - You can choose the regression model

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp_iterative, ['Parch', 'Fare', 'Age']))


from sklearn.impute import KNNImputer
imp_knn = KNNImputer(n_neighbors=2)
imp_knn.fit_transform(X[['Parch', 'Fare', 'Age']])


# **How KNNImputer works:**
# 
# 1. Find the row in which **Age is missing**
# 2. Find the n_neighbors "nearest" rows in which **Age is not missing**
# 3. Calculate the mean of Age from the "nearest" rows

# ## 7.6 Q&A: What are the best practices for missing value imputation?

# **Types of missing data:**
# 
# - **Missing Completely At Random (MCAR):**
#   - No relationship between missingness and underlying data
#   - Example: Booking agent forgot to gather Age
# - **Missing Not At Random (MNAR):**
#   - Relationship between missingness and underlying data
#   - Example: Older passengers declined to give their Age
# - **Missing due to a structural deficiency:**
#   - Data omitted for a specific purpose
#   - Example: Staff members did not pay a Fare

# **Advice for MCAR imputation:**
# 
# - **Small dataset:** IterativeImputer is more effective than mean imputation
# - **Large dataset:** IterativeImputer and mean imputation work equally well
# - No benefit to adding a missing indicator

# **Advice for MNAR imputation:**
# 
# - Mean imputation is more effective than IterativeImputer
# - Add a missing indicator
# - Use a powerful, non-linear model

# **Advice for structural deficiency imputation:**
# 
# - Impute a logical and reasonable user-defined value
# - Add a missing indicator

# **Advantages of histogram-based gradient boosting trees:**
# 
# - Built-in support for missing values
# - Lower computational cost than IterativeImputer
# - Performs well across many missing value scenarios

# ## 7.7 Q&A: What's the difference between ColumnTransformer and FeatureUnion?

imp_indicator = SimpleImputer(add_indicator=True)
imp_indicator.fit_transform(X[['Age']])


imp.fit_transform(X[['Age']])


from sklearn.impute import MissingIndicator
indicator = MissingIndicator()


indicator.fit_transform(X[['Age']])


from sklearn.pipeline import make_union
union = make_union(imp, indicator)


union.fit_transform(X[['Age']])


ct = make_column_transformer(
    (imp, ['Age']),
    (indicator, ['Age']))
ct.fit_transform(X)


# **FeatureUnion vs ColumnTransformer:**
# 
# - **FeatureUnion:**
#   - Single input column
#   - Applies multiple different transformations to that column in parallel
# - **ColumnTransformer:**
#   - Multiple input columns
#   - Applies a different transformation to each column in parallel

# # Chapter 8: Fixing common workflow problems

# ## 8.1 Two new problems

df = pd.read_csv('http://bit.ly/MLtrain')
df.shape


df_new = pd.read_csv('http://bit.ly/MLnewdata')
df_new.shape


df.isna().sum()


df_new.isna().sum()


# **Features with missing values:**
# 
# - Problematic:
#   - **Embarked:** Missing values in df
#   - **Fare:** Missing value in df_new
# - Not problematic:
#   - **Cabin:** Not currently using
#   - **Age:** Already being imputed

# ## 8.2 Problem 1: Missing values in a categorical feature

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']


X = df[cols]
y = df['Survived']


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    ('passthrough', ['Parch', 'Fare']))


ct.fit_transform(X)


# **How OneHotEncoder handles missing values:**
# 
# - **Before version 0.24:** Errors if the input contains missing values
# - **Starting in version 0.24:** Treats missing values as a new category

# **Imputation strategies for categorical features:**
# 
# - Most frequent value
# - User-defined value

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')


imp_ohe = make_pipeline(imp_constant, ohe)


imp_ohe.fit_transform(X[['Embarked']])


imp_ohe[1].categories_


ohe.fit_transform(imp_constant.fit_transform(X[['Embarked']]))


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    ('passthrough', ['Parch', 'Fare']))


# **Notes about the imp_ohe Pipeline:**
# 
# - Treated like a transformer because all of its steps are transformers
# - Imputation step won't affect the Sex column

ct.fit_transform(X)


# ## 8.3 Problem 2: Missing values in the new data

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))


ct.fit_transform(X)


# **What will be imputed for Fare?**
# 
# - **X:** No missing Fare values, thus no imputation of Fare
# - **X_new:** Missing Fare value, thus **impute the mean of Fare in X** during prediction

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)


X_new = df_new[cols]
pipe.predict(X_new)


# ## 8.4 Q&A: How do I see the feature names output by the ColumnTransformer?

ct.fit_transform(X)


ct.get_feature_names()


# **Changes to get_feature_names:**
# 
# - **Starting in version 1.0:** get_feature_names replaced with get_feature_names_out
# - **Starting in version 1.1:** get_feature_names_out available for all transformers

ct.transformers_


ct.named_transformers_['pipeline'].named_steps['onehotencoder'].get_feature_names()


len(ct.named_transformers_['countvectorizer'].get_feature_names())


# **Features output by each transformer:**
# 
# - **Pipeline:** 6 features (Embarked and Sex)
# - **CountVectorizer:** 1509 features (Name)
# - **SimpleImputer:** 2 features (Age and Fare)
# - **passthrough:** 1 feature (Parch)

# ## 8.5 Q&A: Why did we create a Pipeline inside of the ColumnTransformer?

imp_ohe = make_pipeline(imp_constant, ohe)


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']))


ct.fit_transform(X)


ct = make_column_transformer(
    (imp_ohe, ['Embarked']),
    (ohe, ['Sex']))


ct.fit_transform(X)


ct = make_column_transformer(
    (imp_constant, ['Embarked']),
    (ohe, ['Embarked', 'Sex']))


ct.fit_transform(X)


# **Pipeline vs ColumnTransformer:**
# 
# - **Pipeline:**
#   - Output of one step becomes the input to the next step
#   - **imp_ohe:** Output of **imp_constant** becomes the input to **ohe**
# - **ColumnTransformer:**
#   - Transformers operate in parallel
#   - **ct:** Output of each transformer is stacked beside the other transformer outputs

# ## 8.6 Q&A: Which imputation strategy should I use with categorical features?

# **Imputation strategies for categorical features:**
# 
# - **Constant user-defined value:**
#   - Treats missing values as a new category (recommended)
#   - Important if the majority of values are missing
# - **Most frequent value:**
#   - Acceptable if only a small number of values are missing

# **Possible problem with imputing a constant value:**
# 
# - **Condition:** The feature only has missing values in the new data
# - **Solution:** Set handle_unknown to 'ignore' for the OneHotEncoder
# - **Alternative:** Impute the most frequent value, and leave handle_unknown set to 'error'

# ## 8.7 Q&A: Should I impute missing values before all other transformations?

pipe


# **Impute missing values as a first step?**
# 
# - **Current Pipeline:**
#   - **Step 1:** All data transformations
#   - **Step 2:** Model
# - **Alternative Pipeline:**
#   - **Step 1:** Missing value imputation
#   - **Step 2:** All other data transformations
#   - **Step 3:** Model

ct1 = make_column_transformer(
    (imp_constant, ['Embarked']),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Sex', 'Name', 'Parch']))


ct2 = make_column_transformer(
    (ohe, [0, 3]),
    (vect, 4),
    ('passthrough', [1, 2, 5]))


pipe = make_pipeline(ct1, ct2, logreg)
pipe.fit(X, y)


pipe.predict(X_new)


# ## 8.8 Q&A: What methods can I use with a Pipeline?

# **Rules for Pipeline steps:**
# 
# - All steps other than the final step must be a **transformer**
# - Final step can be a **model** or a **transformer**

# **Pipeline ends in a model:**
# 
# - **fit:**
#   - All steps before the final step run **fit_transform**
#   - Final step runs **fit**
# - **predict:**
#   - All steps before the final step run **transform**
#   - Final step runs **predict**

# **Pipeline ends in a transformer:**
# 
# - **fit_transform:**
#   - All steps run **fit_transform**
# - **transform:**
#   - All steps run **transform**
# - **fit:**
#   - All steps before the final step run **fit_transform**
#   - Final step runs **fit**

# # Chapter 9: Workflow review #2

# ## 9.1 Recap of our workflow

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']


df = pd.read_csv('http://bit.ly/MLtrain')
X = df[cols]
y = df['Survived']


df_new = pd.read_csv('http://bit.ly/MLnewdata')
X_new = df_new[cols]


imp = SimpleImputer()
imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
ohe = OneHotEncoder()
vect = CountVectorizer()


imp_ohe = make_pipeline(imp_constant, ohe)


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))


logreg = LogisticRegression(solver='liblinear', random_state=1)


pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)


# ## 9.2 Comparing ColumnTransformer and Pipeline

# <img src="https://www.dataschool.io/files/complex_pipeline.png" width="625">

# **ColumnTransformer vs Pipeline:**
# 
# - **ColumnTransformer:**
#   - Selects subsets of columns, transforms them independently, stacks the results side-by-side
#   - Only includes transformers
#   - Does not have steps (transformers operate in parallel)
# - **Pipeline:**
#   - Series of steps that occur in order
#   - Output of each step becomes the input to the next step
#   - Last step is a model or transformer, all other steps are transformers

pipe


# ## 9.3 Why not use pandas for transformations?

# &nbsp; | scikit-learn | pandas
# :--- | :--- | :---
# **Encoding text data** | CountVectorizer | Not available
# **One-hot encoding** | OneHotEncoder | get_dummies (results in larger DataFrame)
# **Imputing missing values** | SimpleImputer & other imputers | fillna (results in data leakage)
# **Cross-validation and tuning** | Your entire Pipeline | Just your model

# ## 9.4 Preventing data leakage

# **What is data leakage?**
# 
# - Inadvertently including knowledge from the testing data when training a model

# **Why is data leakage problematic?**
# 
# - Your model evaluation scores will be less reliable
# - You might make bad decisions when tuning hyperparameters
# - You will overestimate how well your model will perform on new data

# **Imputation on the full dataset can cause data leakage:**
# 
# - Your model evaluation procedure is supposed to simulate the future
# - Imputation based on your full dataset "leaks" information about the future into model training

# **Other transformations on the full dataset can also cause data leakage:**
# 
# - Feature scaling
# - One-hot encoding
# - Any transformation which incorporates information about other rows

# **How does scikit-learn prevent data leakage?**
# 
# - Transformers have separate fit and transform steps
# - Pipeline methods call fit_transform and transform at the appropriate times
# - cross_val_score splits the data prior to performing transformations
