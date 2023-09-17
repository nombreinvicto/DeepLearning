#!/usr/bin/env python
# coding: utf-8

# # Course: [Master Machine Learning with scikit-learn](https://courses.dataschool.io/view/courses/master-machine-learning-with-scikit-learn)
# 
# ## Chapters 1-5
# 
# *© 2022 Data School. All rights reserved.*

# # Chapter 1: Introduction

# ## 1.1 Course overview

# **High-level topics:**
# 
# - Handling missing values, text data, categorical data, and class imbalance
# - Building a reusable workflow
# - Feature engineering, selection, and standardization
# - Avoiding data leakage
# - Tuning your entire workflow

# **How you will benefit from this course:**
# 
# - Knowledge of best practices
# - Confidence when tackling new ML problems
# - Ability to anticipate and solve problems
# - Improved code quality
# - Better, faster results

# ## 1.2 scikit-learn vs Deep Learning

# **Benefits of scikit-learn:**
# 
# - Consistent interface to many models
# - Many tuning parameters (but sensible defaults)
# - Workflow-related functionality
# - Exceptional documentation
# - Active community support

# **Drawbacks of deep learning:**
# 
# - More computational resources
# - Higher learning curve
# - Less interpretable models

# ## 1.3 Prerequisite skills

# **scikit-learn prerequisites:**
# 
# - Loading a dataset
# - Defining the features and target
# - Training and evaluating a model
# - Making predictions with new data

# **New to scikit-learn?**
# 
# - Enroll in "Introduction to Machine Learning with scikit-learn" (free)
# - Available at https://courses.dataschool.io
# - Complete lessons 1 through 7

# ## 1.4 Course setup and software versions

# **How to install scikit-learn and pandas:**
# 
# - **Option 1:** Install together
#   - **Anaconda:** https://www.anaconda.com/products/distribution
# - **Option 2:** Install separately
#   - **scikit-learn:** https://scikit-learn.org
#   - **pandas:** https://pandas.pydata.org

import sklearn
sklearn.__version__


# **scikit-learn version:**
# 
# - **Course version:** 0.23.2
# - **Minimum version:** 0.20.2

# **How to install scikit-learn 0.23.2:**
# 
# - **Option 1:** conda install scikit-learn==0.23.2
# - **Option 2:** pip install -U scikit-learn==0.23.2

import pandas
pandas.__version__


# **Using Google Colab with the course:**
# 
# - Similar to the Jupyter Notebook
# - Runs in your browser
# - Free (but requires a Google account)
# - Available at https://colab.research.google.com

# ## 1.5 Course outline

# **Chapters:**
# 
# 1. Introduction
# 2. Review of the Machine Learning workflow
# 3. Encoding categorical features
# 4. Improving your workflow with ColumnTransformer and Pipeline
# 5. Workflow review #1
# 6. Encoding text data
# 7. Handling missing values
# 8. Fixing common workflow problems
# 9. Workflow review #2
# 10. Evaluating and tuning a Pipeline
# 11. Comparing linear and non-linear models
# 12. Ensembling multiple models
# 13. Feature selection
# 14. Feature standardization
# 15. Feature engineering with custom transformers
# 16. Workflow review #3
# 17. High-cardinality categorical features
# 18. Class imbalance
# 19. Class imbalance walkthrough
# 20. Going further

# **Lesson types:**
# 
# - Core lessons
# - Q&A lessons

# **Why not focus on algorithms?**
# 
# - Workflow will have a greater impact on your results
# - Reusable workflow enables you to try many different algorithms
# - Hard to know (in advance) which algorithm will work best

# ## 1.6 Course datasets

# **Datasets:**
# 
# - Titanic
# - US census
# - Mammography scans

# **Why use smaller datasets?**
# 
# - Easier and faster access to files
# - Reduced computational time
# - Greater understanding of the course material

# ## 1.7 Meet your instructor

# **About me:**
# 
# - Founder of Data School
# - Teaching data science for 7+ years
# - Passionate about teaching people who are new to data science
# - Live in Asheville, North Carolina
# - Degree in Computer Engineering

# # Chapter 2: Review of the Machine Learning workflow

# ## 2.1 Loading and exploring a dataset

import pandas as pd
df = pd.read_csv('http://bit.ly/MLtrain', nrows=10)
df = pd.read_csv('titanic_train.csv', nrows=10)


df


# **Machine Learning terminology:**
# 
# - **Target:** Goal of prediction
# - **Classification:** Problem with a categorical target
# - **Feature:** Input to the model (column)
# - **Sample:** Single observation (row)
# - **Training data:** Data with known target values

# **Feature selection methods:**
# 
# - Human intuition
# - Domain knowledge
# - Data exploration
# - Automated methods

# **Currently selected features:**
# 
# - **Parch:** Number of parents or children aboard with that passenger
# - **Fare:** Amount the passenger paid

X = df[['Parch', 'Fare']]
X


y = df['Survived']
y


X.shape


y.shape


# ## 2.2 Building and evaluating a model

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear', random_state=1)


# **Requirements for model evaluation:**
# 
# - **Procedure:** K-fold cross-validation
# - **Metric:** Classification accuracy

# **Steps of 3-fold cross-validation:**
# 
# 1. Split rows into 3 subsets (A, B, C)
# 2. A & B is training set, C is testing set
#   - Train model on training set
#   - Make predictions on testing set
#   - Evaluate predictions
# 3. Repeat with A & C as training set, B as testing set
# 4. Repeat with B & C as training set, A as testing set
# 5. Calculate the mean of the scores

from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean()


# ## 2.3 Using the model to make predictions

# **Ways to improve the model:**
# 
# - Hyperparameter tuning
# - Adding or removing features
# - Trying a different model

logreg.fit(X, y)


# **Important points about model fitting:**
# 
# - Train your model on the entire dataset before making predictions
# - Assignment statement is unnecessary
# - Passing pandas objects is fine
# - Only prints parameters that have changed (version 0.23 or later)

df_new = pd.read_csv('http://bit.ly/MLnewdata', nrows=10)
df_new = pd.read_csv('titanic_new.csv', nrows=10)
df_new


X_new = df_new[['Parch', 'Fare']]
X_new


logreg.predict(X_new)


# ## 2.4 Q&A: How do I adapt this workflow to a regression problem?

# **Adapting this workflow for regression:**
# 
# 1. Choose a different model
# 2. Choose a different evaluation metric

# ## 2.5 Q&A: How do I adapt this workflow to a multiclass problem?

# **Types of classification problems:**
# 
# - **Binary:** Two output classes
# - **Multiclass:** More than two output classes

# **How classifiers handle multiclass problems:**
# 
# - Many are inherently multiclass
# - Others can be extended using "one-vs-one" or "one-vs-rest" strategies

# ## 2.6 Q&A: Why should I select a Series for the target?

df['Survived']


df[['Survived']]


df['Survived'].shape


df[['Survived']].shape


df['Survived'].to_numpy()


df[['Survived']].to_numpy()


# **Multilabel vs multiclass problems:**
# 
# - **Multilabel:** Each sample can have more than one label
# - **Multiclass:** Each sample can have one label

# **Multilabel vs multiclass targets:**
# 
# - **Multilabel:** 2-dimensional y (DataFrame)
# - **Multiclass:** 1-dimensional y (Series)

# ## 2.7 Q&A: How do I add the model's predictions to a DataFrame?

predictions = pd.Series(logreg.predict(X_new), index=X_new.index,
                        name='Prediction')


pd.concat([X_new, predictions], axis='columns')


# ## 2.8 Q&A: How do I determine the confidence level of each prediction?

logreg.predict(X_new)


logreg.predict_proba(X_new)


# **Array of predicted probabilities:**
# 
# - One row for each sample
# - One column for each class

logreg.predict_proba(X_new)[:, 1]


# ## 2.9 Q&A: How do I check the accuracy of the model's predictions?

# **Checking model accuracy:**
# 
# - **Not possible:** Target value is unknown or is private data
# - **Possible:** Target value is known

# ## 2.10 Q&A: What do the "solver" and "random_state" parameters do?

logreg = LogisticRegression(solver='liblinear', random_state=1)


# <img src="https://www.dataschool.io/files/solver_comparison.png">

# **Default solver for logistic regression:**
# 
# - **Before version 0.22:** liblinear
# - **Starting in version 0.22:** lbfgs

# **Advice for random_state:**
# 
# - Set random_state to any integer when a random process is involved
# - Allows your code to be reproducible

# ## 2.11 Q&A: How do I show all of the model parameters?

logreg


logreg.get_params()


from sklearn import set_config
set_config(print_changed_only=False)


logreg


set_config(print_changed_only=True)


# ## 2.12 Q&A: Should I shuffle the samples when using cross-validation?

cross_val_score(logreg, X, y, cv=3, scoring='accuracy')


from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(3)
cross_val_score(logreg, X, y, cv=kf, scoring='accuracy')


# **Stratified sampling:**
# 
# - Ensures that each fold is representative of the dataset
# - Produces more reliable cross-validation scores

kf = StratifiedKFold(3, shuffle=True, random_state=1)
cross_val_score(logreg, X, y, cv=kf, scoring='accuracy')


# **When to shuffle your samples:**
# 
# - **Samples in arbitrary order:** Shuffling not needed
# - **Samples are ordered:** Shuffling needed

# **How to shuffle your samples:**
# 
# - **Classification:** StratifiedKFold
# - **Regression:** KFold

# # Chapter 3: Encoding categorical features

# ## 3.1 Introduction to one-hot encoding 

# **How to run the code above:**
# 
# - **Jupyter Notebook:**
#   - Select this cell
#   - Click "Cell" menu, then "Run All Above"
# - **JupyterLab:**
#   - Select this cell
#   - Click "Run" menu, then "Run All Above Selected Cell"

df


# **Currently selected features:**
# 
# - **Parch:** Number of parents or children aboard with that passenger
# - **Fare:** Amount the passenger paid
# - **Embarked:** Port the passenger embarked from
# - **Sex:** Male or Female

# **Unordered categorical data:**
# 
# - Contains distinct categories
# - No inherent logical ordering to the categories
# - Also called "nominal data"

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)


# **Matrix representations:**
# 
# - **Sparse:** More efficient and performant
# - **Dense:** More readable

# **Why use double brackets?**
# 
# - **Single brackets:**
#   - Outputs a Series
#   - Could be interpreted as a single feature or a single sample
# - **Double brackets:**
#   - Outputs a single-column DataFrame
#   - Interpreted as a single feature

ohe.fit_transform(df[['Embarked']])


# **Output of OneHotEncoder:**
# 
# - One column for each unique value
# - One non-zero value in each row:
#   - 1, 0, 0 means "C"
#   - 0, 1, 0 means "Q"
#   - 0, 0, 1 means "S"

ohe.categories_


# **Why use one-hot encoding?**
# 
# - Model can learn the relationship between each level and the target value
# - Example: Model might learn that "C" passengers have a higher survival rate than "not C" passengers

# **Why not encode as a single feature?**
# 
# - **Pretend:**
#   - C: high survival rate
#   - Q: low survival rate
#   - S: high survival rate
# - **Single feature would need two coefficients:**
#   - Negative coefficient for impact of Q (with respect to C)
#   - Positive coefficient for impact of S (with respect to Q)

# ## 3.2 Transformer methods: fit, transform, fit_transform

# **Generic transformer methods:**
# 
# - **fit:** Transformer learns something
# - **transform:** Transformer uses what it learned to do the data transformation

# **OneHotEncoder methods:**
# 
# - **fit:** Learn the categories
# - **transform:** Create the feature matrix using those categories

# ## 3.3 One-hot encoding of multiple features

ohe.fit_transform(df[['Embarked', 'Sex']])


ohe.categories_


# **Decoding the output array:**
# 
# - **First three columns:**
#   - 1, 0, 0 means "C"
#   - 0, 1, 0 means "Q"
#   - 0, 0, 1 means "S"
# - **Last two columns:**
#   - 1, 0 means "female"
#   - 0, 1 means "male"
# - **Example:**
#   - 0, 0, 1, 0, 1 means "S, male"
#   - 1, 0, 0, 1, 0 means "C, female"

# **How to manually add Embarked and Sex to the model:**
# 
# 1. Stack Parch and Fare side-by-side with OneHotEncoder output
# 2. Repeat the same process with new data

# **Problems with a manual approach:**
# 
# - Repeating steps is inefficient and error-prone
# - Complexity will increase

# ## 3.4 Q&A: When should I use transform instead of fit_transform?

demo_train = pd.DataFrame({'letter':['A', 'B', 'C', 'B']})
demo_train


ohe.fit_transform(demo_train)


# **Example of fit_transform on training data:**
# 
# - **fit:** Learn 3 categories (A, B, C)
# - **transform:** Create feature matrix with 3 columns

demo_test = pd.DataFrame({'letter':['A', 'C', 'A']})
demo_test


ohe.fit_transform(demo_test)


# **Example of fit_transform on testing data:**
# 
# - **fit:** Learn 2 categories (A, C)
# - **transform:** Create feature matrix with 2 columns

ohe.fit_transform(demo_train)


ohe.transform(demo_test)


# **Correct process:**
# 
# 1. Run fit_transform on training data:
#   - **fit:** Learn 3 categories (A, B, C)
#   - **transform:** Create feature matrix with 3 columns
# 2. Run transform on testing data:
#   - **transform:** Create feature matrix with 3 columns

# ## 3.5 Q&A: What happens if the testing data includes a new category?

demo_train


ohe.fit_transform(demo_train)


ohe.categories_


demo_test_unknown = pd.DataFrame({'letter':['A', 'C', 'D']})
demo_test_unknown


ohe.transform(demo_test_unknown)


ohe = OneHotEncoder(sparse=False, categories=[['A', 'B', 'C', 'D']])


ohe.fit_transform(demo_train)


ohe.transform(demo_test_unknown)


# **Why you might not know all possible categories:**
# 
# - Rare categories aren't present in your set of samples
# - New categories are added later

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')


ohe.fit_transform(demo_train)


ohe.transform(demo_test_unknown)


# **Advice for OneHotEncoder:**
# 
# 1. Start with handle_unknown set to 'error'
# 2. If possible, specify the categories manually
# 3. If necessary, set handle_unknown to 'ignore' and then retrain your model

# ## 3.6 Q&A: Should I drop one of the one-hot encoded categories?

demo_train


ohe.fit_transform(demo_train)


# **You can drop the first column:**
# 
# - Contains redundant information
# - Avoids collinearity between features

ohe = OneHotEncoder(sparse=False, drop='first')
ohe.fit_transform(demo_train)


# **Decoding the output array (after dropping the first column):**
# 
# - 0, 0 means "A"
# - 1, 0 means "B"
# - 0, 1 means "C"

# **Should you drop the first column?**
# 
# - **Advantages:**
#   - Useful if perfectly collinear features will cause problems (does not apply to most models)
# - **Disadvantages:**
#   - Incompatible with handle_unknown='ignore'
#   - Introduces bias if you standardize features or use a regularized model

# ## 3.7 Q&A: How do I encode an ordinal feature?

# **Types of categorical data:**
# 
# - Unordered (nominal data)
# - Ordered (ordinal data)

df


# **Options for encoding Pclass:**
# 
# - **Ordinal encoding:** Creates one feature
# - **One-hot encoding:** Creates three features

df_ordinal = pd.DataFrame({'Class': ['third', 'first', 'second', 'third'],
                           'Size': ['S', 'S', 'L', 'XL']})
df_ordinal


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['first', 'second', 'third'],
                                ['S', 'M', 'L', 'XL']])


oe.fit_transform(df_ordinal)


# **Decoding the output array:**
# 
# - **First column:**
#   - 0 means "first"
#   - 1 means "second"
#   - 2 means "third"
# - **Second column:**
#   - 0 means "S"
#   - 1 means "M"
#   - 2 means "L"
#   - 3 means "XL"
# - **Example:**
#   - 2, 0 means "third, S"

ohe = OneHotEncoder(sparse=False, categories=[['first', 'second', 'third'],
                                              ['S', 'M', 'L', 'XL']])
ohe.fit_transform(df_ordinal)


# **Advice for encoding categorical data:**
# 
# - **Ordinal feature stored as numbers:** Leave as-is
# - **Ordinal feature stored as strings:** Use OrdinalEncoder
# - **Nominal feature:** Use OneHotEncoder

# ## 3.8 Q&A: What's the difference between OrdinalEncoder and LabelEncoder?

# &nbsp; | OrdinalEncoder | LabelEncoder
# :--- | :---: | :---:
# Can you define the category order? | Yes | No
# Can you encode multiple features? | Yes | No

# **Outdated uses for LabelEncoder:**
# 
# - Encoding string-based labels for some classifiers
# - Encoding string-based features for OneHotEncoder

# ## 3.9 Q&A: Should I encode numeric features as ordinal features?

df[['Fare']]


from sklearn.preprocessing import KBinsDiscretizer


kb = KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')


kb.fit_transform(df[['Fare']])


# **Why not discretize numeric features?**
# 
# - Makes it harder to learn the actual trends
# - Makes it easier to discover non-existent trends
# - May result in overfitting

# # Chapter 4: Improving your workflow with ColumnTransformer and Pipeline

# ## 4.1 Preprocessing features with ColumnTransformer

# **Problems from Chapter 3:**
# 
# - Need to stack categorical features next to numerical features
# - Need to apply the same preprocessing to new data

# **How to solve those problems:**
# 
# - **ColumnTransformer:** Apply different preprocessing steps to different columns
# - **Pipeline:** Apply the same workflow to training data and new data

cols = ['Parch', 'Fare', 'Embarked', 'Sex']


X = df[cols]
X


ohe = OneHotEncoder()


from sklearn.compose import make_column_transformer
ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    remainder='drop')


# **Tuple elements for make_column_transformer:**
# 
# 1. Transformer object
# 2. List of columns to which the transformer should be applied

ct.fit_transform(X)


# **Output columns:**
# 
# - **Columns 1-3:** Embarked
# - **Columns 4-5:** Sex

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    remainder='passthrough')


ct.fit_transform(X)


# **Output columns:**
# 
# - **Columns 1-3:** Embarked
# - **Columns 4-5:** Sex
# - **Column 6:** Parch
# - **Column 7:** Fare

ct.get_feature_names()


# **Notes about get_feature_names:**
# 
# - **Before version 0.23:** Didn't work with passthrough columns
# - **Starting in version 1.0:** Has been replaced with get_feature_names_out

# **Tuple elements for make_column_transformer (revised):**
# 
# 1. Transformer object or "drop" or "passthrough"
# 2. List of columns to which the transformer should be applied

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    ('passthrough', ['Parch', 'Fare']))


ct.fit_transform(X)


# ## 4.2 Chaining steps with Pipeline

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(ct, logreg)


# **Pipeline steps:**
# 
# 1. Data preprocessing with ColumnTransformer
# 2. Model building with LogisticRegression

pipe.fit(X, y)


# **Fitting the Pipeline:**
# 
# 1. ColumnTransformer converts X (4 columns) into a numeric feature matrix (7 columns)
# 2. LogisticRegression model is fit to the feature matrix

X_t = ct.fit_transform(X)
logreg.fit(X_t, y)


print(X.shape)
print(X_t.shape)


# ## 4.3 Using the Pipeline to make predictions

X_new = df_new[cols]
X_new


pipe.predict(X_new)


# **Predicting with the Pipeline:**
# 
# 1. ColumnTransformer applies the same transformations to X_new
# 2. Fitted LogisticRegression model makes predictions on the transformed version of X_new

X_new_t = ct.transform(X_new)
logreg.predict(X_new_t)


print(X_new.shape)
print(X_new_t.shape)


# **ColumnTransformer methods:**
# 
# 1. Run fit_transform on X:
#   - **fit:** Learn the encoding
#   - **transform:** Apply the encoding to create 7 columns
# 2. Run transform on X_new:
#   - **transform:** Apply the encoding to create 7 columns

# ## 4.4 Q&A: How do I drop some columns and passthrough others?

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    ('drop', ['Fare']),
    remainder='passthrough')
ct.fit_transform(X)


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    ('passthrough', ['Parch']),
    remainder='drop')
ct.fit_transform(X)


# ## 4.5 Q&A: How do I transform the unspecified columns?

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    ('drop', ['Fare']),
    remainder=scaler)
ct.fit_transform(X)


# ## 4.6 Q&A: How do I select columns from a NumPy array?

X_array = X.to_numpy()
X_new_array = X_new.to_numpy()


X_array


ct = make_column_transformer(
    (ohe, [2, 3]),
    remainder='passthrough')
ct.fit_transform(X_array)


ct = make_column_transformer(
    (ohe, slice(2, 4)),
    remainder='passthrough')
ct.fit_transform(X_array)


ct = make_column_transformer(
    (ohe, [False, False, True, True]),
    remainder='passthrough')
ct.fit_transform(X_array)


# **Options for selecting columns from a NumPy array:**
# 
# - Integer position
# - Slice
# - Boolean mask

pipe = make_pipeline(ct, logreg)


pipe.fit(X_array, y)
pipe.predict(X_new_array)


# ## 4.7 Q&A: How do I select columns by data type?

from sklearn.compose import make_column_selector


select_object = make_column_selector(dtype_include=object)
select_number = make_column_selector(dtype_include='number')


ct = make_column_transformer(
    (ohe, select_object),
    ('passthrough', select_number))
ct.fit_transform(X)


exclude_object = make_column_selector(dtype_exclude=object)


ct = make_column_transformer(
    (ohe, select_object),
    ('passthrough', exclude_object))
ct.fit_transform(X)


select_datetime = make_column_selector(dtype_include='datetime')
select_category = make_column_selector(dtype_include='category')


select_multiple = make_column_selector(dtype_include=[object, 'category'])


# ## 4.8 Q&A: How do I select columns by column name pattern?

select_ES = make_column_selector(pattern='E|S')


ct = make_column_transformer(
    (ohe, select_ES),
    remainder='passthrough')
ct.fit_transform(X)


# ## 4.9 Q&A: Should I use ColumnTransformer or make_column_transformer?

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('OHE', ohe, ['Embarked', 'Sex']),
     ('pass', 'passthrough', ['Parch', 'Fare'])])
ct


# **Tuple elements for ColumnTransformer:**
# 
# 1. Transformer name
# 2. Transformer object or "drop" or "passthrough"
# 3. List of columns to which the transformer should be applied

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    ('passthrough', ['Parch', 'Fare']))
ct


# &nbsp; | ColumnTransformer | make_column_transformer
# :--- | :---: | :---:
# Allows custom names? | Yes | No
# Allows transformer weights? | Yes | No

# ## 4.10 Q&A: Should I use Pipeline or make_pipeline?

from sklearn.pipeline import Pipeline
pipe = Pipeline([('preprocessor', ct), ('classifier', logreg)])
pipe


# **Tuple elements for Pipeline:**
# 
# 1. Step name
# 2. Model or transformer object

pipe.named_steps.keys()


pipe = make_pipeline(ct, logreg)
pipe


pipe.named_steps.keys()


# &nbsp; | Pipeline | make_pipeline
# :--- | :---: | :---:
# Allows custom names? | Yes | No

# ## 4.11 Q&A: How do I examine the steps of a Pipeline?

pipe.fit(X, y)


pipe.named_steps.keys()


pipe.named_steps['columntransformer']


pipe.named_steps['logisticregression']


pipe.named_steps['columntransformer'].get_feature_names()


pipe.named_steps['logisticregression'].coef_


pipe.named_steps.logisticregression.coef_


pipe['logisticregression'].coef_


pipe[1].coef_


# # Chapter 5: Workflow review #1

# ## 5.1 Recap of our workflow

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


cols = ['Parch', 'Fare', 'Embarked', 'Sex']


df = pd.read_csv('http://bit.ly/MLtrain', nrows=10)
X = df[cols]
y = df['Survived']


df_new = pd.read_csv('http://bit.ly/MLnewdata', nrows=10)
X_new = df_new[cols]


ohe = OneHotEncoder()


ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    ('passthrough', ['Parch', 'Fare']))


logreg = LogisticRegression(solver='liblinear', random_state=1)


pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)


# ## 5.2 Comparing ColumnTransformer and Pipeline

# <img src="https://www.dataschool.io/files/simple_pipeline.png" width="400">

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

# ## 5.3 Creating a Pipeline diagram

from sklearn import set_config
set_config(display='diagram')


pipe


print(pipe)


set_config(display='text')
pipe

