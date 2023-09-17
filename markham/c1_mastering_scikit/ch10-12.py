#!/usr/bin/env python
# coding: utf-8

# # Course: [Master Machine Learning with scikit-learn](https://courses.dataschool.io/view/courses/master-machine-learning-with-scikit-learn)
# 
# ## Chapters 10-12
# 
# *© 2023 Data School. All rights reserved.*

# # Chapter 10: Evaluating and tuning a Pipeline

# ## 10.1 Evaluating a Pipeline with cross-validation

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import set_config


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
pipe.fit(X, y);


set_config(display='diagram')


from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Baseline (no tuning):** 0.811

# **Steps of 5-fold cross-validation on a Pipeline:**
# 
# 1. Split data into 5 folds (A, B, C, D, E)
#   - ABCD is training set
#   - E is testing set
# 2. Pipeline is fit on training set
#   - ABCD is transformed
#   - Model is fit on transformed data
# 3. Pipeline makes predictions on testing set
#   - E is transformed (using step 2 transformations)
#   - Model makes predictions on transformed data
# 4. Calculate accuracy of those predictions
# 5. Repeat the steps above 4 more times, with a different testing set each time
# 6. Calculate the mean of the 5 scores

# **Why does cross_val_score split the data first?**
# 
# - **Proper cross-validation:**
#   - Data is split (step 1) before transformations (steps 2 and 3)
#   - Imputation values and vocabulary are computed using training set only
#   - Prevents data leakage
# - **Improper cross-validation:**
#   - Transformations are performed before data is split
#   - Imputation values and vocabulary are computed using full dataset
#   - Causes data leakage

# ## 10.2 Tuning a Pipeline with grid search

# **Statistics terminology:**
# 
# - **Hyperparameters:** Values that you set
#   - **Example:** C value of logistic regression
# - **Parameters:** Values learned from the data
#   - **Example:** Coefficients of logistic regression model

# **scikit-learn terminology:**
# 
# - **Hyperparameter tuning:** Tuning a model or a Pipeline
# - **Parameter:** Anything passed to a class
#   - **LogisticRegression:** C, random_state
#   - **SimpleImputer:** strategy

# **Hyperparameter tuning with GridSearchCV:**
# 
# - You define which values to try for each parameter
# - It cross-validates every combination of those values

# **Benefits of tuning a Pipeline:**
# 
# - Tunes the model and transformers simultaneously
# - Prevents data leakage

# ## 10.3 Tuning the model

# **LogisticRegression tuning parameters:**
# 
# - **penalty:** Type of regularization
#   - 'l1'
#   - 'l2' (default)
# - **C:** Amount of regularization
#   - 0.1
#   - 1 (default)
#   - 10

pipe.named_steps.keys()


# **Parameter dictionary for GridSearchCV:**
# 
# - **Key:** step__parameter
#   - 'logisticregression__penalty'
#   - 'logisticregression__C'
# - **Value:** List of values to try
#   - ['l1', 'l2']
#   - [0.1, 1, 10]

params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
params


from sklearn.model_selection import GridSearchCV


grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)


results = pd.DataFrame(grid.cv_results_)
results


results.sort_values('rank_test_score')


# **Pipeline accuracy scores:**
# 
# - **Grid search (2 parameters):** 0.818 👈
# - **Baseline (no tuning):** 0.811

# ## 10.4 Tuning the transformers

# **Options for expanding the grid search:**
# 
# - **Initial idea:** Set C=10 and penalty='l1', then only search transformer parameters
# - **Better approach:** Search for best combination of C, penalty, and transformer parameters

pipe.named_steps['columntransformer'].named_transformers_


# **OneHotEncoder tuning parameter:**
# 
# - **drop:** Method for dropping a column of each feature
#   - None (default)
#   - 'first'

params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']


list(pipe.get_params().keys())


# **CountVectorizer tuning parameter:**
# 
# - **ngram_range:** Selection of word n-grams to be extracted as features
#   - (1, 1) (default)
#   - (1, 2)

params['columntransformer__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]


# **SimpleImputer tuning parameter:**
# 
# - **add_indicator:** Option to add a missing indicator column
#   - False (default)
#   - True

params['columntransformer__simpleimputer__add_indicator'] = [False, True]


params


grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)


results = pd.DataFrame(grid.cv_results_)
results.sort_values('rank_test_score')


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828 👈
# - **Grid search (2 parameters):** 0.818
# - **Baseline (no tuning):** 0.811

grid.best_score_


grid.best_params_


# ## 10.5 Using the best Pipeline to make predictions

type(grid.best_estimator_)


grid.best_estimator_


grid.predict(X_new)


# ## 10.6 Q&A: How do I save the best Pipeline for future use?

type(grid.best_estimator_)


import pickle


with open('pipe.pickle', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)


with open('pipe.pickle', 'rb') as f:
    pipe_from_pickle = pickle.load(f)


pipe_from_pickle.predict(X_new)


import joblib


joblib.dump(grid.best_estimator_, 'pipe.joblib')


pipe_from_joblib = joblib.load('pipe.joblib')


pipe_from_joblib.predict(X_new)


# **Warnings for pickle and joblib objects:**
# 
# - May be version-specific and architecture-specific
# - Can be poisoned with malicious code

# **Alternatives to pickle and joblib:**
# 
# - Examples: ONNX, PMML
# - Save a model representation for making predictions
# - Work across environments and architectures

# ## 10.7 Q&A: How do I speed up a grid search?

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy', verbose=1)
grid.fit(X, y)


grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X, y)


grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)


# ## 10.8 Q&A: How do I tune a Pipeline with randomized search?

params


more_params = params.copy()
more_params['logisticregression__C'] = [0.01, 0.1, 1, 10, 100, 1000]


# **How to use RandomizedSearchCV:**
# 
# - **n_iter:** Specify the number of randomly-chosen parameter combinations to cross-validate
# - **random_state:** Set to any integer for reproducibility

from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(pipe, more_params, cv=5, scoring='accuracy', n_iter=10,
                          random_state=1, n_jobs=-1)
rand.fit(X, y)


pd.DataFrame(rand.cv_results_)


rand.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828
# - **Randomized search (more C values):** 0.827 👈
# - **Grid search (2 parameters):** 0.818
# - **Baseline (no tuning):** 0.811

rand.best_params_


# **Why use RandomizedSearchCV instead of GridSearchCV?**
# 
# - Similar results in far less time
# - Easier to control the computational budget
# - Freedom to tune many more parameters
# - Can use a much finer grid

import numpy as np
np.linspace(0, 1, 101)


np.logspace(-2, 3, 6)


# ## 10.9 Q&A: What's the target accuracy we are trying to achieve?

# **When is a model "good enough"?**
# 
# - **Useful model:** Outperforms null accuracy
# - **Best possible model:** Usually impossible to know the theoretical maximum accuracy
# - **Practical model:** Continue improving until you run out of resources

y.value_counts(normalize=True)


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828
# - **Randomized search (more C values):** 0.827
# - **Grid search (2 parameters):** 0.818
# - **Baseline (no tuning):** 0.811
# - **Null model:** 0.616 👈

# ## 10.10 Q&A: Is it okay that our model includes thousands of features?

pipe.named_steps['columntransformer']


pipe.named_steps['columntransformer'].fit_transform(X)


cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828
# - **Randomized search (more C values):** 0.827
# - **Grid search (2 parameters):** 0.818
# - **Baseline (no tuning):** 0.811 👈
# - **Null model:** 0.616

grid.best_estimator_.named_steps['columntransformer']


grid.best_estimator_.named_steps['columntransformer'].fit_transform(X)


grid.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828 👈
# - **Randomized search (more C values):** 0.827
# - **Grid search (2 parameters):** 0.818
# - **Baseline (no tuning):** 0.811
# - **Null model:** 0.616

no_name_ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))


no_name_ct.fit_transform(X).shape


no_name_pipe = make_pipeline(no_name_ct, logreg)
cross_val_score(no_name_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828
# - **Randomized search (more C values):** 0.827
# - **Grid search (2 parameters):** 0.818
# - **Baseline (no tuning):** 0.811
# - **Baseline excluding Name (no tuning):** 0.783 👈
# - **Null model:** 0.616

# **What did we learn?**
# 
# - Name column contains more predictive signal than noise
# - More features than samples does not necessarily result in overfitting

# ## 10.11 Q&A: How do I examine the coefficients of a Pipeline?

grid.best_estimator_.named_steps['logisticregression'].coef_


grid.best_estimator_.named_steps['columntransformer'].get_feature_names()


grid.best_estimator_.named_steps['columntransformer'].transformers_


# ## 10.12 Q&A: Should I split the dataset before tuning the Pipeline?

# **Goals of a grid search:**
# 
# - Choose the best parameters for the Pipeline
# - Estimate its performance on new data when using these parameters

grid.best_params_


grid.best_score_


# **Is it okay to use the same data for both goals?**
# 
# - **Yes:** If your main objective is to choose the best parameters
# - **No:** If you need a realistic estimate of performance on new data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=1, stratify=y)


training_grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
training_grid.fit(X_train, y_train)


training_grid.best_params_


training_grid.score(X_test, y_test)


# **Pipeline accuracy scores:**
# 
# - **Grid search (5 parameters):** 0.828
# - **Randomized search (more C values):** 0.827
# - **Grid search (2 parameters):** 0.818
# - **Grid search (estimate for new data):** 0.816 👈
# - **Baseline (no tuning):** 0.811
# - **Baseline excluding Name (no tuning):** 0.783
# - **Null model:** 0.616

best_pipe = training_grid.best_estimator_
best_pipe.fit(X, y)


best_pipe.predict(X_new)


# **Guidelines for using this process:**
# 
# - **Only use the testing set once:**
#   - If used multiple times, performance estimates will become less reliable
# - **You must have enough data:**
#   - If training set is too small, grid search won't find the optimal parameters
#   - If testing set is too small, it won't provide a reliable performance estimate

# ## 10.13 Q&A: What is regularization?

# **Brief explanation of regularization:**
# 
# - Constrains the size of model coefficients to minimize overfitting
# - Reduces the variance of an overly complex model to help the model generalize
# - Decreases model flexibility so that it follows the true patterns in the data

# # Chapter 11: Comparing linear and non-linear models

# ## 11.1 Trying a random forest model

# **Random forest model:**
# 
# - Non-linear model
# - Based on decision trees
# - Different properties from logistic regression

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1, n_jobs=-1)


rf_pipe = make_pipeline(ct, rf)
rf_pipe


cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (LR):** 0.828
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811 👈

# ## 11.2 Tuning random forests with randomized search

rf_params = params.copy()
rf_params


del rf_params['logisticregression__penalty']
del rf_params['logisticregression__C']
rf_params


rf_params = {k:v for k, v in params.items() if k.startswith('col')}
rf_params


# **Two-step approach to hyperparameter tuning:**
# 
# 1. **Randomized search:** Test a variety of parameters and values, then examine the results for trends
# 2. **Grid search:** Use an optimized set of parameters and values based on what you learned from step 1

rf_pipe.named_steps.keys()


# **RandomForestClassifier tuning parameters:**
# 
# - **n_estimators:** Number of decisions trees in the forest
#   - 100 (default)
#   - 300
#   - 500
#   - 700
# - **min_samples_leaf:** Minimum number of samples at a leaf node
#   - 1 (default)
#   - 2
#   - 3
# - **max_features:** Number of features to consider when choosing a split
#   - 'sqrt' (default)
#   - None
# - **bootstrap:** Whether bootstrap samples are used when building trees
#   - True (default)
#   - False

rf_params['randomforestclassifier__n_estimators'] = [100, 300, 500, 700]
rf_params['randomforestclassifier__min_samples_leaf'] = [1, 2, 3]
rf_params['randomforestclassifier__max_features'] = ['sqrt', None]
rf_params['randomforestclassifier__bootstrap'] = [True, False]
rf_params


# WARNING: 5 minutes
rf_rand = RandomizedSearchCV(rf_pipe, rf_params, cv=5, scoring='accuracy',
                             n_iter=100, random_state=1, n_jobs=-1)
rf_rand.fit(X, y)


rf_rand.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (LR):** 0.828
# - **Randomized search (RF):** 0.825 👈
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

# ## 11.3 Further tuning with grid search

results = pd.DataFrame(rf_rand.cv_results_)
results.sort_values('rank_test_score').head(20)


# **Trends in the randomized search results:**
# 
# - **n_estimators:**
#   - Higher numbers are performing better
#   - Remove 100, add 900
# - **min_samples_leaf:**
#   - Higher numbers are performing better
#   - Remove 1, add 4 and 5
# - **max_features:**
#   - None is performing better
#   - Remove 'sqrt'
# - **bootstrap:**
#   - True is performing better
#   - Remove False
# - **Transformer parameters:**
#   - No clear trends
#   - Leave as-is

rf_params['randomforestclassifier__n_estimators'] = [300, 500, 700, 900]
rf_params['randomforestclassifier__min_samples_leaf'] = [2, 3, 4, 5]
rf_params['randomforestclassifier__max_features'] = [None]
rf_params['randomforestclassifier__bootstrap'] = [True]
rf_params


# WARNING: 10 minutes
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X, y)


rf_grid.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (RF):** 0.829 👈
# - **Grid search (LR):** 0.828
# - **Randomized search (RF):** 0.825
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

rf_grid.best_params_


# ## 11.4 Q&A: How do I tune two models with a single grid search?

both_pipe = Pipeline([('preprocessor', ct), ('classifier', logreg)])


params1 = {}
params1['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
params1['classifier__penalty'] = ['l1', 'l2']
params1['classifier__C'] = [0.1, 1, 10]
params1['classifier'] = [logreg]
params1


params2 = {}
params2['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
params2['classifier__n_estimators'] = [300, 500]
params2['classifier__min_samples_leaf'] = [3, 4]
params2['classifier'] = [rf]
params2


both_params = [params1, params2]
both_params


both_grid = GridSearchCV(both_pipe, both_params, cv=5, scoring='accuracy',
                         n_jobs=-1)
both_grid.fit(X, y)


pd.DataFrame(both_grid.cv_results_)


both_grid.best_score_


both_grid.best_params_


# **Extensions of this approach:**
# 
# - Tune different preprocessing parameters for each model
# - Tune two different preprocessor objects

# ## 11.5 Q&A: How do I tune two models with a single randomized search?

both_rand = RandomizedSearchCV(both_pipe, both_params, cv=5, scoring='accuracy',
                               n_iter=10, random_state=1, n_jobs=-1)
both_rand.fit(X, y)


pd.DataFrame(both_rand.cv_results_)


# # Chapter 12: Ensembling multiple models

# ## 12.1 Introduction to ensembling

# **How to create an ensemble:**
# 
# - **Regression:** Average the predictions
# - **Classification:** Average the predicted probabilities, or let the classifiers vote on the class

# **Why does ensembling work?**
# 
# - "One-off" errors made by each model will be discarded when ensembling
# - Ensemble has a lower variance than any individual model

# ## 12.2 Ensembling logistic regression and random forests

logreg = LogisticRegression(solver='liblinear', random_state=1)
pipe = make_pipeline(ct, logreg)
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


rf = RandomForestClassifier(random_state=1, n_jobs=-1)
rf_pipe = make_pipeline(ct, rf)
cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR):** 0.811 👈
# - **Baseline (RF):** 0.811 👈

from sklearn.ensemble import VotingClassifier
vc = VotingClassifier([('clf1', logreg), ('clf2', rf)], voting='soft', n_jobs=-1)


# **Voting options for VotingClassifier:**
# 
# - **soft:** Average the predicted probabilities
# - **hard:** Majority vote using class predictions

vc_pipe = make_pipeline(ct, vc)
vc_pipe


# ## 12.3 Combining predicted probabilities

pipe.fit(X, y)
pipe.predict_proba(X_new)[:3]


rf_pipe.fit(X, y)
rf_pipe.predict_proba(X_new)[:3]


vc_pipe.fit(X, y)
vc_pipe.predict_proba(X_new)[:3]


vc_pipe.predict(X_new)[:3]


pipe.predict_proba(X_new)[80]


rf_pipe.predict_proba(X_new)[80]


vc_pipe.predict_proba(X_new)[80]


cross_val_score(vc_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC soft voting):** 0.818 👈
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

# ## 12.4 Combining class predictions

vc = VotingClassifier([('clf1', logreg), ('clf2', rf)], voting='hard', n_jobs=-1)
vc_pipe = make_pipeline(ct, vc)


cross_val_score(vc_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC hard voting):** 0.820 👈
# - **Baseline (VC soft voting):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

# **Why is this result misleading?**
# 
# - In the case of a tie, hard voting always chooses class 0
# - Thus hard voting is performing better than soft voting by chance

# ## 12.5 Choosing a voting strategy

# **Soft voting vs hard voting:**
# 
# - **Soft voting:**
#   - Preferred if you have an even number of models (especially two)
#   - Preferred if all models are well-calibrated
#   - Only works if all models have the predict_proba method
# - **Hard voting:**
#   - Preferred if some models are not well-calibrated
#   - Does not require the predict_proba method

vc = VotingClassifier([('clf1', logreg), ('clf2', rf)], voting='soft', n_jobs=-1)
vc_pipe = make_pipeline(ct, vc)


# ## 12.6 Tuning an ensemble with grid search

vc_params = {k:v for k, v in params.items() if k.startswith('col')}
vc_params


vc_pipe.named_steps.keys()


vc_pipe.named_steps['votingclassifier'].named_estimators


vc_params['votingclassifier__clf1__penalty'] = ['l1', 'l2']
vc_params['votingclassifier__clf1__C'] = [1, 10]
vc_params['votingclassifier__clf2__n_estimators'] = [100, 300]
vc_params['votingclassifier__clf2__min_samples_leaf'] = [2, 3]
vc_params


# WARNING: 1 minute
vc_grid = GridSearchCV(vc_pipe, vc_params, cv=5, scoring='accuracy', n_jobs=-1)
vc_grid.fit(X, y)


vc_grid.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC soft voting):** 0.834 👈
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC hard voting):** 0.820
# - **Baseline (VC soft voting):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

vc_grid.best_params_


vc_grid.predict(X_new)


# ## 12.7 Q&A: When should I use ensembling?

# **Should you ensemble?**
# 
# - **Advantages:**
#   - Improves model performance
# - **Disadvantages:**
#   - Adds more complexity
#   - Decreases interpretability

# **Advice for ensembling:**
# 
# - Include at least 3 models
# - Models should be performing well on their own
# - Ideal if they generate predictions using different processes

# ## 12.8 Q&A: How do I apply different weights to the models in an ensemble?

vc = VotingClassifier([('clf1', logreg), ('clf2', rf)], voting='soft',
                      weights=[2, 1], n_jobs=-1)
vc_pipe = make_pipeline(ct, vc)


pipe.predict_proba(X_new)[:3]


rf_pipe.predict_proba(X_new)[:3]


vc_pipe.fit(X, y)
vc_pipe.predict_proba(X_new)[:3]


cross_val_score(vc_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC soft voting):** 0.834
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC hard voting):** 0.820
# - **Baseline (VC soft voting):** 0.818
# - **Baseline (VC soft voting with LR weighted):** 0.816 👈
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

vc_params['votingclassifier__weights'] = [(1, 1), (2, 1), (1, 2)]

