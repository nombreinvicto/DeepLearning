#!/usr/bin/env python
# coding: utf-8

# # Course: [Master Machine Learning with scikit-learn](https://courses.dataschool.io/view/courses/master-machine-learning-with-scikit-learn)
# 
# ## Chapters 10-16
# 
# *© 2024 Data School. All rights reserved.*

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


# # Chapter 13: Feature selection

# ## 13.1 Introduction to feature selection

# **Potential benefits of feature selection:**
# 
# - Higher accuracy
# - Greater interpretability
# - Faster training
# - Lower costs

# **Feature selection methods:**
# 
# - Human intuition
# - Domain knowledge
# - Data exploration
# - Automated methods

# **Methods for automated feature selection:**
# 
# - Intrinsic methods
# - Filter methods
# - Wrapper methods

# ## 13.2 Intrinsic methods: L1 regularization

# **What are intrinsic methods?**
# 
# - Feature selection happens automatically during model building
# - Also called: implicit methods, embedded methods

grid.best_params_


# **LogisticRegression tuning parameters:**
# 
# - **penalty:** Type of regularization
# - **C:** Amount of regularization

# **How does L1 regularization do feature selection?**
# 
# - Regularization shrinks model coefficients to help the model to generalize
# - L1 regularization shrinks some coefficients to zero, which removes those features

grid.best_estimator_.named_steps['logisticregression'].coef_


grid.best_estimator_.named_steps['logisticregression'].coef_[0].shape


sum(grid.best_estimator_.named_steps['logisticregression'].coef_[0] == 0)


pipe.named_steps['logisticregression'].get_params()


sum(pipe.named_steps['logisticregression'].coef_[0] == 0)


# **Advantages and disadvantages of intrinsic methods:**
# 
# - **Advantages:**
#   - No added computation
#   - No added steps
# - **Disadvantages:**
#   - Model-dependent

# ## 13.3 Filter methods: Statistical test-based scoring

# **How filter methods work:**
# 
# 1. Each feature is scored by its relationship to the target
# 2. Top scoring features (most informative features) are provided to the model

pipe


cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC):** 0.818
# - **Baseline (LR):** 0.811 👈
# - **Baseline (RF):** 0.811

# **How SelectPercentile works:**
# 
# 1. Scores each feature using the statistical test you specify
# 2. Passes to the model the percentage of features you specify

from sklearn.feature_selection import SelectPercentile, chi2
selection = SelectPercentile(chi2, percentile=50)


fs_pipe = make_pipeline(ct, selection, logreg)
fs_pipe


cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with SelectPercentile):** 0.819 👈
# - **Baseline (VC):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

# **SelectPercentile vs SelectKBest:**
# 
# - **SelectPercentile:** Specify percentage of features to keep
# - **SelectKBest:** Specify number of features to keep

# ## 13.4 Filter methods: Model-based scoring

# **How SelectFromModel works:**
# 
# 1. Scores each feature using the model you specify
#   - Model is fit on all features
#   - Coefficients or feature importances are used as scores
# 2. Passes to the prediction model features that score above a threshold you specify

# **Models that can be used by SelectFromModel:**
# 
# - Logistic regression
# - Linear SVC
# - Tree-based models
# - Any other model with coefficients or feature importances

logreg_selection = LogisticRegression(solver='liblinear', penalty='l1',
                                      random_state=1)


from sklearn.feature_selection import SelectFromModel
selection = SelectFromModel(logreg_selection, threshold='mean')


fs_pipe = make_pipeline(ct, selection, logreg)
fs_pipe


cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with SelectFromModel LR):** 0.826 👈
# - **Baseline (LR with SelectPercentile):** 0.819
# - **Baseline (VC):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

from sklearn.ensemble import ExtraTreesClassifier
et_selection = ExtraTreesClassifier(n_estimators=100, random_state=1)


selection = SelectFromModel(et_selection, threshold='mean')
fs_pipe = make_pipeline(ct, selection, logreg)
fs_pipe


cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with SelectFromModel LR):** 0.826
# - **Baseline (LR with SelectPercentile):** 0.819
# - **Baseline (VC):** 0.818
# - **Baseline (LR with SelectFromModel ET):** 0.815 👈
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

fs_params = params.copy()
fs_params['selectfrommodel__threshold'] = ['mean', '1.5*mean', -np.inf]


fs_params


# WARNING: 1 minute
fs_grid = GridSearchCV(fs_pipe, fs_params, cv=5, scoring='accuracy', n_jobs=-1)
fs_grid.fit(X, y)


fs_grid.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (LR with SelectFromModel ET):** 0.832 👈
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with SelectFromModel LR):** 0.826
# - **Baseline (LR with SelectPercentile):** 0.819
# - **Baseline (VC):** 0.818
# - **Baseline (LR with SelectFromModel ET):** 0.815
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

fs_grid.best_params_


# ## 13.5 Filter methods: Summary

# **Advantages and disadvantages of filter methods:**
# 
# - **Advantages:**
#   - Runs quickly (usually)
# - **Disadvantages:**
#   - Scores are not always correlated with predictive value
#   - Scores are calculated only once

# ## 13.6 Wrapper methods: Recursive feature elimination

# **Filter methods vs wrapper methods:**
# 
# - **Filter methods:** Features are scored once
# - **Wrapper methods:** Features are scored multiple times

# **How RFE works:**
# 
# 1. Scores each feature using the model you specify
#   - Model is fit on all features
#   - Coefficients or feature importances are used as scores
# 2. Removes the single worst scoring feature
# 3. Repeats steps 1 and 2 until it reaches the number of features you specify
# 4. Passes the remaining features to the prediction model

# **SelectFromModel vs RFE:**
# 
# - **SelectFromModel:** Scores your features a single time
# - **RFE:** Scores your features many times
#   - More computationally expensive
#   - May better capture the relationships between features

from sklearn.feature_selection import RFE
selection = RFE(logreg_selection, step=10)


fs_pipe = make_pipeline(ct, selection, logreg)
fs_pipe


cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (LR with SelectFromModel ET):** 0.832
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with SelectFromModel LR):** 0.826
# - **Baseline (LR with SelectPercentile):** 0.819
# - **Baseline (VC):** 0.818
# - **Baseline (LR with SelectFromModel ET):** 0.815
# - **Baseline (LR with RFE LR):** 0.814 👈
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

fs_params = params.copy()
fs_params['rfe__n_features_to_select'] = [None, 500]
fs_params


# WARNING: 2 minutes
fs_grid = GridSearchCV(fs_pipe, fs_params, cv=5, scoring='accuracy', n_jobs=-1)
fs_grid.fit(X, y)


fs_grid.best_score_


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (LR with SelectFromModel ET):** 0.832
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with SelectFromModel LR):** 0.826
# - **Grid search (LR with RFE LR):** 0.822 👈
# - **Baseline (LR with SelectPercentile):** 0.819
# - **Baseline (VC):** 0.818
# - **Baseline (LR with SelectFromModel ET):** 0.815
# - **Baseline (LR with RFE LR):** 0.814
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

fs_grid.best_params_


# **Advantages and disadvantages of RFE:**
# 
# - **Advantages:**
#   - Captures the relationships between features
# - **Disadvantages:**
#   - Scores are not always correlated with predictive value
#   - Computationally expensive
#   - Does not look ahead when removing features ("greedy" approach)

# ## 13.7 Q&A: How do I see which features were selected?

fs_pipe.named_steps.keys()


fs_pipe[0].fit_transform(X)


fs_pipe[0:2].fit_transform(X, y)


fs_pipe[1].get_support()


len(fs_pipe[1].get_support())


fs_pipe[1].get_support().sum()


fs_pipe[0].get_feature_names()


# ## 13.8 Q&A: Are the selected features the "most important" features?

# **Feature selection vs feature importance:**
# 
# - Multiple sets of features may perform similarly
# - Especially likely if there are many more features than samples ("p >> n")
# - Thus, feature selection does not necessarily determine feature importance

# ## 13.9 Q&A: Is it okay for feature selection to remove one-hot encoded categories?

# **Feature selection of one-hot encoded categories:**
# 
# - Feature selection examines each feature column independently (regardless of its "origin")
# - Each one-hot encoded column is conceptually independent from the others
# - Thus, it's acceptable for feature selection to ignore the origin of each column when removing features

# # Chapter 14: Feature standardization

# ## 14.1 Standardizing numerical features

# **Why is feature standardization useful?**
# 
# - Some models assume that features are centered around zero and have similar variances
# - Those models may perform poorly if that assumption is incorrect

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))


imp_scaler = make_pipeline(imp, scaler)


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp_scaler, ['Age', 'Fare', 'Parch']))


scaler_pipe = make_pipeline(ct, logreg)
scaler_pipe


cross_val_score(scaler_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (LR with SelectFromModel ET):** 0.832
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811
# - **Baseline (LR with numerical features standardized):** 0.810 👈

# **Why didn't feature standardization help?**
# 
# - Regularized linear models often benefit from standardization
# - However, the liblinear solver is robust to unscaled data

# ## 14.2 Standardizing all features

# **Why not use StandardScaler?**
# 
# - Our ColumnTransformer outputs a sparse matrix
# - Centering would cause memory issues by creating a dense matrix

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))


scaler_pipe = make_pipeline(ct, scaler, logreg)
scaler_pipe


cross_val_score(scaler_pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (LR with SelectFromModel ET):** 0.832
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (VC):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (LR with all features standardized):** 0.811 👈
# - **Baseline (RF):** 0.811
# - **Baseline (LR with numerical features standardized):** 0.810

# ## 14.3 Q&A: How do I see what scaling was applied to each feature?

scaler_pipe.fit(X, y)
scaler_pipe.named_steps['maxabsscaler'].scale_


# ## 14.4 Q&A: How do I turn off feature standardization within a grid search?

scaler_params = {}
scaler_params['logisticregression__C'] = [0.1, 1, 10]
scaler_params['maxabsscaler'] = ['passthrough', MaxAbsScaler()]


scaler_grid = GridSearchCV(scaler_pipe, scaler_params, cv=5, scoring='accuracy',
                           n_jobs=-1)
scaler_grid.fit(X, y)


pd.DataFrame(scaler_grid.cv_results_)


scaler_grid.best_params_


# ## 14.5 Q&A: Which models benefit from standardization?

# **When is feature standardization likely to be useful?**
# 
# - **Useful:**
#   - Distance-based models (KNN, SVM)
#   - Regularized models (linear or logistic regression with L1/L2)
# - **Not useful:**
#   - Tree-based models (random forests)

# # Chapter 15: Feature engineering with custom transformers

# ## 15.1 Why not use pandas for feature engineering?

# **Options for feature engineering:**
# 
# - **pandas:** Create features on original dataset, pass updated dataset to scikit-learn
# - **scikit-learn:** Create features using custom transformers
#   - Requires more work
#   - All transformations can be included in a Pipeline

# ## 15.2 Transformer 1: Rounding numerical values

df = pd.read_csv('http://bit.ly/MLtrain', nrows=10)
df


np.ceil(df[['Fare']])


from sklearn.preprocessing import FunctionTransformer
ceiling = FunctionTransformer(np.ceil)


ceiling.fit_transform(df[['Fare']])


ct = make_column_transformer(
    (ceiling, ['Fare']))
ct.fit_transform(df)


# ## 15.3 Transformer 2: Clipping numerical values

df


np.clip(df[['Age']], a_min=5, a_max=60)


clip = FunctionTransformer(np.clip, kw_args={'a_min':5, 'a_max':60})


clip.fit_transform(df[['Age']])


ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age']))
ct.fit_transform(df)


# ## 15.4 Transformer 3: Extracting string values

df


df['Cabin'].str.slice(0, 1)


df[['Cabin']].apply(lambda x: x.str.slice(0, 1))


def first_letter(df):
    return df.apply(lambda x: x.str.slice(0, 1))


def first_letter(df):
    return pd.DataFrame(df).apply(lambda x: x.str.slice(0, 1))


first_letter(df[['Cabin']])


letter = FunctionTransformer(first_letter)
letter.fit_transform(df[['Cabin']])


ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age']),
    (letter, ['Cabin']))
ct.fit_transform(df)


# ## 15.5 Rules for transformer functions

# **Input and output of transformer functions:**
# 
# - **Input:**
#   - 1D is allowed
#   - 2D is preferred: Enables it to accept multiple columns
# - **Output:**
#   - 2D is required

# ## 15.6 Transformer 4: Combining two features

df


df[['SibSp', 'Parch']].sum(axis=1)


np.array(df[['SibSp', 'Parch']]).sum(axis=1).reshape(-1, 1)


def sum_cols(df):
    return np.array(df).sum(axis=1).reshape(-1, 1)


sum_cols(df[['SibSp', 'Parch']])


total = FunctionTransformer(sum_cols)
total.fit_transform(df[['SibSp', 'Parch']])


ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age']),
    (letter, ['Cabin']),
    (total, ['SibSp', 'Parch']))
ct.fit_transform(df)


# ## 15.7 Revising the transformers

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))


ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age']),
    (letter, ['Cabin']),
    (total, ['SibSp', 'Parch']))


# **Issues to handle when updating the ColumnTransformer:**
# 
# 1. **Cabin** and **SibSp** weren't originally included
# 2. **Fare** and **Age** have missing values
# 3. **Cabin** is non-numeric and has missing values

df = pd.read_csv('http://bit.ly/MLtrain')


cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age', 'Cabin', 'SibSp']
X = df[cols]
X_new = df_new[cols]


imp_ceiling = make_pipeline(imp, ceiling)
imp_clip = make_pipeline(imp, clip)


X['Cabin'].str.slice(0, 1).value_counts(dropna=False)


# **Why are rare categories problematic for cross-validation?**
# 
# - Rare category values may all show up in the same testing fold
# - The rare category won't be learned during fit and will be treated as an unknown category
# - OneHotEncoder will error when it encounters an unknown category

ohe_ignore = OneHotEncoder(handle_unknown='ignore')


letter_imp_ohe = make_pipeline(letter, imp_constant, ohe_ignore)


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp_ceiling, ['Fare']),
    (imp_clip, ['Age']),
    (letter_imp_ohe, ['Cabin']),
    (total, ['SibSp', 'Parch']))


ct.fit_transform(X)


pipe = make_pipeline(ct, logreg)
pipe


cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy scores:**
# 
# - **Grid search (VC):** 0.834
# - **Grid search (LR with SelectFromModel ET):** 0.832
# - **Grid search (RF):** 0.829
# - **Grid search (LR):** 0.828
# - **Baseline (LR with more features):** 0.826 👈
# - **Baseline (VC):** 0.818
# - **Baseline (LR):** 0.811
# - **Baseline (RF):** 0.811

pipe.fit(X, y)
pipe.predict(X_new)


# ## 15.8 Q&A: How do I fix incorrect data types within a Pipeline?

demo = pd.DataFrame({'A': ['10', '20', '30'],
                     'B': ['40', '50', '60'],
                     'C': [70, 80, 90],
                     'D': ['x', 'y', 'z']})
demo


demo.dtypes


demo[['A', 'B']].astype('int')


demo[['A', 'B']].astype('int').dtypes


def make_integer(df):
    return pd.DataFrame(df).astype('int')


integer = FunctionTransformer(make_integer)
integer.fit_transform(demo[['A', 'B']])


integer.fit_transform(demo[['A', 'B']]).dtypes


demo.loc[2, 'B'] = ''


demo


integer.fit_transform(demo[['A', 'B']])


demo[['A', 'B']].apply(pd.to_numeric)


demo[['A', 'B']].apply(pd.to_numeric).dtypes


def make_number(df):
    return pd.DataFrame(df).apply(pd.to_numeric)


number = FunctionTransformer(make_number)
number.fit_transform(demo[['A', 'B']])


# ## 15.9 Q&A: How do I create features from datetime data?

ufo = pd.read_csv('http://bit.ly/ufosample', parse_dates=['Date'])
ufo = pd.read_csv('ufo.csv', parse_dates=['Date'])
ufo


ufo.dtypes


ufo['Date'].dt.day


def day_of_month(df):
    return df.apply(lambda x: x.dt.day)


day_of_month(ufo[['Date']])


def day_of_month(df):
    return pd.DataFrame(df).apply(lambda x: pd.to_datetime(x).dt.day)


def day_of_month(df):
    return pd.DataFrame(df, dtype=np.datetime64).apply(lambda x: x.dt.day)


ufo = pd.read_csv('http://bit.ly/ufosample')
ufo.dtypes


day_of_month(ufo[['Date']])


day = FunctionTransformer(day_of_month)
day.fit_transform(ufo[['Date']])


# ## 15.10 Q&A: How do I create feature interactions?

# **When are feature interactions useful?**
# 
# - When the combined impact of features is different from their independent impacts
# - **Example:**
#   - A and B (individually) each have a small positive impact
#   - A and B (combined) has a larger positive impact than expected

X[['Fare', 'SibSp', 'Parch']].to_numpy()


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False, interaction_only=True)


poly.fit_transform(X[['Fare', 'SibSp', 'Parch']])


# **Output columns:**
# 
# 1. Fare
# 2. SibSp
# 3. Parch
# 4. Fare * SibSp
# 5. Fare * Parch
# 6. SibSp * Parch

# **How to choose feature interactions:**
# 
# - Use expert knowledge
# - Explore the data
# - Create all possible interactions
#   - Not practical with a large number of features
#   - Increases risk of false positives

# **When are feature interactions not useful?**
# 
# - Tree-based models can learn feature interactions on their own
# - Linear models can sometimes replace the information supplied by interaction terms
# - **Conclusion:** Evaluate the model with and without interaction terms

# ## 15.11 Q&A: How do I save a Pipeline with custom transformers?

with open('pipe.pickle', 'wb') as f:
    pickle.dump(pipe, f)


def first_letter(df):
    return pd.DataFrame(df).apply(lambda x: x.str.slice(0, 1))


def sum_cols(df):
    return np.array(df).sum(axis=1).reshape(-1, 1)


import pandas as pd
import numpy as np


import pickle
with open('pipe.pickle', 'rb') as f:
    pipe_from_pickle = pickle.load(f)


cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age', 'Cabin', 'SibSp']
df_new = pd.read_csv('http://bit.ly/MLnewdata')
X_new = df_new[cols]


pipe_from_pickle.predict(X_new)


import cloudpickle
with open('pipe.pickle', 'wb') as f:
    cloudpickle.dump(pipe, f)


with open('pipe.pickle', 'rb') as f:
    pipe_from_pickle = pickle.load(f)


pipe_from_pickle.predict(X_new)


# ## 15.12 Q&A: Can FunctionTransformer be used with any transformation?

# **Stateless transformations:**
# 
# - **ceiling:** Rounding up to the next integer
# - **clip:** Limiting values to a range
# - **letter:** Extracting the first letter
# - **total:** Adding two columns

# **Stateful transformations:**
# 
# - **OneHotEncoder:** fit learns the categories
# - **CountVectorizer:** fit learns the vocabulary
# - **SimpleImputer:** fit learns the value to impute
# - **MaxAbsScaler:** fit learns the scale of each feature

# # Chapter 16: Workflow review #3

# ## 16.1 Recap of our workflow

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age', 'Cabin', 'SibSp']


df = pd.read_csv('http://bit.ly/MLtrain')
X = df[cols]
y = df['Survived']


df_new = pd.read_csv('http://bit.ly/MLnewdata')
X_new = df_new[cols]


imp = SimpleImputer()
imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
ohe = OneHotEncoder()
ohe_ignore = OneHotEncoder(handle_unknown='ignore')
vect = CountVectorizer()


def first_letter(df):
    return pd.DataFrame(df).apply(lambda x: x.str.slice(0, 1))


def sum_cols(df):
    return np.array(df).sum(axis=1).reshape(-1, 1)


ceiling = FunctionTransformer(np.ceil)
clip = FunctionTransformer(np.clip, kw_args={'a_min':5, 'a_max':60})
letter = FunctionTransformer(first_letter)
total = FunctionTransformer(sum_cols)


imp_ohe = make_pipeline(imp_constant, ohe)
imp_ceiling = make_pipeline(imp, ceiling)
imp_clip = make_pipeline(imp, clip)
letter_imp_ohe = make_pipeline(letter, imp_constant, ohe_ignore)


ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp_ceiling, ['Fare']),
    (imp_clip, ['Age']),
    (letter_imp_ohe, ['Cabin']),
    (total, ['SibSp', 'Parch']))


logreg = LogisticRegression(solver='liblinear', random_state=1)


pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe


pipe.predict(X_new)


# ## 16.2 What's the role of pandas?

# **Uses for pandas in the data science workflow:**
# 
# - **All projects:** Data exploration and visualization
# - **ML projects:** Testing out data transformations
# - **Non-ML projects:** Executing data transformations
