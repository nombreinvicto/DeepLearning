#!/usr/bin/env python
# coding: utf-8

# # Course: [Master Machine Learning with scikit-learn](https://courses.dataschool.io/view/courses/master-machine-learning-with-scikit-learn)
# 
# ## Chapters 17-20
# 
# *© 2024 Data School. All rights reserved.*

# # Chapter 17: High-cardinality categorical features

# ## 17.1 Recap of nominal and ordinal features

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn import set_config


df = pd.read_csv('http://bit.ly/MLtrain')
y = df['Survived']


logreg = LogisticRegression(solver='liblinear', random_state=1)
rf = RandomForestClassifier(random_state=1, n_jobs=-1)


set_config(display='diagram')


# **Types of categorical features:**
# 
# - **Nominal:** Unordered categories
#   - Embarked
#   - Sex
# - **Ordinal:** Ordered categories
#   - Pclass

# **Advice for encoding categorical data:**
# 
# - **Nominal feature:** Use OneHotEncoder
# - **Ordinal feature stored as numbers:** Leave as-is
# - **Ordinal feature stored as strings:** Use OrdinalEncoder

# **Why use OneHotEncoder for Embarked?**
# 
# - **OneHotEncoder:**
#   - Outputs 3 features
#   - Model can learn the relationship between each feature and the target value
# - **OrdinalEncoder:**
#   - Outputs 1 feature
#   - Implies an ordering that doesn't inherently exist
#   - Linear model can't necessarily learn the relationships in the data

df['Embarked'].value_counts()


# ## 17.2 Preparing the census dataset

census = pd.read_csv('http://bit.ly/censusdataset')
census = pd.read_csv('census.csv')


census.describe(include='object')


# **Categorical features in census dataset:**
# 
# - **High-cardinality (3 of 8):** education, occupation, native-country
# - **Nominal (7 of 8):** All except education

census['class'].value_counts(normalize=True)


census_cols = ['workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country']
census_X = census[census_cols]
census_y = census['class']


# ## 17.3 Setting up the encoders

ohe = OneHotEncoder()
oe = OrdinalEncoder()


oe.fit(census_X).categories_


census_X['native-country'].value_counts()


# **Resolving the cross-validation error caused by rare categories:**
# 
# - **OneHotEncoder:** Set handle_unknown='ignore'
# - **OrdinalEncoder (before 0.24):** Define the categories in advance
# - **OrdinalEncoder (starting in 0.24):** Set handle_unknown='use_encoded_value'

ohe_ignore = OneHotEncoder(handle_unknown='ignore')


cats = [census_X[col].unique() for col in census_X[census_cols]]
cats


oe_cats = OrdinalEncoder(categories=cats)


# ## 17.4 Encoding nominal features for a linear model

ohe_ignore.fit_transform(census_X).shape


oe_cats.fit_transform(census_X).shape


ohe_logreg = make_pipeline(ohe_ignore, logreg)
oe_logreg = make_pipeline(oe_cats, logreg)


cross_val_score(ohe_logreg, census_X, census_y, cv=5, scoring='accuracy').mean()


cross_val_score(oe_logreg, census_X, census_y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy when encoding nominal features:**
# 
# - Linear model:
#   - **OneHotEncoder:** 0.833
#   - **OrdinalEncoder:** 0.755

# ## 17.5 Encoding nominal features for a non-linear model

ohe_rf = make_pipeline(ohe_ignore, rf)
oe_rf = make_pipeline(oe_cats, rf)


# WARNING: 30 seconds
cross_val_score(ohe_rf, census_X, census_y, cv=5, scoring='accuracy').mean()


cross_val_score(oe_rf, census_X, census_y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy when encoding nominal features:**
# 
# - Linear model:
#   - **OneHotEncoder:** 0.833
#   - **OrdinalEncoder:** 0.755
# - Non-linear model:
#   - **OneHotEncoder:** 0.826 👈
#   - **OrdinalEncoder:** 0.825 👈

# ## 17.6 Combining the encodings

census_X['education'].unique()


cats = [[' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th',
         ' 11th', ' 12th', ' HS-grad', ' Some-college', ' Assoc-voc',
         ' Assoc-acdm', ' Bachelors', ' Masters', ' Prof-school', ' Doctorate']]
oe_cats = OrdinalEncoder(categories=cats)


ct = make_column_transformer(
    (oe_cats, ['education']),
    remainder=ohe_ignore)


ct.fit_transform(census_X).shape


oe_ohe_logreg = make_pipeline(ct, logreg)
oe_ohe_rf = make_pipeline(ct, rf)


cross_val_score(oe_ohe_logreg, census_X, census_y, cv=5, scoring='accuracy').mean()


# WARNING: 30 seconds
cross_val_score(oe_ohe_rf, census_X, census_y, cv=5, scoring='accuracy').mean()


# **Pipeline accuracy when encoding nominal features:**
# 
# - Linear model:
#   - **OneHotEncoder:** 0.833
#   - **OrdinalEncoder:** 0.755
#   - **OneHotEncoder (7 features) + OrdinalEncoder (education):** 0.832 👈
# - Non-linear model:
#   - **OneHotEncoder:** 0.826
#   - **OrdinalEncoder:** 0.825
#   - **OneHotEncoder (7 features) + OrdinalEncoder (education):** 0.825 👈

# ## 17.7 Best practices for encoding

# **Summary of best practices:**
# 
# - **Nominal features, linear model:**
#   - OneHotEncoder
# - **Nominal features, non-linear model:**
#   - OneHotEncoder
#   - OrdinalEncoder
#     - Don't define category ordering
#     - Much faster than OneHotEncoder (if features have high cardinality)
# - **Ordinal features:**
#   - OneHotEncoder
#   - OrdinalEncoder
#     - Define category ordering
#     - Much faster than OneHotEncoder (if features have high cardinality)

# # Chapter 18: Class imbalance

# ## 18.1 Introduction to class imbalance

# **Overview of class imbalance:**
# 
# - "Class imbalance" is when classes are not equally represented
# - Inherent to many domains
# - Can occur in binary and multiclass problems
# - Binary problems have a "majority class" and a "minority class"
# - Makes it harder for the model to learn patterns in the minority class
# - Most datasets have some class imbalance
# - Greater class imbalance requires more specialized techniques

y.value_counts(normalize=True)


census_y.value_counts(normalize=True)


# ## 18.2 Preparing the mammography dataset

scan = pd.read_csv('http://bit.ly/scanrecords')
scan = pd.read_csv('scan.csv')
scan


scan['class'].value_counts(normalize=True)


scan['class'] = scan['class'].map({"'-1'":0, "'1'":1})
scan


scan_X = scan.drop('class', axis='columns')
scan_y = scan['class']


# ## 18.3 Evaluating a model with train/test split

# **Steps of train/test split:**
# 
# 1. Split rows into training and testing sets
# 2. Train model on training set
# 3. Make predictions on testing set
# 4. Evaluate predictions

# **Cross-validation vs train/test split:**
# 
# - Cross-validation is just repeated train/test split
# - Cross-validation outputs a lower variance estimate of performance

X_train, X_test, y_train, y_test = train_test_split(scan_X, scan_y,
                                                    test_size=0.25, random_state=1,
                                                    stratify=scan_y)


# **Stratified sampling:**
# 
# - Ensures that each set is representative of the dataset
# - Especially important when there is severe class imbalance

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ## 18.4 Exploring the results with a confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(logreg, X_test, y_test, cmap='Blues')


# &nbsp; | Predicted 0 | Predicted 1
# :--- | :---: | :---:
# **True label 0** | True Negatives | False Positives
# **True label 1** | False Negatives | True Positives

# ## 18.5 Calculating rates from a confusion matrix

confusion_matrix(y_test, y_pred)


28 / (37 + 28)


2721 / (2721 + 10)


10 / (2721 + 10)


# **Calculated rates:**
# 
# - **True Positive Rate:** 0.431
#   - Recall for class 1
# - **True Negative Rate:** 0.996
#   - Recall for class 0
# - **False Positive Rate:** 0.004
#   - 1 minus True Negative Rate

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


confusion_matrix(y_test, np.ones_like(y_pred))


# ## 18.6 Using AUC as the evaluation metric

y_score = logreg.predict_proba(X_test)[:, 1]
y_score


# **Area Under the ROC Curve (AUC):**
# 
# - Measures how well the model separates the classes
# - Wants the model to assign higher probabilites to class 1 samples than class 0 samples
# - Can be used with any classifier that outputs predicted probabilities

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_score)


# **AUC scores:**
# 
# - **Perfect model:** 1.0
# - **Uninformed model:** 0.5

# **Decision threshold:**
# 
# - Predicted probability value above which a model will predict the positive class
# - Default threshold is **0.5**:
#   - Probability of **0.7** -> predicts class 1
#   - Probability of **0.2** -> predicts class 0

confusion_matrix(y_test, y_pred)


sum(y_score > 0.5)


# **What can we infer so far?**
# 
# - **High AUC:** Model is already doing a good job separating the classes
# - **Low TPR:** Default decision threshold is not serving us well

from sklearn.metrics import plot_roc_curve
disp = plot_roc_curve(logreg, X_test, y_test)


# **Interpreting the ROC curve:**
# 
# - Plot of the TPR vs FPR for all possible decision thresholds
# - Move to another point on the curve by changing the threshold
#   - Threshold is not shown on this plot
# - AUC is the percentage of the box underneath the curve

# **Next steps:**
# 
# 1. Improve the model's AUC
# 2. Explore alternative decision thresholds

# ## 18.7 Cost-sensitive learning

# **How to improve the model's AUC?**
# 
# - **Any technique covered in the course:**
#   - Hyperparameter tuning
#   - Feature selection
#   - Trying non-linear models
#   - Etc.
# - **Cost-sensitive learning:**
#   - Particularly useful when there's class imbalance
#   - Insight: Not all prediction errors have the same "cost"
#   - If positive samples are rare:
#     - **False Negatives** have a higher cost
#     - **False Positives** have a lower cost

# **How does cost-sensitive learning work?**
# 
# - Gives more weight to samples from the minority class (positive class)
#   - Model is penalized more for False Negatives than False Positives
# - Model's goal is to minimize total cost
#   - Model may be biased toward predicting the minority class

logreg_cost = LogisticRegression(solver='liblinear', class_weight='balanced',
                                 random_state=1)


logreg_cost.fit(X_train, y_train)
y_pred = logreg_cost.predict(X_test)
y_score = logreg_cost.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_score)


print(classification_report(y_test, y_pred))


# **Changes due to cost-sensitive learning:**
# 
# - **TPR:** 0.43 -> 0.88
# - **FPR:** 0.00 -> 0.11

# ## 18.8 Tuning the decision threshold

disp = plot_roc_curve(logreg_cost, X_test, y_test)


confusion_matrix(y_test, y_pred)


sum(y_score > 0.5)


sum(y_score > 0.25)


(y_score > 0.25) * 1


confusion_matrix(y_test, y_score > 0.25)


print(classification_report(y_test, y_score > 0.25))


# **Changes due to decreasing the threshold:**
# 
# - **TPR:** 0.88 -> 0.94
# - **FPR:** 0.11 -> 0.30

confusion_matrix(y_test, y_score > 0.75)


print(classification_report(y_test, y_score > 0.75))


# **Changes due to increasing the threshold:**
# 
# - **TPR:** 0.88 -> 0.77
# - **FPR:** 0.11 -> 0.04

# # Chapter 19: Class imbalance walkthrough

# ## 19.1 Best practices for class imbalance

# **New concepts from chapter 18:**
# 
# - Class imbalance
# - Confusion matrix, TPR, TNR, FPR
# - Classification report
# - ROC curve and AUC
# - Decision threshold
# - Cost-sensitive learning

# **Workflow improvements in chapter 19:**
# 
# 1. Use cross-validation instead of train/test split
#   - Easier to use
#   - More reliable performance estimates
# 2. Use new data to tune the decision threshold
#   - More reliable TPR/FPR estimates

# ## 19.2 Step 1: Splitting the dataset

# **Different uses of train/test split:**
# 
# - **Chapter 18:** Set aside data for model evaluation
# - **Chapter 19:** Set aside data for tuning the decision threshold

X_train, X_test, y_train, y_test = train_test_split(scan_X, scan_y,
                                                    test_size=0.25, random_state=1,
                                                    stratify=scan_y)


# ## 19.3 Step 2: Optimizing the model on the training set

cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()


im_params = {}
im_params['penalty'] = ['l1', 'l2']
im_params['C'] = [0.1, 1, 10]
im_params['class_weight'] = [None, 'balanced', {0:1, 1:99}, {0:3, 1:97}]


training_grid = GridSearchCV(logreg, im_params, cv=5, scoring='roc_auc', n_jobs=-1)
training_grid.fit(X_train, y_train)
training_grid.best_score_


training_grid.best_params_


best_model = training_grid.best_estimator_


# ## 19.4 Step 3: Evaluating the model on the testing set

y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]


roc_auc_score(y_test, y_score)


disp = plot_roc_curve(best_model, X_test, y_test)


confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))


# ## 19.5 Step 4: Tuning the decision threshold

confusion_matrix(y_test, y_score > 0.55)


print(classification_report(y_test, y_score > 0.55))


# **Changes due to increasing the threshold:**
# 
# - **TPR:** 0.95 -> 0.92
# - **FPR:** 0.24 -> 0.20

# ## 19.6 Step 5: Retraining the model and making predictions

best_model.fit(scan_X, scan_y)


np.random.seed(1)
scan_X_new = np.random.randint(0, 3, (4, 6))
scan_X_new


scan_y_new_score = best_model.predict_proba(scan_X_new)[:, 1]


(scan_y_new_score > 0.55) * 1


# ## 19.7 Q&A: Should I use an ROC curve or a precision-recall curve?

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]


confusion_matrix(y_test, y_pred)


62 / (3 + 62)


62 / (649 + 62)


# **Calculated rates:**
# 
# - **Recall:** 0.95
# - **Precision:** 0.09

print(classification_report(y_test, y_pred))


from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(best_model, X_test, y_test)


# **Interpreting the precision-recall curve:**
# 
# - Plot of precision vs recall for all possible decision thresholds
# - Move to another point on the curve by changing the threshold
# - Average precision is the percentage of the box underneath the curve

from sklearn.metrics import average_precision_score
average_precision_score(y_test, y_score)


# **Precision-recall scores:**
# 
# - **Perfect model:** 1.0
# - **Uninformed model:** 0.02 in this case (fraction of positive samples)

confusion_matrix(y_test, y_pred)


# **Calculated rates:**
# 
# - **TPR / Recall:** 0.95
# - **FPR:** 0.24
# - **Precision:** 0.09

# **Critique of AUC in cases of class imbalance:**
# 
# - **Results of severe class imbalance:**
#   - Very large number of True Negatives
#   - FPR will be artificially low
#   - AUC will be artificially high and is no longer realistic
# - **Example:**
#   - Increase True Negatives from 2082 to 200000
#   - FPR would drop from 0.24 to 0.003
#   - AUC would increase
#   - Precision would still be 0.09
# - **Proposed solution:**
#   - Use precision-recall curve and average precision
#   - More realistic because it ignores the number of True Negatives

# &nbsp; | Predicted 0 | Predicted 1
# :--- | :---: | :---:
# **True label 0** | 200000 | 649
# **True label 1** | 3 | 62

# **My responses to this crititque:**
# 
# - **Neither metric is inherently better:**
#   - AUC focuses on both classes
#   - Average precision focuses on positive class only
#   - In this case: both classes are relevant
# - **FPR and precision are both artificially low:**
#   - Excellent model can still have a low precision
#   - Example: Model can have FPR of 0.003 and TPR of 1.0, but precision of 0.09
# - **AUC score itself is irrelevant:**
#   - Our goal is to choose between models
#   - Maximizing AUC helps you choose the most skillful model
#   - Balance TPR and FPR based on your priorities
#   - AUC score itself is never your business objective

# **Summary of AUC vs average precision:**
# 
# - Both are reasonable metrics to maximize (even with class imbalance)
# - Neither metric perfectly captures model performance
# - AUC focuses on both classes, average precision focuses on positive class only

# ## 19.8 Q&A: Can I use a different metric such as F1 score?

# **Alternative metrics in cases of class imbalance:**
# 
# - F1 score or F-beta score
# - Balanced accuracy
# - Cohen's kappa
# - Matthews correlation coefficient

# **Advantage of AUC and average precision:**
# 
# - They don't require you to choose a decision threshold
# - You can first maximize classifier performance, then alter the decision threshold

# **Disadvantage of F1 score (and others) during model tuning:**
# 
# - You're optimizing hyperparameters based on a decision threshold of 0.5
# - An alternative decision threshold might lead to a more optimal model

# ## 19.9 Q&A: Should I use resampling to fix class imbalance?

# **What is resampling?**
# 
# - Transforming the training data in order to balance the classes
# - Fixes class imbalance at the dataset level

# **Common resampling approaches:**
# 
# - **Undersampling (downsampling):**
#   - Deleting samples from the majority class
#   - Risk of deleting important samples
# - **Oversampling (upsampling):**
#   - Creating new samples of the minority class
#   - Done through duplication or simulation (SMOTE)
#   - Risk of adding meaningless new samples

# **Is resampling helpful?**
# 
# - Sometimes helpful, sometimes not
# - No one algorithm works best across all datasets and models

# **How to implement resampling:**
# 
# - Not yet supported by scikit-learn
# - Use imbalanced-learn library (compatible with scikit-learn)

# **Advice for proper resampling:**
# 
# - Treat like any other preprocessing technique
#   - Include in a Pipeline to avoid data leakage
# - Only apply to training data
#   - Model should be tested on natural, imbalanced data

# # Chapter 20: Going further

# ## 20.1 Q&A: How do I read the scikit-learn documentation?

# **Key pages in the documentation:**
# 
# - **API reference:** List of classes and functions in each module
# - **Class documentation:** Detailed view of a class
# - **User Guide:** Advice for proper usage of a class or function
# - **Examples:** More complex usage examples
# - **Glossary:** Definitions of important terms

# ## 20.2 Q&A: How do I stay up-to-date with new scikit-learn features?

# **Pages to review after each release:**
# 
# - **Release highlights:** Most important or exciting changes
# - **Detailed release notes:** All new features, enhancements, and API changes
#   - Only review the modules you use
#   - Read the class documentation for more details
#   - Read the pull request for further context

# ## 20.3 Q&A: How do I improve my Machine Learning skills?

# **Practice what you've learned:**
# 
# - Choose different types of problems and datasets
# - Learn about other topics and modules we didn't cover

# **Study Machine Learning models:**
# 
# - **Benefits:**
#   - Learn which models are worth trying
#   - Learn how to tune those models
#   - Learn how to interpret those models
# - **Resource:**
#   - An Introduction to Statistical Learning (book): https://www.statlearning.com

# ## 20.4 Q&A: How do I learn Deep Learning?

# **Learning Deep Learning:**
# 
# - **Why?**
#   - Deep Learning will provide superior results for some specialized problems
# - **How?**
#   - Practical Deep Learning for Coders (course): https://course.fast.ai
#   - Higher learning curve than Machine Learning, but based on many of the same principles
