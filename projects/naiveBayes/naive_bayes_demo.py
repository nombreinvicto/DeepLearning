# %%
# Working with a demo sms list
demo_sms = [
    "Hi!, How is it going?",
    "Call me when you get back home!",
    "Long time no see! Did you return?"
]
# %%
# convert all sms into lowercase
lower_case_sms = []

for sms in demo_sms:
    lower_case_sms.append(sms.lower())
lower_case_sms
# %%
# remove all punctuations from the sms
from string import punctuation
punctuation_less_sms = []

for sms in lower_case_sms:
    punctuation_less_sms.append("".join([char for char in sms if char not in punctuation]))
punctuation_less_sms
# %%
# Tokenising thoe words of the text messages
tokenised_sms = []

for sms in punctuation_less_sms:
    tokenised_sms.append(sms.split(" "))
tokenised_sms
# %%
# Creating a word frequency matrix for each of those text messages
from collections import Counter

for sms in tokenised_sms:
    print(Counter(sms))
# %%
# Different Approach - using Scikit learns Count Vectoriser - Does all the work for you
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(demo_sms)
cv.transform(demo_sms)
cv.get_feature_names()
# %%
# Looking at the word frequency matrix
import pandas as pd
pd.set_option("display.max_columns", 500)
frequency_matrix = pd.DataFrame(data=cv.transform(demo_sms).toarray(),
                                columns=cv.get_feature_names())
frequency_matrix
# %%
## Naive Bayes in ScikitLearn

# 1. Load and Preprocess Data
filepath = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\PyPlay\naiveBayes"
sms_data = pd.read_table("SMSSpamCollection")
sms_data.head()
# %%
# Adding labels to the columns
sms_data = pd.read_table("SMSSpamCollection", names=["label", "sms"])

# Converting Categorical Targets into numerical targets
sms_data["label"] = sms_data["label"].map({"ham": 0, "spam": 1})
sms_data.head()
# %%
# 2. Splitting data into training and testing sets
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(sms_data['sms'], sms_data['label'],
                                                random_state=42, test_size=0.33)

print("Training Features: ")
print(xtrain.head())
print("Training Targets: ")
print(ytrain.head())
# %%
# 3. Initialising a Naive Bayes Model and train on training dataset
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
cv = CountVectorizer()
transformed_xtrain = cv.fit_transform(xtrain)
transformed_xtest = cv.transform(xtest)

model.fit(transformed_xtrain, ytrain)
# %%
# 4. Measure performance on testing dataset after training
from sklearn.metrics import accuracy_score, precision_score, recall_score
predicted_ytest = model.predict(transformed_xtest)
print(f"Accuracy on testing dataset: {accuracy_score(ytest, predicted_ytest)}")
print(f"Precision on testing dataset: {precision_score(ytest, predicted_ytest)}")
print(f"Recall on testing dataset: {recall_score(ytest, predicted_ytest)}")
# %%
# 5. Measure performance on training dataset after training
from sklearn.metrics import accuracy_score, precision_score, recall_score
predicted_ytrain = model.predict(transformed_xtrain)
print(f"Accuracy on testing dataset: {accuracy_score(ytrain, predicted_ytrain)}")
print(f"Precision on testing dataset: {precision_score(ytrain, predicted_ytrain)}")
print(f"Recall on testing dataset: {recall_score(ytrain, predicted_ytrain)}")
