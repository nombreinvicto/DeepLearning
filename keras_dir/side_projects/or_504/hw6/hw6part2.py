import pandas as pd

pd.set_option("display.max_columns", 1000)
filepath = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\PyPlay\or504_hw5\hw6"
goals_data = pd.read_excel(filepath + "\goals.xls")
# %%
goals_data.head()
# %%
goals_data['Pressure'] = goals_data['Pressure'].map({"Y": 1, "N": 0})
goals_data['Good'] = goals_data['Good'].map({"Y": 1, "N": 0})
goals_data.head()
# %%
# Converting multiclass categorical inut data into numerical data
goals_data_dummied = pd.get_dummies(goals_data)
goals_data_dummied.head()
# %%
goals_features = goals_data_dummied.drop(['Good'], axis=1, inplace=False)
goals_targets = pd.DataFrame(goals_data_dummied['Good'], columns=['Good'])
print("\n Coaster Features: ")
print(goals_features.head())
print("\n Coaster Targets")
print(goals_targets.head())

# %%
from sklearn import preprocessing

feature_scaler = preprocessing.MinMaxScaler()
scaled_goals_features = pd.DataFrame(data=feature_scaler.fit_transform(goals_features),
                                     columns=goals_features.columns.values)
# %%
target_scaler = preprocessing.MinMaxScaler()
scaled_goals_targets = pd.DataFrame(data=target_scaler.fit_transform(goals_targets),
                                    columns=goals_targets.columns.values)
print("\n The following are the scaled features: ")
print(scaled_goals_features.head())
print("\n The following are the scaled targets: ")
print(scaled_goals_targets.head())


# %%
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(scaled_goals_features, scaled_goals_targets,
                                                random_state=42,
                                                test_size=0.25)
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import optimizers

sgd = optimizers.SGD(lr=0.01, momentum=0.02)


model = Sequential()
# My model
model.add(Dense(256, activation='relu', input_shape=(12,)))
model.add(Dropout(.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Just NN model
# model.add(Dense(7, activation=activation_func, input_shape=(6,)))
# model.add(Dropout(.1))
# model.add(Dense(1))

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# %%
import numpy as np

train_features = np.array(xtrain)
#train_targets = np.array(ytrain)

test_features = np.array(xtest)
#test_targets = np.array(ytest)

final_targets_train = np.array(keras.utils.to_categorical(ytrain, 2))
final_targets_test = np.array(keras.utils.to_categorical(ytest, 2))

model.fit(train_features, final_targets_train, epochs=5000, batch_size=10, verbose=1)

# %%
# Evaluating the model on the training and testing set
score = model.evaluate(test_features, final_targets_test)
print("\n Accuracy Score:", score)
# %%
print("\n These are the test data targets: ")
print(final_targets_test)
# %%
print("\n These are the predicted test data targets: ")
predictions = model.predict(test_features)
print(predictions)
# %%
# Testing our sample data
temp = goals_data.drop(['Good'], axis=1)
sample = np.array([[57, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])
sample_df = pd.DataFrame(sample, columns=xtrain.columns.values)
sample_df

scaled_sample_data = pd.DataFrame(feature_scaler.transform(sample_df),
                                  columns=sample_df.columns.values)
scaled_sample_data

# %%
sample_prediction = model.predict(np.array(scaled_sample_data))
sample_prediction
# %%
# Plotting Error Values
import matplotlib.pyplot as plt

accuracy_history = model.history.history['acc']
plt.plot(accuracy_history, marker='+', color='r')
plt.xlabel("Epochs")
plt.ylabel("Accuracy Scores")
plt.title("Accuracy Trend Over 5000 Epochs")
plt.show()
