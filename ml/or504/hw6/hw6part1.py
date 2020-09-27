import pandas as pd

filepath = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\PyPlay\or504_hw5\hw6"
coaster_data = pd.read_excel(filepath + "\coaster.xls")
# %%
coaster_data.head()
# %%
coaster_data['Type'] = coaster_data['Type'].map({"Steel": 1, "Wooden": 0})
coaster_data['Inversion'] = coaster_data['Inversion'].map({"Yes": 1, "No": 0})
coaster_data.head()

# %%
coaster_features = coaster_data.drop(['Speed'], axis=1, inplace=False)
coaster_targets = coaster_data.drop(['Type', 'Duration', 'Height', 'Drop', 'Length',
                                     'Inversion'], axis=1, inplace=False)
print("\n Coaster Features: ")
print(coaster_features.head())
print("\n Coaster Targets")
print(coaster_targets.head())

# %%
from sklearn import preprocessing

feature_scaler = preprocessing.MinMaxScaler()
scaled_coaster_features = pd.DataFrame(data=feature_scaler.fit_transform(coaster_features),
                                       columns=coaster_features.columns.values)
# %%
target_scaler = preprocessing.MinMaxScaler()
scaled_coaster_targets = pd.DataFrame(data=target_scaler.fit_transform(coaster_targets),
                                      columns=coaster_targets.columns.values)
print("\n The following are the scaled features: ")
print(scaled_coaster_features.head())
print("\n The following are the scaled targets: ")
print(scaled_coaster_targets.head())

# %%
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(scaled_coaster_features, scaled_coaster_targets,
                                                random_state=42,
                                                test_size=0.12)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import optimizers

sgd = optimizers.SGD(lr=0.01, momentum=0.02)
activation_func = 'relu'

model = Sequential()
# My model
# model.add(Dense(256, activation=activation_func, input_shape=(6,)))
# model.add(Dropout(.2))
# model.add(Dense(128, activation=activation_func))
# model.add(Dropout(.1))
# model.add(Dense(64, activation=activation_func))
# model.add(Dropout(.1))
# model.add(Dense(1))

# Just NN model
model.add(Dense(7, activation=activation_func, input_shape=(6,)))
model.add(Dropout(.1))
model.add(Dense(1))

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mse'])
model.summary()
# %%
import numpy as np

train_features = np.array(xtrain)
train_targets = np.array(ytrain)

test_features = np.array(xtest)
test_targets = np.array(ytest)

model.fit(train_features, train_targets, epochs=5000, batch_size=5, verbose=0)

# %%
# Evaluating the model on the training and testing set
score = model.evaluate(test_features, test_targets)
print("\n Mean Square Error:", score)
# %%
print("\n These are the test data targets: ")
print(test_targets)
# %%
print("\n These are the predicted test data targets: ")
predictions = model.predict(test_features)
print(predictions)
# %%
# Testing our sample data
sample = [1, 170, 300, 306, 6100, 0]
scaled_sample_features = feature_scaler.transform(np.array([sample]))
#print(scaled_sample_features)
# %%
sample_prediction = model.predict(scaled_sample_features)
#print(sample_prediction)
# %%
print("\n This is the predicted speed of the sample dataset: ")
descaled_sample_prediction = target_scaler.inverse_transform(sample_prediction)
print(descaled_sample_prediction)
# %%
# Plotting Error Values
import matplotlib.pyplot as plt

mse_errors = model.history.history['mean_squared_error']
plt.plot(mse_errors, marker='+', color='r')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Errors")
plt.title("Mean Squared Error Trend Over 5000 Epochs")
plt.show()