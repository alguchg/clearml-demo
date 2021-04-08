import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from clearml import Task, Dataset


# Connecting ClearML
task = Task.init(project_name="uchicago", task_name="training v1")

# get dataset with split/test
dataset = Dataset.get(dataset_project='uchicago', dataset_name='dataset2')

# get a read only version of the data
dataset_folder = dataset.get_local_copy()

# open the dataset pickle file
with open(dataset_folder + '/iris_dataset.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)


# train the model
model = LogisticRegression(solver='liblinear', multi_class='auto')
model.fit(X_train, y_train)

# store the trained model
joblib.dump(model, 'model.pkl', compress=True)

# print model predication results
result = model.score(X_test, y_test)
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.figure(1, figsize=(4, 3))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.get_cmap('viridis'))
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.title('Iris Types')
plt.show()

