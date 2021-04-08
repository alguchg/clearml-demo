import os
import pickle
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split


# Connecting ClearML
task = Task.init(project_name="uchicago", task_name="process dataset")

# get the original dataset
dataset = Dataset.get(dataset_project='uchicago', dataset_name='dataset1')

# create a copy that we can change,
dataset_folder = dataset.get_mutable_local_copy(target_folder='working_dataset', overwrite=True)
print(f"dataset_folder: {dataset_folder}")

# open the dataset pickle file
with open(dataset_folder + '/iris_dataset.pkl', 'rb') as f:
    iris = pickle.load(f)

# "process" data (i.e. we split it into train/test)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# store the dataset split into a pickle file
with open(dataset_folder + '/iris_dataset.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)

# create a new version of the dataset with the pickle file
new_dataset = Dataset.create(
    dataset_project='uchicago', dataset_name='dataset2', parent_datasets=[dataset])
new_dataset.sync_folder(local_path=dataset_folder)
new_dataset.upload()
new_dataset.finalize()

print('we are done')
