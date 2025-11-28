import pandas as pd

melbourne_file_path = 'C:/Users/IT Intern/Documents/GitHub/machine_learning_training_ground/Datasets/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
print(y)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.head())