#Practice of k-fold cross-validation in machine learning using scikit-learn
import pandas as pd

# Load dataset
melbourne_file_path = 'C:/Users/IT Intern/Documents/GitHub/machine_learning_training_ground/Datasets/melb_data.csv'

data = pd.read_csv(melbourne_file_path)

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

#Select target
y = data.Price


#Define a pipeline that uses a imputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

my_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

#obtain the cross validation scores
from sklearn.model_selection import cross_val_score

#multiply by -1 to get positive MAE scores (as sklearn returns negative MAE)
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print("MAE scores from cross-validation:")
print(scores)