import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load your CSV dataset
data = pd.read_csv("D:\electrognathograph.csv")

# Calculate the difference between truth values and experimental values
data['left_diff'] = data['left'] - data['left extremum primary']
data['right_diff'] = data['right'] - data['right extremum primary']
data['left_arbitrary_diff'] = data['left'] - data['left extremum arbitrary']
data['right_arbitrary_diff'] = data['right'] - data['right extremum arbitrary']

# Create the target variables
data['left_calibration'] = data['left_diff'] + data['left_arbitrary_diff']
data['right_calibration'] = data['right_diff'] + data['right_arbitrary_diff']

# Select features and targets
left_features = ['left_diff', 'left_arbitrary_diff']
right_features = ['right_diff', 'right_arbitrary_diff']
left_target = 'left_calibration'
right_target = 'right_calibration'

# Split the data into training and test sets for left and right calibration separately
X_left_train, X_left_test, y_left_train, y_left_test = train_test_split(data[left_features], data[left_target], test_size=0.2, random_state=42)
X_right_train, X_right_test, y_right_train, y_right_test = train_test_split(data[right_features], data[right_target], test_size=0.2, random_state=42)

# Initialize and train the Linear Regression models for left and right calibration
left_model = LinearRegression()
right_model = LinearRegression()

left_model.fit(X_left_train, y_left_train)
right_model.fit(X_right_train, y_right_train)

# Make predictions on the test sets for left and right calibration
y_left_pred = left_model.predict(X_left_test)
y_right_pred = right_model.predict(X_right_test)

# Evaluate the models for left and right calibration
left_mae = mean_absolute_error(y_left_test, y_left_pred)
right_mae = mean_absolute_error(y_right_test, y_right_pred)

print(f'Left Calibration Mean Absolute Error: {left_mae}')
print(f'Right Calibration Mean Absolute Error: {right_mae}')

# Now you can use the trained models to generate calibration values for new experimental values for the left and right sides separately
