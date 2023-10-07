import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load the csv file
df = pd.read_csv(r"C:\Users\Barath\Desktop\iitk assignment 2\output_file.csv")


# Create a StandardScaler instance
scaler = StandardScaler()

# Fit and transform the DataFrame using the scaler
df_normalized = scaler.fit_transform(df)

# Convert df_normalized into a new DataFrame with column names
df_normalized = pd.DataFrame(df_normalized, columns=df.columns)


# column
for col in df_normalized.columns:
    new_col = col.replace(' ', '_')  # Replace spaces with underscores
    df_normalized.rename(columns={col: new_col}, inplace=True)


'''
# Define the lower and upper percentiles for winsorization (1% and 99%)
lower_percentile = 10
upper_percentile = 20

# Create a copy of the DataFrame to store winsorized values
df_winsorized = df_normalized.copy()

# Iterate through each column and winsorize the data
for column in df_winsorized.columns:
    winsorized_values = stats.mstats.winsorize(df_winsorized[column], limits=(lower_percentile / 100, upper_percentile / 100))
    df_winsorized[column] = winsorized_values
'''
y = df['quality']

data = df.drop(['quality'], axis=1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20)

# Create an instance of Decision tree regressor

dtr = DecisionTreeRegressor()
# Fit the model to the training data
dtr.fit(X_train,y_train)


# Make predictions on the test data
dtr_pred = dtr.predict(X_test)

## Define the wine quality labels in the desired order
wine_quality_labels = ["Poor (3)", "Below Average (4)", "Average (5)", "Good (6)", "Very Good (7)", "Excellent (8)"]

# Get the predicted values for wine quality
predicted_values = dtr.predict(X_test)
'''
# Map the predicted values to wine quality labels
predicted_labels = [wine_quality_labels[min(max(int(value), 0), len(wine_quality_labels)-1)] for value in predicted_values]

# Now, `predicted_labels` contains the wine quality labels corresponding to the predicted values
'''
predicted_quality = round(predicted_values[0])  # Round the predicted value
predicted_label = wine_quality_labels[min(max(predicted_quality, 0), len(wine_quality_labels)-1)]  # Map to a label


# Save the model as a pickle file
pickle.dump(dtr, open("model.pkl", "wb"))
