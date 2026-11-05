import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#loading the dataset
data = pd.read_csv('house_data.csv')
X = data[['Size', 'Bedrooms', 'Age']]
y = data['Price']
# creating linear regression model object
model = LinearRegression()
# train the model using the dataset
model.fit(X, y)
# save the trained model to a file
joblib.dump(model, 'house_model.pkl')
#print confirmation msg
print("Model trained and saved as 'house_model.pkl'")