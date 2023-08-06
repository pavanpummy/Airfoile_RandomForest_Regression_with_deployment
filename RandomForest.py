# RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score


df = pd.read_csv("airfoil_self_noise.dat", sep='\t', header=None)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], random_state=0, test_size= 0.2)

# Create the Random Forest model
rf_regressor = RandomForestRegressor()

# Define a dictionary of hyperparameters and their distributions to sample from
param_dist = {
    'n_estimators': randint(5, 200), 
    'max_depth': randint(5, 50), 
    'min_samples_split': randint(2, 20),  
    'min_samples_leaf': randint(1, 20) 
}

# Create RandomizedSearchCV object with Random Forest model, the parameter distribution, and the number of iterations (n_iter)
random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_dist, n_iter=10, cv=5)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_

print("Best hyperparameters:", best_params)

y_rf_pred = random_search.predict(X_test)
r2_randomforest_value = r2_score(y_test, y_rf_pred)

print("R-squared value (accuracy):", r2_randomforest_value)

# Creating a picket byte file
with open('model.pkl', 'wb') as file:
    pickle.dump(random_search, file)