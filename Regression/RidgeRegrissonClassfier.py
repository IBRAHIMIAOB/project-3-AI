from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

def RidgeRegressionClass(alpha=1.0, degree=2):
    print("""
------------------------------------------------------------------
Ridge Regression (Alpha {}, Degree {})""".format(alpha, degree))
    data = pd.read_csv("DataSets/insurance.csv") 
    To_Factor = ["sex" , "smoker","region"]
    for col in To_Factor:
        data[col] , unique_value = data[col].factorize()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform features into polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Fit ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_poly)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("---------------------------------------")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("---------------------------------------")

    return [mse, r2]
def predict_insurance_charges(InputList, alpha=1.0, degree=2):
    input_data = pd.DataFrame([InputList], columns=['age', 'sex', 'bmi', "children" ,  'smoker', 'region'])

    data = pd.read_csv('DataSets/insurance.csv')

    To_Factor = ["sex", "smoker", "region"]
    for col in To_Factor:
        data[col], unique_value = data[col].factorize()

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)

    input_data_poly = poly_features.transform(input_data)
    prediction = model.predict(input_data_poly)
    print("---------------------------------------")
    print(f"Your insurance charge is {prediction[0]}")
    print("---------------------------------------")

    return prediction[0]