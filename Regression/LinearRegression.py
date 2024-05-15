from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

def LinearRegressionClass():
    print("""
------------------------------------------------------------------
Linear Regression """)
    data = pd.read_csv("DataSets/insurance.csv") 
    To_Factor = ["sex" , "smoker","region"]
    for col in To_Factor:
        data[col] , unique_value = data[col].factorize()
    X = data.iloc[: , :-1]
    y = data.iloc[: , -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("---------------------------------------")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("---------------------------------------")

    return [mse, r2] 
