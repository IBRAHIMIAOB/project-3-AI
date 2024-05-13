from sklearn.metrics import f1_score , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

def SupportVectorMachineClass():
    print("""
        ------------------------------------------------------------------
          Support Vector Machine """)
    
    data = pd.read_csv("DataSets/diabetes_prediction_dataset.csv")
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    data['smoking_history'] = data['smoking_history'].map({"never": 0, "No Info": 1, "former": 2, "ever": 3})
    data.dropna(inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("---------------------------------------")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("---------------------------------------")
    return [accuracy, f1]
SupportVectorMachineClass()