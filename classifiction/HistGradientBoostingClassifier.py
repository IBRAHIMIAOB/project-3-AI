import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def HistGradientBoostingClassifierr():
    print("""
------------------------------------------------------------------
HistGradientBoosting Classifier""")
    data = pd.read_csv('DataSets/diabetes_prediction_dataset.csv')
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    data['smoking_history'] = data['smoking_history'].map({"never": 0, "No Info": 1, "former": 2, "ever": 3})
    data.dropna(inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hist_gb_classifier = HistGradientBoostingClassifier()

    hist_gb_classifier.fit(X_train, y_train)

    y_pred = hist_gb_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("---------------------------------------")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("---------------------------------------")

    choice = input("Do you want to plot the feature importance (y / n)? ")
    if choice.lower() == "y":
        result = permutation_importance(hist_gb_classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(X.columns)), result.importances_mean[sorted_idx])
        plt.yticks(range(len(X.columns)), X.columns[sorted_idx])
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Permutation Importance')
        plt.show()

    return [accuracy, f1]

HistGradientBoostingClassifierr()
