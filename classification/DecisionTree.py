import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
def DecisionTreeClassifierr():
    print("""
------------------------------------------------------------------
Decision Tree Classifier""")
    data = pd.read_csv('DataSets/diabetes_prediction_dataset.csv')
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    data['smoking_history'] = data['smoking_history'].map({"never" : 0 ,"No Info":1 , "former": 2 , "ever" : 3 })
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_classifier = DecisionTreeClassifier()

    dt_classifier.fit(X_train, y_train)

    y_pred = dt_classifier.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("---------------------------------------")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("---------------------------------------")
    
    choice = input("Do you want to plot the tree (y / n) ? (Tree is so big plotting it may take time) : ")
    if choice == "y" or choice =="Y":
        plt.figure(figsize=(12, 8))
        plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['0', '1'])  # Adjust class_names as per your dataset
        plt.show()
    
    return [accuracy , f1]

DecisionTreeClassifierr()
