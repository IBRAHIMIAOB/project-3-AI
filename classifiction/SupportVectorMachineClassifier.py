from sklearn.svm import SVC

def SupportVectorMachineClass():
    print("""
        ------------------------------------------------------------------
          Support Vector Machine Class""")
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['gender'] = self.data['gender'].map({'Male': 0, 'Female': 1})
        self.data['smoking_history'] = self.data['smoking_history'].map({"never": 0, "No Info": 1, "former": 2, "ever": 3})

    def train(self):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
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
        return accuracy, f1
