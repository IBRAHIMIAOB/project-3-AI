import numpy as np 
import matplotlib.pyplot as plt
#________________________________Clusters
from Clusters.Kmeans import Kmeans
from Clusters.DBSCAN import DBSCANN
from Clusters.hierarchical_clustering import hierarchical_clustering
#________________________________Classifiers
from classification.RandomForestClassifier import RandomForestClassifierr
from classification.LogisticRegressionClassifier import LogisticRegressionClass
from classification.adaboostclassifier import AdaBoostClassifierr
from classification.DecisionTree import DecisionTreeClassifierr
from classification.GradientBoostingMachines import GradientBoostingClassifierr , predict_diabetes
from classification.HistGradientBoostingClassifier import HistGradientBoostingClassifierr
from classification.IsolationForestclassifier import IsolationForestClassifier
from classification.k_Nearest_Neighbors import kNNClassifierr
from classification.NaiveBayes import NaiveBayesClassifier
from classification.ExtraTreesClassifier import ExtraTreesClassifierr
from classification.Multi_layer_perceptron import MLPClassifierFunc











while True:
    print("""
    1- Clusters 
    2- classification
    3- regressions
    4- Predict Diabetes by inputs
    """)
    Choice = input("Enter your choice : ")
    if Choice == "1":
        Array = []
        Array.append(Kmeans())
        Array.append(DBSCANN())
        Array.append(hierarchical_clustering())
        data = np.array(Array)
        algorithms = ['Kmeans', 'DBSCAN', 'Hierarchical clustering']
        numbers1 = data[:, 0]
        numbers2 = data[:, 1]
        bar_width = 0.35
        r1 = np.arange(len(numbers1))
        r2 = [x + bar_width for x in r1]
        plt.bar(r1, numbers1, color='b', width=bar_width, edgecolor='grey', label='Silhouette Score')
        plt.bar(r2, numbers2, color='r', width=bar_width, edgecolor='grey', label='Davies-Bouldin score')
        plt.xlabel('Algorithms', fontweight='bold')
        plt.xticks([r + bar_width/2 for r in range(len(numbers1))], algorithms)
        plt.ylabel('Numbers')
        plt.title('Comparison of Algorithms')
        plt.legend()
        plt.show()
        print("givin the Plot DBSCAN is The best Clustering algorithm for Spine DataSet since it has High Silhouette Score and low Davies-Bouldin Score \n ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ")
        
        
        
    elif Choice == "2":
        Array = []
        Array.append(RandomForestClassifierr())
        Array.append(LogisticRegressionClass())
        Array.append(NaiveBayesClassifier())
        Array.append(ExtraTreesClassifierr())
        Array.append(HistGradientBoostingClassifierr())
        Array.append(GradientBoostingClassifierr())
        Array.append(IsolationForestClassifier())
        Array.append(kNNClassifierr())
        Array.append(DecisionTreeClassifierr())
        Array.append(AdaBoostClassifierr())
        Array.append(MLPClassifierFunc())
        data = np.array(Array)
        algorithms = ['RandomForest', 'LogisticRegression', 'NaiveBayes' , "ExtraTrees" , "HistGradientBoosting" , "GradientBoosting" , "IsolationForest" ,"kNN" ,"DecisionTree" ,"AdaBoost" ,"Multi layer perceptron"]
        numbers1 = data[:, 0]
        numbers2 = data[:, 1]
        bar_width = 0.35
        r1 = np.arange(len(numbers1))
        r2 = [x + bar_width for x in r1]
        plt.bar(r1, numbers1, color='b', width=bar_width, edgecolor='grey', label='accuracy score')
        plt.bar(r2, numbers2, color='r', width=bar_width, edgecolor='grey', label='F1 score')
        plt.xlabel('Algorithms', fontweight='bold')
        plt.xticks([r + bar_width/2 for r in range(len(numbers1))], algorithms)
        plt.ylabel('Numbers')
        plt.title('Comparison of Algorithms')
        plt.legend()
        plt.show()
        
        
    elif Choice == "3":
        pass
    elif Choice == "4":
        try : 
            gender = int(input("Enter Gender (0 for Male , 1 for Female) :"))
            if not(gender == 1 or gender ==0):
                raise Exception
            age = int(input("Enter The Age :"))
            hypertension = int(input("is there any hypertension ?(0 for no , 1 for yes) :"))
            heart_disease = int(input("is there any heart disease ? ?(0 for no , 1 for yes) :"))
            smoking_history = int(input("Smoking history (\n0 for never \n 1 for No info \n 2 for former \n 3 for ever):"))
            bmi = float(input("enter your BMI : "))
            HbA1c_level = float(input("Enter your HbA1c level range(4 -9): "))
            blood_glucose_level = float(input("Enter your blood glucose level : "))
        except Exception as e:
            print(e.__str__())
            pass
            
        predict_diabetes([gender , age , hypertension , heart_disease , smoking_history , bmi , HbA1c_level , blood_glucose_level])