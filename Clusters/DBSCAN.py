

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
def DBSCANN():
    print("""
---------------------------------------------------------------------------
DBSCAN
      
      
      """)
    data = pd.read_csv("DataSets/Dataset_spine.csv")
    X = data[["Col1" ,"Col2" ,"Col3" ,"Col4" ,"Col5" ,"Col6" ,"Col7" ,"Col8" ,"Col9" ,"Col10" ,"Col11" ,"Col12" , ]]


    eps_values = [i for i in range(40 , 101)]  # Trying Values from 40 - 100
    min_samples_values = [i for i in range(4 ,11)]  # Trying Values from 1 - 5
    best_davies_bouldin = float('inf')
    best_eps = None
    best_min_samples = None

    print("-------------------------------------------\nSearching for best Epsilon in Range of : [40,100] and min samples in range of [4, 10]  ")
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X)
            dbscan_davies_bouldin = davies_bouldin_score(X, dbscan.labels_)
            if dbscan_davies_bouldin < best_davies_bouldin:
                best_davies_bouldin = dbscan_davies_bouldin
                best_eps = eps
                best_min_samples = min_samples

    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan.fit(X)
    cluster_labels = dbscan.labels_

    silhouette = silhouette_score(X, cluster_labels)
    davies_bouldin = davies_bouldin_score(X, cluster_labels)

    print("-------------------------------------------\nBest Hyperparameters:")
    print("Epsilon (eps):", best_eps)
    print("Min Samples:", best_min_samples)
    print("\n-------------------------------------------\nEvaluation Scores:")
    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Score:", davies_bouldin)
    print("---------------------------------------------")
    return [silhouette ,davies_bouldin ]
