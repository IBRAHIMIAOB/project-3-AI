

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
def hierarchical_clustering():
    
    print("""
---------------------------------------------------------------------------
hierarchical clustering
      
      
        """)
        
    data = pd.read_csv("DataSets/Dataset_spine.csv")
    X = data[["Col1" ,"Col2" ,"Col3" ,"Col4" ,"Col5" ,"Col6" ,"Col7" ,"Col8" ,"Col9" ,"Col10" ,"Col11" ,"Col12" , ]]

    best_score = -1
    best_n_clusters = None
    best_linkage = None
    print("-------------------------------------------\nSearching for best cluster in Range of : [4,10] ")

    for n_clusters in range(4, 11):  # Adjust the range according to your needs
        for linkage in ['ward', 'complete', 'average', 'single']:  # Try different linkage methods
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            hierarchical.fit(X)
            silhouette = silhouette_score(X, hierarchical.labels_)
            davies_bouldin = davies_bouldin_score(X, hierarchical.labels_)
            
            if silhouette > best_score:
                best_score = silhouette
                best_n_clusters = n_clusters
                best_linkage = linkage

    best_hierarchical = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_linkage)
    best_hierarchical.fit(X)

    best_silhouette = silhouette_score(X, best_hierarchical.labels_)
    best_davies_bouldin = davies_bouldin_score(X, best_hierarchical.labels_)

    print("-------------------------------------------\nBest Hyperparameters:")
    print("Number of Clusters:", best_n_clusters)
    print("Linkage Method:", best_linkage)
    print("-------------------------------------------\n\nEvaluation Scores:")
    print("Silhouette Score:", best_silhouette)
    print("Davies-Bouldin Score:", best_davies_bouldin)
    print("---------------------------------------------")
    return [best_silhouette , best_davies_bouldin]