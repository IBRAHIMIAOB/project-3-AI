


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import matplotlib.pyplot as plt

def Kmeans():
    print("""
---------------------------------------------------------------------------
KMeans
      
      
      """)
    
    data = pd.read_csv("DataSets/Dataset_spine.csv")
    X = data[["Col1" ,"Col2" ,"Col3" ,"Col4" ,"Col5" ,"Col6" ,"Col7" ,"Col8" ,"Col9" ,"Col10" ,"Col11" ,"Col12" , ]]

    k_values = range(1, 11)
    wcss = []

    # Calculate within-cluster sum of squares (WCSS) for each K
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method graph
    plt.plot(k_values, wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()

    print("\n------------------------------------------------\nsince Elbow accurse at k = 4 Then There is 4 clusters\n\n")

    kmeans= KMeans(n_clusters=4)
    kmeans.fit(X)
    print("-------------------------------------------------")
    print(f"KMeans Davies-Bouldin score : {davies_bouldin_score(X , kmeans.labels_)}")
    print(f"KMeans Silhouette Score : {silhouette_score(X , kmeans.labels_)}")
    print("--------------------------------------\n\n")
    return [ silhouette_score(X , kmeans.labels_) , davies_bouldin_score(X , kmeans.labels_) ]