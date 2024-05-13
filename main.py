import numpy as np 
import matplotlib.pyplot as plt
#________________________________Clusters
from Clusters.Kmeans import Kmeans
from Clusters.DBSCAN import DBSCANN
from Clusters.hierarchical_clustering import hierarchical_clustering
#________________________________classiers
from classifiction.RandomForestClassifier import RandomForestClassifierr
while True: 
    print("""
1- Clusters 
2- classification
3- regressions""")
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

        # Create bar plot
        plt.bar(r1, numbers1, color='b', width=bar_width, edgecolor='grey', label='Silhouette Score')
        plt.bar(r2, numbers2, color='r', width=bar_width, edgecolor='grey', label='Davies-Bouldin score')

        # Add xticks on the middle of the group bars
        plt.xlabel('Algorithms', fontweight='bold')
        plt.xticks([r + bar_width/2 for r in range(len(numbers1))], algorithms)

        # Add labels and title
        plt.ylabel('Numbers')
        plt.title('Comparison of Algorithms')
        plt.legend()

        # Show plot
        plt.show()
        
        print("givin the Plot DBSCAN is The best Clustering algorithm for Spine DataSet since it has High Silhouette Score and low Davies-Bouldin Score \n ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ")
        
        
        pass
    elif Choice == "2":
        Array2 = []
        Array2.append(RandomForestClassifierr())
        
        pass
    elif Choice == "3":
        pass