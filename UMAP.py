from Main import dane_skladniki, przepisy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt


zeskalowane_dane = StandardScaler().fit_transform(dane_skladniki)
hiperparametry = [
{"n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean"},
{"n_neighbors": 15, "min_dist": 0.0, "metric": "euclidean"},
{"n_neighbors": 30, "min_dist": 0.5, "metric": "euclidean"},
{"n_neighbors": 15, "min_dist": 0.1, "metric": "manhattan"},
{"n_neighbors": 5, "min_dist": 0.1, "metric": "manhattan"},
{"n_neighbors": 5, "min_dist": 0.1, "metric": "jaccard"},
]
i = 1
for hiperparametr in hiperparametry:
    reducer = umap.UMAP(n_components=2, random_state=42, **hiperparametr)
    wyniki = reducer.fit_transform(zeskalowane_dane)
    plt.figure(figsize = (12,8))
    plt.scatter(wyniki[:,0], wyniki[:,1], s=20)
    plt.title(f"'Projekcja UMAP zbioru przepisów'")
    plt.grid(alpha=0.3)
    plt.figtext(0.5,0.01, f'Hiperparametry: \nn_neighbors: {hiperparametr['n_neighbors']}  min_dist: {hiperparametr['min_dist']}  metric: {hiperparametr['metric']}', ha= 'center')
    plt.savefig(f'UMAPs/wykres_{i}')
    i+=1
    plt.show()
