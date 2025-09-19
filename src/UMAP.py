from Main import dane_skladniki, przepisy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt


def Umap(data, variables, hiperparameter):
    reducer = umap.UMAP(n_components=2, random_state=42, **hiperparameter)
    results = reducer.fit_transform(data)
    plt.figure(figsize=(12, 8))
    plt.scatter(results[:, 0], results[:, 1], s=20)
    for i, podpis in enumerate(variables):
        plt.text(results[i, 0], results[i, 1], podpis, fontsize=6)
    plt.title(f"'Projekcja UMAP zbioru przepisów'")
    plt.grid(alpha=0.3)
    plt.figtext(0.5, 0.01,
                f'Hiperparametry: \nn_neighbors: {hiperparameter['n_neighbors']}  min_dist: {hiperparameter['min_dist']}  metric: {hiperparameter['metric']}',
                ha='center')
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'UMAPs/{nazwa}')
    plt.show()

scaler = StandardScaler()

zeskalowane_dane = scaler.fit_transform(dane_skladniki)
hiperparametry = [
{"n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean"},
{"n_neighbors": 15, "min_dist": 0.0, "metric": "euclidean"},
{"n_neighbors": 30, "min_dist": 0.5, "metric": "euclidean"},
{"n_neighbors": 15, "min_dist": 0.1, "metric": "manhattan"},
{"n_neighbors": 5, "min_dist": 0.1, "metric": "manhattan"},
{"n_neighbors": 5, "min_dist": 0.1, "metric": "jaccard"},
]
for hiperparametr in hiperparametry:
    Umap(zeskalowane_dane, przepisy, hiperparametr)
