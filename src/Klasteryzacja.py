from Main import dane_skladniki, df_binarny_skladniki, df_ograniczony_skladniki, przepisy
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


def hierarchiczna_klasteryzacja(data, variables, metric, method):
        Z = sch.linkage(data, metric=metric, method=method)
        plt.figure(figsize=(17, 8))
        dendrogram = sch.dendrogram(Z, labels=variables.values, leaf_font_size=7)
        plt.title(f'Klasteryzacja hierarchiczna zbioru przepisów.')
        plt.xlabel('Przepisy\n   ')
        plt.ylabel('Odległość')
        plt.figtext(0.5, 0.01, f"Połączenie: \'{method}\'   Odległość: \'{metric}\'", ha='center')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
        plt.savefig(f'Hierarchiczna/{nazwa}.jpg')
        plt.show()

#5A
odleglosci =["euclidean", "cityblock"]
for odleglosc in odleglosci:
    hierarchiczna_klasteryzacja(dane_skladniki, przepisy, odleglosc, 'complete')

#5B
polaczenia = ["complete", "average", "single", "centroid"]
for polaczenie in polaczenia:
    hierarchiczna_klasteryzacja(dane_skladniki, przepisy, 'euclidean', polaczenie)

#5C
datasets = [df_binarny_skladniki, df_ograniczony_skladniki]
for dataset in datasets:
    hierarchiczna_klasteryzacja(dataset, przepisy, 'euclidean', 'complete')
