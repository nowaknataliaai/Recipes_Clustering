from Main import dane_skladniki, df_binarny_skladniki, df_ograniczony_skladniki, dane, df_ograniczony
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#5A
odleglosci =["euclidean", "cityblock"] #cityblock to manhattan

for odleglosc in odleglosci:
    Z = sch.linkage(dane_skladniki, metric=odleglosc, method='complete')
    plt.figure(figsize=(17,8))
    dendrogram = sch.dendrogram(Z, labels=dane["Nazwa"].values, leaf_font_size=7)
    plt.title(f'a. Klasteryzacja hierarchiczna dla odległości \'{odleglosc}\' zbioru przepisów')
    plt.xlabel('Przepisy\n   ')
    plt.ylabel('Odległość')
    plt.figtext(0.5, 0.01, f"\nPołączenie: \'complete\'", ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'Odleglosci/Odleglosc_{odleglosc}.jpg')
    plt.show()

#5B
polaczenia = ["complete", "average", "single", "centroid"] #average - UPGMA, centroid - UPGMC

for polaczenie in polaczenia:
    Z = sch.linkage(dane_skladniki, metric='euclidean', method=polaczenie)
    plt.figure(figsize = (17,8))
    dendrogram = sch.dendrogram(Z, labels=dane["Nazwa"].values, leaf_font_size=7)
    plt.title(f'b. Klasteryzacja hierarchiczna z połączeniem {polaczenie} zbioru przepisów.')
    plt.xlabel('Przepisy\n   ')
    plt.ylabel('Odległość')
    plt.figtext(0.5, 0.01, f"Odległość: \'euclidean\'", ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'Polaczenia/Polaczenie_{polaczenie}.jpg')
    plt.show()

#5C
dataset = {'binary': df_binarny_skladniki, 'ograniczony': df_ograniczony_skladniki}

for nazwa, df in dataset.items():
    Z = sch.linkage(df, metric='euclidean', method='complete')
    plt.figure(figsize = (17,8))
    if nazwa == 'binary':
        plt.title(f'c. Klasteryzacja hierarchiczna binarnego zbioru przepisów.')
        dendrogram = sch.dendrogram(Z, labels=dane["Nazwa"].values, leaf_font_size=7)
    else:
        dendrogram = sch.dendrogram(Z, labels=df_ograniczony["Nazwa"].values, leaf_font_size=7)
        plt.title(f'c. Klasteryzacja hierarchiczna zbioru przepisów dla wybranych 5 składników.')
    plt.xlabel('Przepisy\n   ')
    plt.ylabel('Odległość')
    plt.figtext(0.5, 0.01, f"Połączenie: \'complete\' Odległość: \'euclidean\'", ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'Wykresy_roznych_df/Wykres_{nazwa}.jpg')
    plt.show()