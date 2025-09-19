from Main import df_binarny_skladniki, df_ograniczony_skladniki, dane_skladniki, przepisy
import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from adjustText import adjust_text


def metoda_lokcia(data, max_k):
    inertias = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(inertias, 'o-')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('BezwŁadność (suma kwadratów odległości między centroidami)')
    plt.title('Metoda łokcia')
    plt.grid(True)
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'KMeans/lokiec_{nazwa}.jpg')
    plt.show()


def kmeans_pca(data, k, variables):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(data)
    centroidy = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    pca = PCA(n_components=2)
    x_pca = pca.fit(data).transform(data)
    centroidy_pca = pca.transform(centroidy)
    procent_wariancji = np.round(pca.explained_variance_ratio_ * 100, 1)

    plt.figure(figsize=(16, 8))
    cmap = plt.get_cmap('tab20')
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        plt.scatter(x_pca[labels == lab, 0], x_pca[labels == lab, 1], s=40, alpha=0.6, label=f'Klaster {lab + 1}', c=np.array([cmap(lab % 10)]))

    plt.scatter(centroidy_pca[:, 0], centroidy_pca[:, 1], s=100, marker='X', edgecolor='k', label='Centroidy', c=[cmap(i % 10) for i in range(len(centroidy_pca))])
    podpisy = [plt.text(x_pca[i,0], x_pca[i, 1], variables[i], fontsize=8) for i in range(len(variables))]
    adjust_text(podpisy, arrowprops=dict(arrowstyle='-', color='gray', lw=0.3))
    plt.xlabel(f'PC1 - {procent_wariancji[0]}%')
    plt.ylabel(f'PC2 - {procent_wariancji[1]}%')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f'KMeans (k = {k}) + PCA (2 składowe) \nBezwładność = {inertia:.2f}')
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'KMeans/klasteryzacja_{nazwa}.jpg')
    plt.show()

def kmeans_mca(data, k, variables):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(data)
    inertia = kmeans.inertia_

    mca = prince.MCA(n_components=2)
    results = mca.fit(data).transform(data).values
    centroidy_mca = np.array([results[labels == lab].mean(axis=0) for lab in range(k)])
    procent_wariancji_1 = round(mca.percentage_of_variance_[0], 2)
    procent_wariancji_2 = round(mca.percentage_of_variance_[1], 2)

    plt.figure(figsize=(16, 8))
    cmap = plt.get_cmap('tab20')
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        plt.scatter(results[labels == lab, 0], results[labels == lab, 1], label=f'Klaster {lab + 1}', c=np.array([cmap(lab % 10)]))

    plt.scatter(centroidy_mca[:, 0], centroidy_mca[:, 1], s=100, marker='X', edgecolor='k', label='Centroidy', c=[cmap(i % 10) for i in range(len(centroidy_mca))])
    podpisy = [plt.text(results[i, 0], results[i, 1], variables[i], fontsize=8) for i in range(len(variables))]
    adjust_text(podpisy, arrowprops=dict(arrowstyle='-', color='gray', lw=0.3))
    plt.xlabel(f"PC1 - {procent_wariancji_1}%")
    plt.ylabel(f"PC2 - {procent_wariancji_2}%")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f'KMeans (k = {k}) + MCA (2 składowe) \nBezwładność = {inertia:.2f}')
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'KMeans/klasteryzacja_{nazwa}.jpg')
    plt.show()

scaler = StandardScaler()

skalowane_dane = scaler.fit_transform(dane_skladniki)
metoda_lokcia(skalowane_dane, 100)
kmeans_pca(skalowane_dane, 10, przepisy)

skalowane_binary = pd.DataFrame(scaler.fit_transform(df_binarny_skladniki), columns=df_binarny_skladniki.columns, index=df_binarny_skladniki.index)
metoda_lokcia(skalowane_binary, 100)
kmeans_mca(skalowane_binary,9, przepisy)

skalowane_ograniczony = scaler.fit_transform(df_ograniczony_skladniki)
metoda_lokcia(skalowane_ograniczony, 100)
kmeans_pca(skalowane_ograniczony, 7, przepisy)
